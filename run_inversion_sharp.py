import numpy as np
import simpeg
import dill
from types import SimpleNamespace
from discretize import TensorMesh
from simpeg import maps
from pymatsolver import PardisoSolver
import simpeg.electromagnetics.time_domain as tdem
from simpeg.electromagnetics.utils.em1d_utils import get_vertical_discretization

input_data_dict = dill.load(open("input_gmd_4.pik", "rb"))
inp = SimpleNamespace(**input_data_dict)

source_locations = np.c_[inp.topography[:,0], inp.topography[:,1], inp.topography[:,2]+inp.source_heights]
receiver_locations = np.c_[inp.topography[:,0]+inp.rx_coil_position[0], inp.topography[:,1],  inp.topography[:,2]+inp.source_heights-inp.rx_coil_position[2]]
n_sounding = source_locations.shape[0]

source_list = []
receiver_orientation = 'z'
source_orientation = 'z'
for i_sounding in range(n_sounding):    
    waveform_ch1 = tdem.sources.PiecewiseLinearWaveform(inp.time_input_currents_ch1, inp.input_currents_ch1)
    waveform_ch2 = tdem.sources.PiecewiseLinearWaveform(inp.time_input_currents_ch2, inp.input_currents_ch2)
    source_location = source_locations[i_sounding, :]
    receiver_location = receiver_locations[i_sounding, :]

    # Receiver list

    dbzdt_receiver_ch1 = tdem.receivers.PointMagneticFluxTimeDerivative(
            receiver_location, inp.times_ch1, "z",
            bw_cutoff_frequency=inp.bw_cutoff_frequency_ch1,
            bw_power=inp.bw_power_ch1,
            lp_cutoff_frequency=inp.lp_cutoff_frequency_ch1,
            lp_power=inp.lp_power_ch1,          
    )

    dbzdt_receiver_ch2 = tdem.receivers.PointMagneticFluxTimeDerivative(
            receiver_location, inp.times_ch2, "z",
            bw_cutoff_frequency=inp.bw_cutoff_frequency_ch2,
            bw_power=inp.bw_power_ch2,
            lp_cutoff_frequency=inp.lp_cutoff_frequency_ch2,
            lp_power=inp.lp_power_ch2,          
    )

    # Make a list containing all receivers even if just one

    # Must define the transmitter properties and associated receivers

    source_list.append(tdem.sources.MagDipole(
        [dbzdt_receiver_ch1],
        location=source_location,
        waveform=waveform_ch1,
        orientation=source_orientation,
        i_sounding=i_sounding,
    )
    )
    source_list.append(tdem.sources.MagDipole(
        [dbzdt_receiver_ch2],
        location=source_location,
        waveform=waveform_ch2,
        orientation=source_orientation,
        i_sounding=i_sounding,
    )
    )


survey = tdem.Survey(source_list)
hz = np.r_[inp.thickness, inp.thickness[-1]]

n_layer = len(hz)
nP = n_sounding * n_layer
sigma_map = maps.ExpMap(nP=nP)

simulation = tdem.Simulation1DLayeredStitched(
    survey=survey, 
    thicknesses=inp.thickness, 
    sigmaMap=sigma_map,
    topo=inp.topography, 
    parallel=True, 
    n_cpu=60, 
    verbose=False, 
    solver=PardisoSolver,
)

n_time = inp.times_ch1.size + inp.times_ch2.size

noise_floor = 0.
ignore_value = 9999.
dobs = -np.hstack((inp.data_ch1, inp.data_ch2)).flatten()
dobs[np.isnan(dobs)] = ignore_value
inds_active_dobs = dobs != 9999.
relative_error = np.hstack((inp.data_std_ch1, inp.data_std_ch2)).flatten()
uncertainties = relative_error*np.abs(dobs) + noise_floor
uncertainties[~inds_active_dobs] = np.inf
# Create data ojbect
data_object = simpeg.data.Data(survey, dobs=dobs, standard_deviation=uncertainties)
dmis = simpeg.data_misfit.L2DataMisfit(simulation=simulation, data=data_object)

print (f"Percentage of the active data = {inds_active_dobs.sum()}/{len(dobs)}={inds_active_dobs.sum()/len(dobs)*100:.0f}%")

from simpeg.electromagnetics.utils.em1d_utils import set_mesh_1d
import scipy
from discretize import SimplexMesh
from simpeg.regularization.laterally_constrained import LaterallyConstrained

tri = scipy.spatial.Delaunay(inp.topography[:,:2])
mesh_radial = SimplexMesh(tri.points, tri.simplices)
mesh_vertical = set_mesh_1d(hz)
mesh_reg = [mesh_radial, mesh_vertical]

def get_active_edge_indices_with_distance(mesh_radial, mesh_vertical, maximum_distance=1000):
    nz = mesh_vertical.n_cells
    edge_lengths = mesh_radial.edge_lengths
    inds = edge_lengths < maximum_distance
    indActiveEdges = np.tile(inds.reshape([-1,1]), nz).flatten()
    return inds, indActiveEdges

inds, indActiveEdges = get_active_edge_indices_with_distance(
    mesh_radial, mesh_vertical, maximum_distance=500.
)

reg = LaterallyConstrained(
    mesh_reg, 
    mapping=simpeg.maps.IdentityMap(nP=nP),
    alpha_s = 0.,
    alpha_r = 1.,
    alpha_z = 1./2.,
    active_edges=indActiveEdges,
    norms=np.array([2., 0., 0.])
)
reg.gradient_type = 'components'
opt = simpeg.optimization.ProjectedGNCG(maxIter=20, maxIterCG=50)
invProb = simpeg.inverse_problem.BaseInvProblem(dmis, reg, opt)
beta = simpeg.directives.BetaSchedule(coolingFactor=2, coolingRate=1)
betaest = simpeg.directives.BetaEstimate_ByEig(beta0_ratio=1.)
target = simpeg.directives.TargetMisfit(chifact=1)
precond = simpeg.directives.UpdatePreconditioner()
save_model_dict = simpeg.directives.SaveOutputDictEveryIteration()
save_model_dict.outDict = {}

update_irls = simpeg.directives.Update_IRLS(
    coolingFactor=2,
    coolingRate=1,
    f_min_change=1e-5,
    max_irls_iterations=40,
    chifact_start=0.4,
    chifact_target=0.4,    
)
update_irls.coolEpsFact = 2.

inv = simpeg.inversion.BaseInversion(
    invProb, 
    directiveList=[
        update_irls,
        precond,
        betaest,
        save_model_dict
    ]
)

invProb.counter = opt.counter = simpeg.utils.Counter()
opt.LSshorten = 0.5
opt.remember('xc')
m0 = np.ones(nP) * np.log(1./10.)
mest = inv.run(m0)

import dill
dill.dump(save_model_dict.outDict, open("./output/inversion_results_sharp_precond_comp.pik", "wb"))
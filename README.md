## Inversion of Kansas AEM data using `SimPEG`

## Files: 
- `01_load_aem_data.ipynb`: Jupyter notebook to load the AEM data and create associated files to run the next notebook.
- `02_create_input_data_dict.ipynb`: Jupyter nobook to create the input dictionary used in the following python scripts.
- `03_depth_to_bedrock.ipynb`: Jupyter nobook to estimate depth to bedrock from an inverted resistivity model.
- `04_load_aem_resistivity.ipynb`: Jupyter nobook to load inverted resistivity models, and create `.csv` files.
- `run_inversion_smooth.py`: python script for running a smooth inversion.
- `run_inversion_sharp.py`: python script for running a sharp inversion.

## Installation of Python Packages

### Step 1: `Pip` install

`pip install simpeg pyarrow fastparquet pandas libaarhusxyz`

### Step 2: Use a specific branch of simpeg

`git clone https://github.com/simpeg/simpeg.git`

`cd simpeg`

`git checkout -f em1d_stitched_v022`

`pip install -e .`



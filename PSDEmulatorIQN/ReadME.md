This repositiory is for generating a set of pulse shape simulation and mage-post-proc data for use in a series of IQN (Implicit Quantile Network) code.

The process should be as follows:
> simulate detector - physics using mage (not included in this repo yet)  
> run mage-post-proc to cluster mage files (not included here yet)  
> run process_mpp_output.py (with the relevent flags)  
> run a PSS simualtion (not included here yet)  
> run process_pss_output.py (with relevent flags)  
> run process_ld_data.py (with relevent flags)  

If you run the code witht he default names for the produced files, then the notebook iqn_emulator_test_notebook.ipynb should work without any changes. Note that in the data directory there is included the first pass at the data.

There is a environment.yml file that should contain the software to run the code. To create a conda environment with conda one can run:
> conda env create -f environment.yml

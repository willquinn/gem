This repositiory is for generating a set of pulse shape simulation and mage-post-proc data for use in a series of IQN (Implicit Quantile Network) code.

The process should be as follows:
- simulate detector - physics using mage (not included in this repo yet)
- run mage-post-proc to cluster mage files (not included here yet)
- run process_mpp_output.py (with the relevent flags)
- run a PSS simulation (not included here yet)
- run process_pss_output.py (with relevent flags)
- run process_ml_data.py (with relevent flags)

Originally this code was written in tensorflow but we ran into versioning issues so we have moved to pytorch. The required version of python is 3.11.9 (3.11 mainly) for pre-commit reasons (3.12 doesn't seem to work).

On MAC I recommend using pyenv
```
$ brew install pyenv-virtualenv  
$ pyenv install 3.11.9  
$ pyenv virtualenv 3.11.9 myproject-env  
$ pyenv activate myproject-env
$ pip install -r requirements.txt
```

The main code is found in quantile_network_pytorch.py. The pytorch_train_iqn.py is for training the IQN whose architecture is defined in inputs/model_params.yaml as well as some other sampling constants. I have tried to keep the directory structure clean using logs. To use this code I suggest editing the inputs/model_params.yaml file to set the names and locations of output files you require. Then run:
``` 
$ python pytorch_train_iqn.py --config inputs/model_params.yaml --plotting True
```

All data splitting is handled in that file and then the indexes chosen from training and testing is stored in the output directory (which should be generated for you).

The power of the IQN is being able to sample from its learned CDFs. To do this run:
```
$ python iqn_sampling_pytorch.py --config inputs/model_params.yaml --plotting True
```

This should save an output file which helps with speed if you increase the sampling number (doesn't seem to be linear).

Some useful links for IQN: 
 - [Jets paper](https://arxiv.org/pdf/2111.11415)
 - [Original code repo](https://github.com/alpha-davidson/IQNs-for-Jets)


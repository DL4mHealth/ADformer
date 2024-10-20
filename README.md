# ADformer: A Multi-Granularity Transformer for EEG-Based Alzheimer’s Disease Assessment





### Processed data
The processed data should be put into `dataset/DATA_NAME/` so that each subject file can be located by `dataset/DATA_NAME/Feature/feature_ID.npy`, and the label file can be located by `dataset/DATA_NAME/Label/label.npy`.  

The processed datasets can be manually downloaded at the following links.
* ADSZ dataset: https://drive.google.com/file/d/1jkqawkx1v_vlf5zFOSgIh2LJs_OJMrvf/view?usp=drive_link
* APAVA dataset: https://drive.google.com/file/d/1ARjYLQ-tSEu0r3bffnLn3tCWdUiz4qXI/view?usp=drive_link
* ADFD dataset: https://drive.google.com/file/d/1ACYmO2ai9ad4C2zP3-qJFMjDpTexD321/view?usp=drive_link

Since other three datasets in the paper are private datasets, we do not provide a download link here. 



## Requirements  
  
The recommended requirements are specified as follows:  
* Python 3.8  
* Jupyter Notebook  
* einops==0.4.0
* matplotlib==3.7.0
* numpy==1.23.5
* pandas==1.5.3
* patool==1.12
* reformer-pytorch==1.4.4
* scikit-learn==1.2.2
* scipy==1.10.1
* sktime==0.16.1
* sympy==1.11.1
* torch==2.0.0
* tqdm==4.64.1
* wfdb==4.1.2
* neurokit2==0.2.9
* mne==1.6.1 
* natsort~=8.4.0

The dependencies can be installed by:  
```bash  
pip install -r requirements.txt
```



## Run Experiments
Before running, make sure you have all the processed datasets put under `dataset/`. For Linux users, run each method's shell script in `scripts/classification/`. 
You could also run all the experiments by running the `meta-run.py/` file, which the method included in _skip_list_ will not be run.
For Windows users, see jupyter notebook files. All the experimental scripts are provided cell by cell. 
The gpu device ids can be specified by setting command line `--devices` (e,g, `--devices 0,1,2,3`). 
You also need to change the visible gpu devices in script file by setting `export CUDA_VISIBLE_DEVICES` (e,g, `export CUDA_VISIBLE_DEVICES=0,1,2,3`). 
The gpu devices specified by commend line should be a subset of visible gpu devices.


After training and evaluation, the saved model can be found in`checkpoints/classification/`; 
and the results can be found in  `results/classification/`. 
You can modify the parameters by changing the command line. 
The meaning and explanation of each parameter in command line can be found in `run.py` file. 
Since APAVA is the smallest dataset and faster to run, 
it is recommended to run and test our code with the APAVA dataset to get familiar with the framework.



## Acknowledgement

This codebase is constructed based on the repo: [Time-Series-Library](https://github.com/thuml/Time-Series-Library).
Thanks a lot for their amazing work on implementing state-of-arts time series methods!

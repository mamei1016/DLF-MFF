# DLF-MFF

A deep learning framework for predicting molecular property based on multi-type features fusion

we propose a fusion model named DLF-MFF, which integrates four types of features from molecular fingerprints, 2D molecular graph, 3D molecular graph 
and molecular image to predict molecular property. Moreover, DLF-MFF is applied to identify potential anti-SARS-CoV-2 inhibitor from 2500 
drugs. 

Paper: [https://doi.org/10.1016/j.compbiomed.2023.107911]

# **Command**

### **1. Train**
Use train.py

Args:
  - data_path : The path of input CSV file. *E.g. input.csv*
  - dataset_type : The type of dataset. *E.g. classification  or  regression*
  - save_path : The path to save output model. *E.g. model_save*
  - log_path : The path to record and save the result of training. *E.g. log*

E.g.

`python train.py  --data_path data/test.csv  --dataset_type classification  --save_path model_save  --log_path log`


### **2. Hyperparameters Optimization**
Use hyper_opti.py

Args:
  - data_path : The path of input CSV file. *E.g. input.csv*
  - dataset_type : The type of dataset. *E.g. classification  or  regression*
  - save_path : The path to save output model. *E.g. model_save*
  - log_path : The path to record and save the result of hyperparameters optimization. *E.g. log*

E.g.

`python hyper_opti.py  --data_path data/test.csv  --dataset_type classification  --save_path model_save  --log_path log`


---
# **Data**

We provide the three public benchmark datasets used in our study: *<Data.rar>*

Or you can use your own dataset:
### 1. For training
The dataset file should be a **CSV** file with a header line and label columns. E.g.
'''
SMILES,BT-20
O(C(=O)C(=O)NCC(OC)=O)C,0
FC1=CNC(=O)NC1=O,0
...

## Citation

If you find this repo useful, please cite our paper:

Mei Ma, Xiujuan Lei. A deep learning framework for predicting molecular property based on multi-type features fusion.
Computers in Biology and Medicine,2024,(169):107911.
https://doi.org/10.1016/j.compbiomed.2023.107911.
(https://www.sciencedirect.com/science/article/pii/S0010482523013768)}

## Contact
If you have any question, please contact us: mm1016@qhnu.edu.cn
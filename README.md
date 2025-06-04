# This is the source code for paper: Disentangled Graph Prompting for Out-Of-Distribution Detection. 


### Requirements:

- torch-geometric==2.0.4
- torch-scatter==2.0.93
- torch-sparse==0.6.15
- numpy==1.21.2
- pandas==1.3.0
- python==3.9.15
- scikit-learn=1.0.2
- scipy==1.9.3
- torch==1.11.0
- torchvision==0.12.0

### Training:

Run DGP-GCL:

```  
python DGP_GCL.py --DS xxxx --model_type dgp-gcl --lr xxxx --aug xxxx --DS_pair xxxx --lambda_ xxxx --gamma xxxx --alpha_1 xxxx --alpha_2 xxxx --dgp_lr xxxx
```

Run DGP-Sim:

```  
python DGP_Sim.py --DS xxxx --model_type dgp-sim  --lr xxxx --eta xxxx --DS_pair xxxx --lambda_ xxxx --gamma xxxx --alpha_1 xxxx --alpha_2 xxxx --dgp_lr xxxx
```

To run pre-trained GNNs or their fine-tuned versions, simply modify the --model_type parameter.


We also provide the code to search hyper-parameters, you can use the following command (for TOX21-SIDER dataset) to run it:
```
bash run_grid_search.sh
```
           








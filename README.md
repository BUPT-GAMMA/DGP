**DGP (Disentangled Graph Prompting)** is a prompt-based framework for **graph out-of-distribution (OOD) detection** under the pre-training paradigm.

Instead of fine-tuning the entire GNN encoder, DGP freezes a pre-trained graph encoder and learns **two complementary prompt graphs** to capture fine-grained in-distribution patterns:

- **Class-specific prompt graph** focus on discriminative structures for classification
- **Class-agnostic prompt graph** capture shared patterns across ID graphs

By jointly modeling these two perspectives, DGP enhances the separation between ID and OOD samples and achieves strong performance with high efficiency.

- Paper: https://arxiv.org/abs/2603.29644  
- Venue: TKDE

## ✨ Key Features

- 🔍 **Fine-grained pattern modeling**  
  Disentangles ID patterns into class-specific and class-agnostic components for better OOD detection.

- 🧊 **Pre-training + prompting paradigm**  
  Reuses pre-trained GNN encoders without full fine-tuning, reducing training cost.

- 🧠 **Graph structure-aware prompting**  
  Learns edge-level reweighting to construct prompt graphs instead of modifying node features.

- ⚡ **Efficient and scalable**  
  Avoids end-to-end retraining and achieves significant speedups over prior methods.

## 🧭 Why DGP

Most existing graph OOD methods rely on end-to-end training and implicitly learn ID patterns without explicit supervision.  
DGP takes a different approach: it leverages pre-trained representations and explicitly models **multiple views of ID structure**, enabling more robust and interpretable OOD detection.

By combining **prompt learning** with **graph representation learning**, DGP provides a lightweight yet powerful alternative to traditional methods.

## 🔧 Requirements

We recommend using Python 3.9 with the following dependencies:

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

## 🚀 Training:

### ▶ Run DGP-GCL:

```  
python DGP_GCL.py \
  --DS <dataset> \
  --model_type dgp-gcl \
  --lr <learning_rate> \
  --aug <augmentation_type> \
  --DS_pair <dataset_pair> \
  --lambda_ <lambda> \
  --gamma <gamma> \
  --alpha_1 <alpha1> \
  --alpha_2 <alpha2> \
  --dgp_lr <dgp_lr>
```

### ▶ Run DGP-Sim:

```  
python DGP_Sim.py \
  --DS <dataset> \
  --model_type dgp-sim \
  --lr <learning_rate> \
  --eta <eta> \
  --DS_pair <dataset_pair> \
  --lambda_ <lambda> \
  --gamma <gamma> \
  --alpha_1 <alpha1> \
  --alpha_2 <alpha2> \
  --dgp_lr <dgp_lr>
```

### 🔁 Other Variants

To run pre-trained GNNs or their fine-tuned versions, simply modify the --model_type parameter.

### 🔍 Hyper-parameter Search

We also provide the code to search hyper-parameters, you can use the following command (for TOX21-SIDER dataset) to run it:
```
bash run_grid_search.sh
```
           

## 📄 Citation
```bibtex
@article{yang2026disentangled,
  title   = {Disentangled Graph Prompting for Out-of-Distribution Detection},
  author  = {Cheng Yang and Yu Hao and Qi Zhang and Chuan Shi},
  journal = {arXiv preprint arXiv:2603.29644},
  year    = {2026}
}
```

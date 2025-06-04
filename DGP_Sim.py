import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch_scatter import scatter
import sklearn
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from scipy import interpolate
from scipy.spatial.distance import cdist

from losses import *
from gin import Encoder
from model import PriorDiscriminator, FF
from arguments import arg_parse
from data_loader import get_ood_dataset

class PromptGenerator(nn.Module):
    def __init__(self, emb_dim, hidden_dim=32, num_classes=2):
        super(PromptGenerator, self).__init__()
        self.input_dim = emb_dim
        self.num_classes = num_classes

        self.mlp = nn.Sequential(
            nn.Linear(2 * self.input_dim, 4 * hidden_dim),
            nn.BatchNorm1d(4 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4 * hidden_dim, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, node_emb, edge_index):
        src, dst = edge_index[0], edge_index[1]
        src_emb = node_emb[src]
        dst_emb = node_emb[dst]

        edge_emb = torch.cat([src_emb, dst_emb], dim=1)

        edge_weights = self.mlp(edge_emb)

        edge_weights = torch.nan_to_num(edge_weights, nan=0.5)
        return edge_weights


class DGP(nn.Module):
    def __init__(self, gnn_encoder, hidden_dim, num_classes, backbone_type):
        super(DGP, self).__init__()
        self.gnn_encoder = gnn_encoder
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.backbone_type = backbone_type

        if hasattr(gnn_encoder, 'embedding_dim'):
            self.emb_dim = gnn_encoder.embedding_dim
        else:
            self.emb_dim = hidden_dim * args.num_gc_layers

        if backbone_type in ["DGP-GCL", "DGP-Sim"]:
            for param in self.gnn_encoder.parameters():
                param.requires_grad = False

        self.prompt_generator_specific = None
        self.prompt_generator_agnostic = None

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 3),
            nn.BatchNorm1d(hidden_dim * 3),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

        self.l2_lambda = 0.1
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index, batch, gamma=1.0, edge_weight=None, testing=False):
        if self.prompt_generator_specific is None:
            self.prompt_generator_specific = PromptGenerator(
                emb_dim=self.emb_dim,
                hidden_dim=self.hidden_dim,
                num_classes=self.num_classes,
            ).to(device)
            
            self.prompt_generator_agnostic = PromptGenerator(
                emb_dim=self.emb_dim,
                hidden_dim=self.hidden_dim,
                num_classes=self.num_classes,
            ).to(device)

        _, node_emb = self.gnn_encoder(x, edge_index, batch, notfinal=True)

        specific_weights = self.prompt_generator_specific(node_emb, edge_index)
        agnostic_weights = self.prompt_generator_agnostic(node_emb, edge_index)

        h_specific = self.gnn_encoder(x, edge_index, batch, edge_weight=specific_weights)
        h_agnostic = self.gnn_encoder(x, edge_index, batch, edge_weight=agnostic_weights)

        if isinstance(h_specific, tuple):
            h_specific = h_specific[0]
        if isinstance(h_agnostic, tuple):
            h_agnostic = h_agnostic[0]

        if testing:
            return h_specific, h_agnostic
            
        else:
            z_specific = self.predictor(h_specific)
            z_agnostic = self.predictor(h_agnostic)
            
            return h_specific, z_specific, h_agnostic, z_agnostic

    def loss_disentangle(self, z_specific, y, z_agnostic, lambda_):
        loss_specific = F.cross_entropy(z_specific, y)
        uniform_target = torch.ones_like(z_agnostic) / z_agnostic.size(1)
        log_z_agnostic = F.log_softmax(z_agnostic, dim=1)
        loss_agnostic = F.kl_div(log_z_agnostic, uniform_target, reduction='batchmean')

        l2_reg = 0.0
        for param in self.parameters():
            if param.requires_grad:
                l2_reg += torch.norm(param)

        loss = loss_specific + lambda_ * loss_agnostic + self.l2_lambda * l2_reg
        
        return loss

    def loss_final(self, h_specific, h_agnostic, alpha1, alpha2, h_original):
        dist_specific = ssd(h_original, h_specific).mean()
        dist_agnostic = ssd(h_original, h_agnostic).mean()

        return alpha1 / (dist_specific + 1e-8) + alpha2 / (dist_agnostic + 1e-8)

def metric(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores.detach().cpu())
    precision, recall, _ = precision_recall_curve(labels, scores.detach().cpu())
    fpr95 = float(interpolate.interp1d(tpr, fpr)(0.95))
    # FPR90 = fpr[np.argwhere(tpr >= 0.90)[0]]
    aupr = auc(recall, precision)
    # aupr = 0
    auroc = auc(fpr, tpr)
    return auroc, fpr95, aupr
    
def ssd(ftrain, ftest, clusters=1, th=True):
    if clusters == 1:
        return get_scores_one_cluster(ftrain, ftest, th)
    else:
        if th == True:
            ypred, cluster_centers = kmeans(X=ftrain, num_clusters=clusters, distance='euclidean')
        else:
            ypred = sklearn.cluster.KMeans(n_clusters=clusters).fit_predict(ftrain)
        return get_scores_multi_cluster(ftrain, ftest, ypred, th)
        

def get_scores_one_cluster(ftrain, ftest, th=True):
    if th == True:
        mean = torch.mean(ftrain, dim=0, keepdims=True)
        std = torch.std(ftrain, dim=0, keepdims=True) + 1e-8
        ftrain_norm = (ftrain - mean) / std
        ftest_norm = (ftest - mean) / std

        cov = torch.cov(ftrain_norm.T)
        cov += 1e-6 * torch.eye(cov.shape[0], device=cov.device)
        con_inv = torch.linalg.pinv(cov)

        diff = ftest_norm - torch.mean(ftrain_norm, dim=0, keepdims=True)
        dtest = torch.sum(diff @ con_inv * diff, dim=1)
    else:
        mean = np.mean(ftrain, axis=0, keepdims=True)
        std = np.std(ftrain, axis=0, keepdims=True) + 1e-8
        ftrain_norm = (ftrain - mean) / std
        ftest_norm = (ftest - mean) / std
        
        cov = np.cov(ftrain_norm.T)
        cov += 1e-6 * np.eye(cov.shape[0])
        con_inv = np.linalg.pinv(cov)
        
        diff = ftest_norm - np.mean(ftrain_norm, axis=0, keepdims=True)
        dtest = np.sum(diff @ con_inv * diff, axis=1)
    return dtest

def get_scores_multi_cluster(ftrain, ftest, ypred, th=True):
    if th == True:
        xc = [ftrain[ypred == i] for i in torch.unique(ypred)]
        dtest = [
            torch.sum(
                (ftest - torch.mean(x, axis=0, keepdims=True))
                * (
                    torch.mm(
                        torch.linalg.pinv(torch.cov(x.T)),
                        (ftest - torch.mean(x, dim=0, keepdims=True)).T
                    )
                ).T,
                dim=-1,
            )
            for x in xc
        ]
        dtest, _ = torch.min(torch.vstack(dtest), dim=0)
    else:
        xc = [ftrain[ypred == i] for i in np.unique(ypred)]
        dtest = [
            np.sum(
                (ftest - np.mean(x, axis=0, keepdims=True))
                * (
                    np.matmul(
                        np.linalg.pinv(np.cov(x.T)),
                        (ftest - np.mean(x, dim=0, keepdims=True)).T
                    )
                ).T,
                dim=-1,
            )
            for x in xc
        ]
        dtest = torch.min(np.vstack(dtest), dim=0)
    return dtest
    

class GcnInfomax(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(GcnInfomax, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

    self.local_d = FF(self.embedding_dim)
    self.global_d = FF(self.embedding_dim)

    if self.prior:
        self.prior_d = PriorDiscriminator(self.embedding_dim)

    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


  def forward(self, x, edge_index, batch, num_graphs):

    # batch_size = data.num_graphs
    if x is None:
        x = torch.ones(batch.shape[0]).to(device)

    y, M = self.encoder(x, edge_index, batch)
    
    g_enc = self.global_d(y)
    l_enc = self.local_d(M)

    mode='fd'
    measure='JSD'
    local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)
 
    if self.prior:
        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma
    else:
        PRIOR = 0
    
    return local_global_loss + PRIOR

class simclr(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(simclr, self).__init__()
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior
    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)
    self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))
    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


  def forward(self, x, edge_index, batch, num_graphs, edge_weight=None):
    # batch_size = data.num_graphs
    if x is None:
        x = torch.ones(batch.shape[0]).to(device)
    y, M = self.encoder(x, edge_index, batch, edge_weight=edge_weight)
    y = self.proj_head(y)
    return y

  def loss_cal(self, x, x_aug):
    T = 0.2
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1) 
    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss

def cal_score(args, edge_logits, batch, eva=False):
    bias = 0.0001
    eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
    edge_score = torch.log(eps) - torch.log(1 - eps)
    edge_score = edge_score.to(device)
    edge_score = (edge_score + edge_logits)
    batch_aug_edge_weight = torch.sigmoid(edge_score).squeeze()
    if eva:
        return batch_aug_edge_weight
    row, col = batch.edge_index
    edge_batch = batch.batch[row]
    uni, edge_batch_num = edge_batch.unique(return_counts=True)
    sum_pe = scatter((1 - batch_aug_edge_weight), edge_batch, reduce="sum")

    reg = []
    for b_id in range(args.batch_size):
        if b_id in uni:
            num_edges = edge_batch_num[uni.tolist().index(b_id)]
            reg.append(sum_pe[b_id] / num_edges)
        else:
            pass
    reg = torch.stack(reg)
    reg = reg.mean()
    ratio = 0.4
    ratio = reg / ratio
   
    batch_aug_edge_weight = batch_aug_edge_weight / ratio # edge weight generalization  
    return batch_aug_edge_weight

import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def gen_ran_output(data, model, vice_model, args):
    for (adv_name,adv_param), (name,param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if name.split('.')[0] == 'proj_head':
            adv_param.data = param.data
        else:
            adv_param.data = param.data + args.eta * torch.normal(0,torch.ones_like(param.data)*param.data.std()).to(device)           
    z2 = vice_model(data.x, data.edge_index, data.batch, data.num_graphs)
    return z2

class AverageMeter(object):
    r"""
    Computes and stores the average and current value.
    Adapted from
    https://github.com/pytorch/examples/blob/ec10eee2d55379f0b9c87f4b36fcf8d0723f45fc/imagenet/main.py#L359-L380
    """
    def __init__(self, name=None, fmt='.6f'):
        fmtstr = f'{{val:{fmt}}} ({{avg:{fmt}}})'
        if name is not None:
            fmtstr = name + ' ' + fmtstr
        self.fmtstr = fmtstr
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        avg = self.sum / self.count
        if isinstance(avg, torch.Tensor):
            avg = avg.item()
        return avg

    def __str__(self):
        val = self.val
        if isinstance(val, torch.Tensor):
            val = val.item()
        return self.fmtstr.format(val=val, avg=self.avg)


class TwoAugUnsupervisedDataset(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return self.transform(image), self.transform(image)

    def __len__(self):
        return len(self.dataset)


def run_simgrace(args, model, dataloader, dataloader_valid, dataloader_eval):
    model = simclr(args.hidden_dim, args.num_gc_layers).to(device)
    vice_model = simclr(args.hidden_dim, args.num_gc_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_val = 0
    result_dic = {}

    os.makedirs('DGP_model', exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        loss_all = 0
        
        for data in dataloader:
            optimizer.zero_grad()
            data = data.to(device)

            x2 = gen_ran_output(data, model, vice_model, args)
            x1 = model(data.x, data.edge_index, data.batch, data.num_graphs)

            loss = model.loss_cal(x2, x1)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))

        model.eval()
        with torch.no_grad():
            node_test, g_emb_test, y_test = model.encoder.get_embeddings(dataloader_eval, device=device, is_batch=True)
            node_valid, g_emb_valid, y_valid = model.encoder.get_embeddings(dataloader_valid, device=device, is_batch=True)
            node_train, g_emb_train, y_train = model.encoder.get_embeddings(dataloader, device=device, is_batch=True)

            g_emb_test = np.concatenate(g_emb_test, axis=0)
            g_emb_valid = np.concatenate(g_emb_valid, axis=0)
            g_emb_train = np.concatenate(g_emb_train, axis=0)

            ood_labels = np.zeros(g_emb_test.shape[0])
            ood_slices = int(0.5 * g_emb_test.shape[0])
            ood_labels[ood_slices:] = 1
            
            ood_valid_labels = np.zeros(g_emb_valid.shape[0])
            ood_valid_slices = int(0.5 * g_emb_valid.shape[0])
            ood_valid_labels[ood_valid_slices:] = 1

            score = ssd(torch.from_numpy(g_emb_train).to(device), torch.from_numpy(g_emb_test).to(device))
            score_valid = ssd(torch.from_numpy(g_emb_train).to(device), torch.from_numpy(g_emb_valid).to(device))

            auroc, fpr95, aupr = metric(ood_labels, score)
            auroc_valid, fpr95_valid, aupr_valid = metric(ood_valid_labels, score_valid)
            
            print('Epoch {}: Valid AUC:{:.3f}, FPR95:{:.3f}, AUPR:{:.3f}'.format(
                epoch, float(auroc_valid), float(fpr95_valid), float(aupr_valid)))
            
            if auroc_valid > best_val:
                best_val = auroc_valid
                result_dic["testauc"] = float(auroc)
                result_dic["testfpr"] = float(fpr95)
                result_dic["testaupr"] = float(aupr)
                result_dic["valauc"] = float(auroc_valid)
                result_dic["valfpr"] = float(fpr95_valid)
                result_dic["valaupr"] = float(aupr_valid)
                result_dic['epoch'] = epoch

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val': best_val,
                }, f'DGP_model/{args.DS_pair}sim_pretrained_model.pth')
    
    return result_dic

def run_simgrace_ft(args, model, dataloader, dataloader_valid, dataloader_eval):
    sim_model_path = f'DGP_model/{args.DS_pair}sim_pretrained_model.pth'
    if os.path.exists(sim_model_path):
        checkpoint = torch.load(sim_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pre-trained SimGRACE model from {sim_model_path}")
    else:
        print("Warning: Pre-trained SimGRACE model not found. Training from scratch.")

    ft_model = copy.deepcopy(model)
    ft_model = ft_model.to(device)

    optimizer = torch.optim.Adam(ft_model.parameters(), lr=args.lr)

    predictor = nn.Sequential(
        nn.Linear(args.hidden_dim * args.num_gc_layers, 64),
        nn.ReLU(),
        nn.Linear(64, args.num_classes)
    ).to(device)
    
    predictor_optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)
    
    best_val = 0
    result_dic = {}
    
    for epoch in range(1, epochs + 1):
        ft_model.train()
        predictor.train()
        total_loss = 0
        
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            predictor_optimizer.zero_grad()

            h = ft_model(data.x, data.edge_index, data.batch, data.num_graphs)
            z = F.softmax(predictor(h), dim=1)
            loss_class_specific = F.cross_entropy(z, data.y)
            uniform_y = torch.ones_like(z) / z.size(1)
            loss_class_agnostic = F.cross_entropy(z, uniform_y)
            h_mean = torch.mean(h, dim=0)
            h_std = torch.std(h, dim=0)
            h_normalized = (h - h_mean) / (h_std + 1e-8)
            loss_distance = torch.mean(torch.sum(h_normalized ** 2, dim=1))

            loss = loss_class_specific + args.lambda_ * loss_class_agnostic + args.alpha_1 / loss_distance
            loss.backward()
            optimizer.step()
            predictor_optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch}: Loss: {total_loss:.4f}')

        ft_model.eval()
        predictor.eval()
        with torch.no_grad():
            node_test, g_emb_test, y_test = ft_model.encoder.get_embeddings(dataloader_eval, device=device, is_batch=True)
            node_valid, g_emb_valid, y_valid = ft_model.encoder.get_embeddings(dataloader_valid, device=device, is_batch=True)
            node_train, g_emb_train, y_train = ft_model.encoder.get_embeddings(dataloader, device=device, is_batch=True)

            g_emb_test = np.concatenate(g_emb_test, 0)
            g_emb_valid = np.concatenate(g_emb_valid, 0)
            g_emb_train = np.concatenate(g_emb_train, 0)

            ood_labels = np.zeros(g_emb_test.shape[0])
            ood_slices = int(0.5 * g_emb_test.shape[0])
            ood_labels[ood_slices:] = 1
            
            ood_valid_labels = np.zeros(g_emb_valid.shape[0])
            ood_valid_slices = int(0.5 * g_emb_valid.shape[0])
            ood_valid_labels[ood_valid_slices:] = 1

            score = ssd(torch.from_numpy(g_emb_train).to(device), torch.from_numpy(g_emb_test).to(device))
            score_valid = ssd(torch.from_numpy(g_emb_train).to(device), torch.from_numpy(g_emb_valid).to(device))

            auroc, fpr95, aupr = metric(ood_labels, score)
            auroc_valid, fpr95_valid, aupr_valid = metric(ood_valid_labels, score_valid)

            print(f'Epoch {epoch}: Valid AUC: {auroc_valid:.4f}, FPR95: {fpr95_valid:.4f}, AUPR: {aupr_valid:.4f}')
          
            if auroc_valid > best_val:
                best_val = auroc_valid
                result_dic["testauc"] = float(auroc)
                result_dic["testfpr"] = float(fpr95)
                result_dic["testaupr"] = float(aupr)
                result_dic["valauc"] = float(auroc_valid)
                result_dic["valfpr"] = float(fpr95_valid)
                result_dic["valaupr"] = float(aupr_valid)
                result_dic['epoch'] = epoch

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': ft_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'predictor_state_dict': predictor.state_dict(),
                    'predictor_optimizer_state_dict': predictor_optimizer.state_dict(),
                    'best_val': best_val,
                }, f'DGP_model/{args.DS_pair}sim_ft_best_model.pth')
    
    return result_dic

def run_dgp_sim(args, model, dataloader, dataloader_valid, dataloader_eval):
    sim_model_path = f'DGP_model/{args.DS_pair}sim_pretrained_model.pth'
    if os.path.exists(sim_model_path):
        checkpoint = torch.load(sim_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pre-trained SimGRACE model from {sim_model_path}")
    else:
        print("Warning: Pre-trained SimGRACE model not found. Training from scratch.")

    if hasattr(model.encoder, 'input_dim'):
        input_dim = model.encoder.input_dim
    else:
        input_dim = dataloader.dataset[0].x.shape[1] if hasattr(dataloader, 'dataset') else 1

    if hasattr(model.encoder, 'embedding_dim'):
        enc_dim = model.encoder.embedding_dim
    elif hasattr(model.encoder, 'output_dim'):
        enc_dim = model.encoder.output_dim
    else:
        enc_dim = args.hidden_dim * args.num_gc_layers

    dgp_model = DGP(
        gnn_encoder=model.encoder,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        backbone_type="DGP-Sim"
    ).to(device)

    dgp_model.prompt_generator_specific = PromptGenerator(
        emb_dim=enc_dim,   
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
    ).to(device)

    dgp_model.prompt_generator_agnostic = PromptGenerator(
        emb_dim=enc_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
    ).to(device)

    dgp_params = list(dgp_model.prompt_generator_specific.parameters()) + \
                 list(dgp_model.prompt_generator_agnostic.parameters()) + \
                 list(dgp_model.predictor.parameters())
    
    optimizer = torch.optim.AdamW(
        dgp_params, 
        lr=args.dgp_lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20, eta_min=1e-6
    )

    early_stopping = EarlyStopping(patience=30, verbose=True)

    best_val = 0
    result_dic = {}

    print("Starting iterative training process")
    for epoch in range(1, epochs + 1):
        total_loss = 0
        dgp_model.eval()
        train_embeddings_original = []
        train_embeddings_specific = []
        train_embeddings_agnostic = []
        with torch.no_grad():
            for data in dataloader:
                data = data.to(device)
                h_original, _ = dgp_model.gnn_encoder(data.x, data.edge_index, data.batch)
                h_specific, _, h_agnostic, _ = dgp_model(
                    data.x, data.edge_index, data.batch,
                    edge_weight=None
                )
                train_embeddings_original.append(h_original.cpu())
                train_embeddings_specific.append(h_specific.cpu())
                train_embeddings_agnostic.append(h_agnostic.cpu())
        train_embeddings_original = torch.cat(train_embeddings_original, dim=0).to(device)
        train_embeddings_specific = torch.cat(train_embeddings_specific, dim=0).to(device)
        train_embeddings_agnostic = torch.cat(train_embeddings_agnostic, dim=0).to(device)

        dgp_model.train()
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()

            _, node_emb = dgp_model.gnn_encoder(
                data.x, data.edge_index, data.batch, notfinal=True
            )

            edge_logits_specific = dgp_model.prompt_generator_specific(node_emb, data.edge_index)
            edge_logits_agnostic = dgp_model.prompt_generator_agnostic(node_emb, data.edge_index)

            batch_aug_edge_weight_specific = cal_score(args, edge_logits_specific, data)
            batch_aug_edge_weight_agnostic = cal_score(args, edge_logits_agnostic, data)

            h_specific, z_specific, h_agnostic, z_agnostic = dgp_model(
                data.x, data.edge_index, data.batch,
                edge_weight=batch_aug_edge_weight_specific
            )

            one_hot_target = data.y.view(-1)
            uniform_target = torch.ones_like(z_agnostic, dtype=torch.float).to(device) / args.num_classes
            
            loss_specific = F.nll_loss(z_specific, one_hot_target)
            loss_agnostic = F.kl_div(z_agnostic, uniform_target, reduction='batchmean')
            loss_disentangle = loss_specific + args.lambda_ * loss_agnostic

            loss_disentangle.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(dgp_params, max_norm=1.0)    
            optimizer.step()
            total_loss += loss_disentangle.item()

        dgp_model.train()
        dgp_model.predictor.eval()
        for data in dataloader:
            data = data.to(device)

            optimizer.zero_grad()

            _, node_emb = dgp_model.gnn_encoder(
                data.x, data.edge_index, data.batch, notfinal=True
            )

            edge_logits_specific = dgp_model.prompt_generator_specific(node_emb, data.edge_index)
            edge_logits_agnostic = dgp_model.prompt_generator_agnostic(node_emb, data.edge_index)

            batch_aug_edge_weight_specific = cal_score(args, edge_logits_specific, data)
            batch_aug_edge_weight_agnostic = cal_score(args, edge_logits_agnostic, data)

            h_original, _ = dgp_model.gnn_encoder(data.x, data.edge_index, data.batch)
            h_specific, z_specific, h_agnostic, z_agnostic = dgp_model(
                data.x, data.edge_index, data.batch,
                edge_weight=batch_aug_edge_weight_specific
            )

            dist_specific = ssd(h_original, h_specific)
            dist_agnostic = ssd(h_original, h_agnostic)
            loss_distance = args.alpha_1 / torch.mean(dist_specific) + args.alpha_2 / torch.mean(dist_agnostic)
            
            loss_distance.backward()
            optimizer.step()
            total_loss += loss_distance.item()

        print(f'Epoch {epoch}: Total Loss: {total_loss:.4f}')

        dgp_model.eval()
        with torch.no_grad():
            scores_test = []
            labels_test = []
            scores_valid = []
            labels_valid = []

            for data in dataloader_eval:
                data = data.to(device)
                _, node_emb = dgp_model.gnn_encoder(    
                    data.x, data.edge_index, data.batch, notfinal=True
                )

                edge_logits_specific = dgp_model.prompt_generator_specific(node_emb, data.edge_index)
                edge_logits_agnostic = dgp_model.prompt_generator_agnostic(node_emb, data.edge_index)
                
                batch_aug_edge_weight_specific = cal_score(args, edge_logits_specific, data, eva=True)
                batch_aug_edge_weight_agnostic = cal_score(args, edge_logits_agnostic, data, eva=True)
                
                h_specific = dgp_model.gnn_encoder(data.x, data.edge_index, data.batch, edge_weight=batch_aug_edge_weight_specific)
                h_agnostic = dgp_model.gnn_encoder(data.x, data.edge_index, data.batch, edge_weight=batch_aug_edge_weight_agnostic)
                
                if isinstance(h_specific, tuple):
                    h_specific = h_specific[0]
                if isinstance(h_agnostic, tuple):
                    h_agnostic = h_agnostic[0]

                dist_specific = ssd(train_embeddings_specific, h_specific)
                dist_agnostic = ssd(train_embeddings_agnostic, h_agnostic)
                decision_score = dist_specific + args.gamma * dist_agnostic
                
                scores_test.append(decision_score)
                labels_test.append(data.y)

            for data in dataloader_valid:
                data = data.to(device)
                _, node_emb = dgp_model.gnn_encoder(    
                    data.x, data.edge_index, data.batch, notfinal=True
                )

                edge_logits_specific = dgp_model.prompt_generator_specific(node_emb, data.edge_index)
                edge_logits_agnostic = dgp_model.prompt_generator_agnostic(node_emb, data.edge_index)
                
                batch_aug_edge_weight_specific = cal_score(args, edge_logits_specific, data, eva=True)
                batch_aug_edge_weight_agnostic = cal_score(args, edge_logits_agnostic, data, eva=True)
                
                h_specific = dgp_model.gnn_encoder(data.x, data.edge_index, data.batch, edge_weight=batch_aug_edge_weight_specific)
                h_agnostic = dgp_model.gnn_encoder(data.x, data.edge_index, data.batch, edge_weight=batch_aug_edge_weight_agnostic)
                
                if isinstance(h_specific, tuple):
                    h_specific = h_specific[0]
                if isinstance(h_agnostic, tuple):
                    h_agnostic = h_agnostic[0]

                dist_specific = ssd(train_embeddings_specific, h_specific)
                dist_agnostic = ssd(train_embeddings_agnostic, h_agnostic)
                decision_score = dist_specific + args.gamma * dist_agnostic
                
                scores_valid.append(decision_score)
                labels_valid.append(data.y)

            scores_test = torch.cat(scores_test)
            labels_test = torch.cat(labels_test).cpu()
            scores_valid = torch.cat(scores_valid)
            labels_valid = torch.cat(labels_valid).cpu()

            auroc, fpr95, aupr = metric(labels_test, scores_test)
            auroc_valid, fpr95_valid, aupr_valid = metric(labels_valid, scores_valid)

            print('EPOCH: {}, TEST: AUC:{:.4f}, FPR95:{:.4f}, AUPR:{:.4f}'.format(epoch, float(auroc), float(fpr95), float(aupr)))
            print('EPOCH: {}, VALI: AUC:{:.4f}, FPR95:{:.4f}, AUPR:{:.4f}'.format(epoch, float(auroc_valid), float(fpr95_valid), float(aupr_valid)))

            scheduler.step()

            early_stopping(auroc_valid, dgp_model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

            if auroc_valid >= best_val:
                best_val = auroc_valid
                result_dic["testauc"] = float(auroc)
                result_dic["testfpr"] = float(fpr95)
                result_dic["testaupr"] = float(aupr)
                result_dic["valauc"] = float(auroc_valid)
                result_dic["valfpr"] = float(fpr95_valid)
                result_dic["valaupr"] = float(aupr_valid)
                result_dic['epoch'] = epoch

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': dgp_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val': best_val,
                }, f'DGP_model/{args.DS_pair}dgp_sim_best_model.pth')

    return result_dic

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

if __name__ == '__main__':
    args = arg_parse()
    setup_seed(args.seed)
    print(args)

    accuracies = {'val':[], 'test':[]}
    epochs = args.epochs
    log_interval = 10
    batch_size = args.batch_size
    lr = args.lr
    dgp_lr = args.dgp_lr
    DS = args.DS

    data_train, data_val, data_test, data_train2, dataset_num_features = get_ood_dataset(args)
  
    if DS == "PTC_MR":
        dataset_num_features = 1

    if args.device >= 0:
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    model = simclr(args.hidden_dim, args.num_gc_layers).to(device)
    vice_model = simclr(args.hidden_dim, args.num_gc_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('eta: {}'.format(args.eta))
    print('================')

    if args.model_type == "simgrace":
        print("Running SimGRACE model...")
        result_dic = run_simgrace(args, model, data_train, data_val, data_test)
    elif args.model_type == "simgrace-ft":
        print("Running SimGRACE-ft model...")
        result_dic = run_simgrace_ft(args, model, data_train, data_val, data_test)
    elif args.model_type == "dgp-sim":
        print("Running DGP-Sim model...")
        result_dic = run_dgp_sim(args, model, data_train, data_val, data_test)
    else:
        print("Unknown model type. Using DGP-Sim as default.")
        result_dic = run_dgp_sim(args, model, data_train, data_val, data_test)
    
    print("Final Results:")
    print(f"Test AUC: {result_dic['testauc']:.4f}, FPR95: {result_dic['testfpr']:.4f}, AUPR: {result_dic['testaupr']:.4f}")
    print(f"Valid AUC: {result_dic['valauc']:.4f}, FPR95: {result_dic['valfpr']:.4f}, AUPR: {result_dic['valaupr']:.4f}")
    # print(f"Best Epoch: {result_dic['epoch']}")

  

    

   
    
 

  

    

   
    
 
  

    

   
    
 
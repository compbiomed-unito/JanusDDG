import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import pickle
import seaborn as sns

from torch_geometric.utils import to_networkx
#install required packages
import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)
# Helper function for visualization.
import networkx as nx
import matplotlib.pyplot as plt
from itertools import chain

from torch_geometric.data import Dataset
import torch_geometric.utils as pyg_utils
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool,GATv2Conv
from torch_geometric.nn.models import GCN, GAT
from torch.nn import Linear

from torch_geometric.utils import degree

import torch.nn as nn
from torch_geometric.utils import softmax
import math
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import random



def set_seed(seed):
    random.seed(seed)  # Python random
    np.random.seed(seed)  # Numpy random
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU (un singolo dispositivo)
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU (tutti i dispositivi, se usi multi-GPU)
    torch.backends.cudnn.deterministic = True  # Comportamento deterministico di cuDNN
    torch.backends.cudnn.benchmark = False  # Evita che cuDNN ottimizzi dinamicamente (influisce su riproducibilità)

# Imposta il seed
set_seed(42)




from random import sample

class DeltaDataset(Dataset):
    def __init__(self, data, dim_embedding, inv = False):
        self.data = data
        self.dim_embedding = dim_embedding
        self.inv = inv

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.inv: 
            return {
                'id': sample['id'],
                'wild_type': torch.tensor(sample['mut_type'], dtype=torch.float32),    #inverto mut con wild 
                'mut_type': torch.tensor(sample['wild_type'], dtype=torch.float32),    #inverto mut con wild             
                'length': torch.tensor(sample['length'], dtype=torch.float32),
                'ddg': torch.tensor(-float(sample['ddg']), dtype=torch.float32),       # -ddg
                #'alpha_vec': torch.tensor(-sample['alpha_vec'], dtype=torch.float32),  #-V
                'pos_mut': torch.tensor(sample['pos_mut'], dtype=torch.int64),
                }

        else:
            return {
                'id': sample['id'],
                'wild_type': torch.tensor(sample['wild_type'], dtype=torch.float32),
                'mut_type': torch.tensor(sample['mut_type'],dtype=torch.float32),
                'length': torch.tensor(sample['length'], dtype=torch.float32),
                'ddg': torch.tensor(float(sample['ddg']), dtype=torch.float32),
                #'alpha_vec': torch.tensor(sample['alpha_vec'], dtype=torch.float32),
                'pos_mut': torch.tensor(sample['pos_mut'], dtype=torch.int64),
                }




from torch_geometric.loader import DataLoader
import random

import torch
import torch.nn.functional as F

def collate_fn(batch):
    max_len = max(sample['wild_type'].shape[0] for sample in batch)  # Max sequence length in batch   700
    max_features = max(sample['wild_type'].shape[1] for sample in batch)  # Max feature size

    padded_batch = {
        'id': [],
        'wild_type': [],
        'mut_type': [],
        'length': [],
        'ddg': [],
        #'alpha_vec': [],
        'pos_mut': []
    }

    for sample in batch:
        wild_type_padded = F.pad(sample['wild_type'], (0, max_features - sample['wild_type'].shape[1], 
                                                       0, max_len - sample['wild_type'].shape[0]))
        mut_type_padded = F.pad(sample['mut_type'], (0, max_features - sample['mut_type'].shape[1], 
                                                     0, max_len - sample['mut_type'].shape[0]))

        padded_batch['id'].append(sample['id'])  
        padded_batch['wild_type'].append(wild_type_padded)  
        padded_batch['mut_type'].append(mut_type_padded)  
        padded_batch['length'].append(sample['length'])#append(torch.tensor(sample['length'], dtype=torch.float32))  
        padded_batch['ddg'].append(sample['ddg'])#append(torch.tensor(float(sample['ddg']), dtype=torch.float32))  
        #padded_batch['alpha_vec'].append(sample['alpha_vec'])#append(torch.tensor(sample['alpha_vec'], dtype=torch.float32))  
        padded_batch['pos_mut'].append(sample['pos_mut'])#append(torch.tensor(sample['pos_mut'], dtype=torch.int64))  

    # Convert list of tensors into a single batch tensor
    padded_batch['wild_type'] = torch.stack(padded_batch['wild_type'])  # Shape: (batch_size, max_len, max_features)
    padded_batch['mut_type'] = torch.stack(padded_batch['mut_type'])  
    padded_batch['length'] = torch.stack(padded_batch['length'])  
    padded_batch['ddg'] = torch.stack(padded_batch['ddg'])  
    #padded_batch['alpha_vec'] = torch.stack(padded_batch['alpha_vec'])  
    padded_batch['pos_mut'] = torch.stack(padded_batch['pos_mut'])  

    return padded_batch



def dataloader_generation(E_TYPE, train_path, validation_path, test_path, batch_size = 128, dataloader_shuffle = True, inv= False):
    
    EMBEDDING_TYPE = E_TYPE
    
    if EMBEDDING_TYPE == 'ESM2':

        '''train formato da s2648 + UnionV e DA; 1000 dei DA sono usati nel validation insieme a s669 DA
        '''
        
        dim_embedding = 1280
        
        dataset_train = []
        dataset_validation = []
        dataset_test = []

        
        for path in train_path:
            with open(path, 'rb') as f:
                dataset_train += pickle.load(f)
        
        for path in validation_path:
            with open(path, 'rb') as f:
                dataset_validation += pickle.load(f)
        
        for path in test_path:           
            with open(path, 'rb') as f:
                dataset_test += pickle.load(f)
    
    else:
        assert False
    
    dataset_train = DeltaDataset(dataset_train, dim_embedding, inv = inv)  
    dataset_test = DeltaDataset(dataset_test, dim_embedding, inv = inv)
    dataset_validation = DeltaDataset(dataset_validation, dim_embedding, inv = inv)
    print('ok fin qui')
    
    # Creazione DataLoader
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=dataloader_shuffle, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=dataloader_shuffle, collate_fn=collate_fn)
    dataloader_validation = DataLoader(dataset_validation, batch_size=batch_size, shuffle=dataloader_shuffle, collate_fn=collate_fn)

    return dataloader_train, dataloader_validation, dataloader_test




from torch.utils.data import DataLoader  # Use standard PyTorch DataLoader
import random







import copy

def output_model_from_batch(batch, model, device, train=True):

    '''Dato un modello pytorch e batch restituisce: output_modello, True labels'''
    
    x_wild = batch['wild_type'].float().to(device)
    x_mut = batch['mut_type'].float().to(device)
    labels = batch['ddg'].float().to(device)
    length = batch['length'].to(device)
    output_ddg = model(x_wild, x_mut, length, train = train)
    
    return output_ddg, labels


# def output_model_inv_from_batch(batch, model, device, train=False):

#     '''Dato un modello pytorch e batch restituisce: output_modello, True labels'''
    
#     x_wild = batch['mut_type'].float().to(device)
#     x_mut = batch['wild_type'].float().to(device)
#     labels = -batch['ddg'].float().to(device)
#     length = batch['length'].to(device)
#     output_ddg = model(x_wild, x_mut, length, train = train)
    
#     return output_ddg, labels


def training_and_validation_loop_ddg(model, dataloader_train, dataloader_test, dataloader_validation, path_save_fig, epochs=20, lr =0.001, patience=10):
            
    criterion =nn.MSELoss()# nn.HuberLoss()#nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    pearson_r_train = []
    pearson_r_test = []
    #pearson_r_test_inv =[]
    pearson_r_validation = []
    
    loss_ddg_train = []
    loss_ddg_test = []
    loss_ddg_validation = []

    num_epochs = epochs
    for epoch in range(num_epochs):
            
        # Training Loop
        model.train()
        preds_ddg_train = []
        preds_dgw_train = []
        preds_dgm_train = []
        preds_coerenza_train = []

        labels_tot_epoch = []

        for i, batch in enumerate(dataloader_train):
            train = True
            
            optimizer.zero_grad()
            output_ddg, labels = output_model_from_batch(batch, model, device, train=train)
            
            if isinstance(output_ddg, list):
                # Compute the loss for each output and sum them
                loss_list = [criterion(output_aa, labels) for output_aa in output_ddg]
                loss_ddg = torch.stack(loss_list).sum()
                output_ddg  = torch.mean(torch.stack(output_ddg), dim=0)
            else: 
                loss_ddg = criterion(output_ddg, labels)  #usa se NON uso hydra
            
            tot_loss = loss_ddg
            
            # Backpropagation and optimization
            tot_loss.backward()
            optimizer.step()

            # Collect predictions
            preds_ddg_train.extend(output_ddg.cpu().reshape(-1).tolist())
            labels_tot_epoch.extend(labels.cpu().tolist())

        # Calculate and print train metrics
        train_loss = mean_squared_error(preds_ddg_train, labels_tot_epoch)
        train_correlation = pearsonr(preds_ddg_train, labels_tot_epoch)[0]
        train_spearman = spearmanr(preds_ddg_train, labels_tot_epoch)[0]
        
        loss_ddg_train.append(train_loss)
        pearson_r_train.append(train_correlation)
        
        # Validation Loop
        model.eval()  # Set model to evaluation mode
                
        all_preds_validation = []
        all_labels_validation = []
        all_preds_test = []
        all_labels_test = []
        # all_preds_test_inv = []
        # all_labels_test_inv = []        
        
        with torch.no_grad():  # Disable gradient calculation
            train = False
            for i, batch in enumerate(dataloader_test):

                output_ddg,labels = output_model_from_batch(batch, model, device, train=train) 
                    
                all_preds_test.extend(output_ddg.cpu().reshape(-1).tolist())
                all_labels_test.extend(labels.cpu().tolist())

                # output_ddg_inv,labels_inv = output_model_inv_from_batch(batch, model, device, train=train) 

                # all_preds_test_inv.extend(output_ddg_inv.cpu().reshape(-1).tolist())
                # all_labels_test_inv.extend(labels_inv.cpu().tolist())
            
            
            # Calculate validation metrics
            test_loss = mean_squared_error(all_preds_test, all_labels_test)
            loss_ddg_test.append(test_loss)
            
            test_correlation, _ = pearsonr(all_preds_test, all_labels_test)
            pearson_r_test.append(test_correlation)
            
            # test_correlation_inv, _ = pearsonr(all_preds_test_inv, all_labels_test_inv)
            # pearson_r_test_inv.append(test_correlation_inv)

            for i, batch in enumerate(dataloader_validation):
                output_ddg,labels = output_model_from_batch(batch, model, device, train=train)

                all_preds_validation.extend(output_ddg.cpu().reshape(-1).tolist())
                all_labels_validation.extend(labels.cpu().tolist()) #MESSO UN -  se DEF AL CONTRARIO
            
            # Calculate validation metrics
            val_loss = mean_squared_error(all_preds_validation, all_labels_validation)
            loss_ddg_validation.append(val_loss)
            
            val_correlation, _ = pearsonr(all_preds_validation, all_labels_validation)
            pearson_r_validation.append(val_correlation)

        
        if val_correlation >= max(pearson_r_validation): 
            best_model = copy.deepcopy(model)
            print(f'\033[91mEpoch {epoch+1}/{num_epochs}')
            print(f'Train -      Loss: {train_loss:.4f}, Pearson r: {train_correlation:.4f}, Rho spearman: {train_spearman:.4f}')
            print(f'Validation - Loss: {val_loss:.4f}, Pearson r: {val_correlation:.4f}, Rho spearman: {spearmanr(all_preds_validation, all_labels_validation)[0]:.4f}',)        
            print(f'Test -       Loss: {test_loss:.4f}, Pearson r: {test_correlation:.4f}, Rho spearman: {spearmanr(all_preds_test, all_labels_test)[0]:.4f}\033[0m\n')
            #print(f'Test inv -   Pearson r: {test_correlation_inv:.4f}, Rho spearman: {spearmanr(all_preds_test_inv, all_labels_test_inv)[0]:.4f}\033[0m\n')
      

        else:
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Train -      Loss: {train_loss:.4f}, Pearson r: {train_correlation:.4f}, Rho spearman: {train_spearman:.4f}')
            print(f'Validation - Loss: {val_loss:.4f}, Pearson r: {val_correlation:.4f}, Rho spearman: {spearmanr(all_preds_validation, all_labels_validation)[0]:.4f}',)        
            print(f'Test -       Loss: {test_loss:.4f}, Pearson r: {test_correlation:.4f}, Rho spearman: {spearmanr(all_preds_test, all_labels_test)[0]:.4f}\n')
            #print(f'Test inv -   Pearson r: {test_correlation_inv:.4f}, Rho spearman: {spearmanr(all_preds_test_inv, all_labels_test_inv)[0]:.4f}\033[0m\n')
                  
        if epoch > (np.argmax(pearson_r_validation) + patience):
            print(f'\033[91mEarly stopping at epoch {epoch+1}\033[0m')
            break
    
    pearson_max_val = np.max(pearson_r_validation)

    return pearson_r_train, pearson_r_validation, pearson_r_test, loss_ddg_train, loss_ddg_validation, loss_ddg_test, pearson_max_val, best_model






class Cross_Attention_DDG(nn.Module):
    
    def __init__(self, base_module, cross_att=False, dual_cross_att= False,**transf_parameters):
        super().__init__()
        self.base_ddg = base_module(**transf_parameters, cross_att=cross_att, dual_cross_att= dual_cross_att).to(device)
    
    def forward(self, x_wild, x_mut, length, train = True):

        delta_x = x_wild - x_mut
        output_TCA = self.base_ddg(delta_x, x_wild, length)
        
        return output_TCA   
import torch
import torch.nn as nn


def apply_masked_pooling(position_attn_output, padding_mask):

    # Convert mask to float for element-wise multiplication
    padding_mask = padding_mask.float()

    # Global Average Pooling (GAP) - Exclude padded tokens
    # Sum only over valid positions (padding_mask is False for valid positions)
    sum_output = torch.sum(position_attn_output * (1 - padding_mask.unsqueeze(-1)), dim=1)  # (batch_size, feature_dim)
    valid_count = torch.sum((1 - padding_mask).float(), dim=1)  # (batch_size,)
    gap = sum_output / valid_count.unsqueeze(-1)  # Divide by number of valid positions

    # Global Max Pooling (GMP) - Exclude padded tokens
    # Set padded positions to -inf so they don't affect the max computation
    position_attn_output_masked = position_attn_output * (1 - padding_mask.unsqueeze(-1)) + (padding_mask.unsqueeze(-1) * (- 1e10))
    gmp, _ = torch.max(position_attn_output_masked, dim=1)  # (batch_size, feature_dim)

    return gap, gmp


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=3700):
        super(SinusoidalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape (1, max_len, embedding_dim)
        self.register_buffer('pe', pe)  # Salvato come tensore fisso (non parametro)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerRegression(nn.Module):
    def __init__(self, input_dim=1280, num_heads=8, dropout_rate=0., num_experts=1, f_activation = nn.ReLU(), kernel_size=15, cross_att = False,
                dual_cross_att=False):
        
        super(TransformerRegression, self).__init__()
        self.cross_att = cross_att
        self.dual_cross_att = dual_cross_att
        
        print(f'Cross Attention: {cross_att}')
        print(f'Dual Cross Attention: {dual_cross_att}')

        self.embedding_dim = input_dim
        self.act = f_activation
        self.max_len = 3700 #lunghezza massima proteina
        out_channels = 128  #num filtri conv 1D
        kernel_size = 20
        padding = 0
        
        self.conv1d = nn.Conv1d(in_channels=self.embedding_dim, 
                                             out_channels=out_channels, 
                                             kernel_size=kernel_size, 
                                             padding=padding) 
        
        self.conv1d_wild = nn.Conv1d(in_channels=self.embedding_dim, 
                                             out_channels=out_channels, 
                                             kernel_size=kernel_size, 
                                             padding=padding)

        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        # Cross-attention layers
        self.positional_encoding = SinusoidalPositionalEncoding(out_channels, 3700)
        self.speach_att_type = True
        self.multihead_attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, dropout=dropout_rate, batch_first=True )
        self.inverse_attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, dropout=dropout_rate, batch_first =True)
        
        if cross_att:
            # Router (learns which expert to choose per token)
            if dual_cross_att:
                dim_position_wise_FFN = out_channels*2
            else:
                dim_position_wise_FFN = out_channels


        else:
            dim_position_wise_FFN = out_channels
        
        self.norm3 = nn.LayerNorm(dim_position_wise_FFN)
        self.norm4 = nn.LayerNorm(dim_position_wise_FFN)        
        self.router = nn.Linear(dim_position_wise_FFN, num_experts) #dim_position_wise_FFN*2
        # Mixture of Experts (Switch FFN)
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(dim_position_wise_FFN, 512),
            self.act,
            nn.Linear(512, dim_position_wise_FFN)
        ) for _ in range(num_experts)])

        self.Linear_ddg = nn.Linear(dim_position_wise_FFN*2, 1)

            

    def create_padding_mask(self, length, seq_len, batch_size):
        """
        Create a padding mask for multihead attention.
        length: Tensor of shape (batch_size,) containing the actual lengths of the sequences.
        seq_len: The maximum sequence length.
        batch_size: The number of sequences in the batch.
        
        Returns a padding mask of shape (batch_size, seq_len).
        """
        mask = torch.arange(seq_len, device=length.device).unsqueeze(0) >= length.unsqueeze(1)
        return mask



    def forward(self, delta_w_m, x_wild, length):
            # Add positional encoding
            
            delta_w_m = delta_w_m.transpose(1, 2)  # (batch_size, feature_dim, seq_len) -> (seq_len, batch_size, feature_dim)
            C_delta_w_m = self.conv1d(delta_w_m)
            #C_delta_w_m = self.act(C_delta_w_m)  #CASTRENSE USA RELU IO NON AVEVO MESSO NULLA 
            C_delta_w_m = C_delta_w_m.transpose(1, 2)  # (seq_len, batch_size, feature_dim) -> (batch_size, seq_len, feature_dim)
            C_delta_w_m = self.positional_encoding(C_delta_w_m)
            
            x_wild = x_wild.transpose(1, 2)  # (batch_size, feature_dim, seq_len) -> (seq_len, batch_size, feature_dim)
            C_x_wild = self.conv1d_wild(x_wild)
            #C_x_wild = self.act(C_x_wild)  #CASTRENSE USA RELU IO NON AVEVO MESSO NULLA 
            C_x_wild = C_x_wild.transpose(1, 2)  # (seq_len, batch_size, feature_dim) -> (batch_size, seq_len, feature_dim)
            C_x_wild = self.positional_encoding(C_x_wild)            
            
            batch_size, seq_len, feature_dim = C_x_wild.size()

            padding_mask = self.create_padding_mask(length, seq_len, batch_size)        

            if self.cross_att :
                if self.dual_cross_att:
                    
                    if self.speach_att_type:
                        print('ATTENTION TYPE: Dual cross Attention\n q = wild , k = delta, v = delta and q = delta , k = wild, v = wild \n ----------------------------------')
                        self.speach_att_type = False
                        
                    direct_attn_output, _ = self.multihead_attention(C_x_wild, C_delta_w_m, C_delta_w_m, key_padding_mask=padding_mask)
                    direct_attn_output += C_delta_w_m 
                    direct_attn_output = self.norm1(direct_attn_output)                        
                    
                    inverse_attn_output, _ = self.inverse_attention(C_delta_w_m, C_x_wild, C_x_wild, key_padding_mask=padding_mask)                   
                    inverse_attn_output += C_x_wild  
                    inverse_attn_output = self.norm2(inverse_attn_output)
                    
                    attn_output = torch.cat([direct_attn_output, inverse_attn_output], dim=-1)
                    #combined_output = self.norm3(combined_output)

                else:
                    if self.speach_att_type:
                        print('ATTENTION TYPE: Cross Attention \n q = wild , k = delta, v = delta  \n ----------------------------------')
                        self.speach_att_type = False

                    attn_output, _ = self.multihead_attention(C_x_wild, C_delta_w_m, C_delta_w_m, key_padding_mask=padding_mask)
                    attn_output += C_delta_w_m 
                    attn_output = self.norm1(attn_output) 
            
            else:
                if self.speach_att_type:
                    print('ATTENTION TYPE: Self Attention \n q = delta , k = delta, v = delta  \n ----------------------------------')
                    self.speach_att_type = False
                
                attn_output, _ = self.multihead_attention(C_delta_w_m, C_delta_w_m, C_delta_w_m, key_padding_mask=padding_mask)
                attn_output += C_delta_w_m
                attn_output = self.norm1(attn_output)


            ########
            # Route tokens to experts
            routing_logits = self.router(attn_output)  # Shape: [batch, seq_len, num_experts]
            routing_weights = F.softmax(routing_logits, dim=-1)  # Probability distribution over experts
            expert_indices = torch.argmax(routing_weights, dim=-1)  # Choose the most probable expert for each token
            
            # Apply selected expert
            batch_size, seq_len, embed_dim = attn_output.shape
            output = torch.zeros_like(attn_output)
            for i in range(self.num_experts):
                mask = (expert_indices == i).unsqueeze(-1).float()  # Mask for tokens assigned to expert i
                expert_out = self.experts[i](attn_output) * mask  # Apply expert only to selected tokens
                output += expert_out  # Aggregate expert outputs
            ############

            position_attn_output = attn_output + output

            position_attn_output = self.norm3(position_attn_output)
    
            gap, gmp = apply_masked_pooling(position_attn_output, padding_mask)
    
            # Concatenate GAP and GMP
            pooled_output = torch.cat([gap, gmp], dim=-1)  # (batch_size, 2 * feature_dim)
    
            # Pass through FFNN to predict DDG
            x = self.Linear_ddg(pooled_output)        
            
            return x.squeeze(-1)





import gc




E_TYPE='ESM2'
folds = [0, 1, 2, 3, 4]

for Model_num in [0,1,2,3,4]:    
    val_set = [folds[Model_num]]  # L'elemento corrente è il test set
    train_set = list(chain(folds[:Model_num], folds[Model_num+1:]))  # Tutti gli altri sono il training set
    
    train_path = [f's2450_fold_{i}.pkl' for i in train_set]+[f's2450_fold_{i}_inv.pkl' for i in train_set]
    val_path = [f's2450_fold_{i}.pkl' for i in val_set]+[f's2450_fold_{i}_inv.pkl' for i in val_set]
    test_path = ['s669_Castrense.pkl']
    
    dataloader_train, dataloader_validation, dataloader_test = dataloader_generation(E_TYPE, train_path = train_path, validation_path = val_path,
                                                                                     test_path = test_path, batch_size = 6,
                                                                                     dataloader_shuffle = True, inv= False)
    

    #PROVA base base
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    lr = 1e-4
    input_dim = 1280
    
    transf_parameters={'input_dim':1280, 'num_heads':8,
                        'dropout_rate':0.,}
    
    
    patience = 100
    
    DDG_model = TransformerRegression
    
    Final_model = Cross_Attention_DDG(DDG_model, cross_att = True, dual_cross_att=True, **transf_parameters)
    
    path_save_fig = 'DDGemb \n ----------------------------------'
    print(path_save_fig)
    p_tr,p_val, p_te, l_tr,l_val, l_te, pearson_max_val, best_model = training_and_validation_loop_ddg(Final_model, dataloader_train, dataloader_test,
                                                                                       dataloader_validation,
                                                                                       path_save_fig, epochs=500, lr =lr,patience = patience)
    
    
    
    torch.save(best_model, f'DDGemb_Cross_{Model_num}.pth')


        # Eliminare la variabile
    del dataloader_train
    del dataloader_test
    del dataloader_validation

    # Forzare il Garbage Collector
    gc.collect()
        # pearson_memory.append(pearson_max_val)
    
    print(f'{p_te=},\n{p_val=}\n', flush=True)



















import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
import json

class RankingDataset(Dataset):
    
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        # self.keys = df['query_id']
        
    def __len__(self):
        return len(self.df)
    
    def transform(self, idx):
        mean = np.mean(self.df.iloc[idx]['fl_features'], axis = 0)
        std = np.std(self.df.iloc[idx]['fl_features'], axis = 0)
        return mean, std
    
    def __getitem__(self, idx):
        # mean, std = self.transform(idx)
        feats = json.loads(self.df['fl_features'][idx])
        trgts =  json.loads(self.df['target'][idx])
        feats = torch.tensor(feats)
        trgts = torch.tensor(trgts)
        
        return idx, ([feats], trgts)
    
def calulate_ranking_metrics(fin_targets, fin_outputs, **kwargs):
    fin_targets = torch.tensor(fin_targets)
    fin_outputs = torch.tensor(fin_outputs)
    num_of_rates = fin_outputs.shape[-1]
    output = fin_outputs.softmax(dim=-1)
    outputs = torch.sum(output * torch.arange(num_of_rates, device = output.device), dim = -1).unsqueeze(-1)
    mask = torch.tensor(kwargs['mask'])
    # print(len(mask), len(fin_outputs), fin_outputs[0].shape, fin_targets[0].shape, flush=True)
    ndcg_scores = {key: [] for key in [5,10,None]}
    for i in range(len(fin_outputs)):
        query_mask = mask[i]  
        query_outputs = outputs[i][query_mask].squeeze()
        query_targets = fin_targets[i][query_mask]
        if query_targets.numel() > 1:
            # print(query_outputs.shape, query_targets.shape, flush=True )
            for i,k in enumerate(ndcg_scores.keys()):
                c = k
                if query_targets.sum() != 0:
                    ndcg = ndcg_score(
                        [query_targets.cpu().numpy()],
                        [query_outputs.cpu().numpy()],
                    k = c)
                else :
                    ndcg = 1
                ndcg_scores[k].append(ndcg)
    avg_ndcg5_epoch = sum(ndcg_scores[5]) / len(ndcg_scores[5]) if ndcg_scores[5] else 0.0
    avg_ndcg10_epoch = sum(ndcg_scores[10]) / len(ndcg_scores[10]) if ndcg_scores[10] else 0.0
    avg_ndcg_epoch = sum(ndcg_scores[None]) / len(ndcg_scores[None]) if ndcg_scores[None] else 0.0
        
    df = pd.DataFrame(columns=['web10k'],
                      index=['ndcg5',
                             'ndcg10',
                             'ndcg'])
    
    df.loc['ndcg10', 'web10k'] = avg_ndcg10_epoch
    df.loc['ndcg5', 'web10k'] = avg_ndcg5_epoch
    df.loc['ndcg', 'web10k'] = avg_ndcg_epoch
    
    if kwargs['verbose']:
        print(df)
    
    return df
    
    
'''
========================================================================================================================
'''
class ResidualBlock(nn.Module):
    def __init__(self, size, dropout_rate=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, sublayer_output):
        return self.layer_norm(x + self.dropout(sublayer_output))
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_hidden, dropout_rate):
        super().__init__()
        
        self.multihead_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads)
        self.attention_residual = ResidualBlock(d_model, dropout_rate)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, d_model)
        )
        self.ffn_residual = ResidualBlock(d_model, dropout_rate)
        
    def forward(self, x, attention_mask=None, padding_mask=None):
        # Multi-head attention
        attn_output, _ = self.multihead_attention(
            query=x, key=x, value=x, attn_mask=attention_mask, key_padding_mask=padding_mask
        )
        x = self.attention_residual(x, attn_output)

        # Feed-forward
        ffn_output = self.feed_forward(x)
        x = self.ffn_residual(x, ffn_output)
        return x
         
class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, ffn_hidden, input_dim, dropout_rate, output_dim = 5):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model, bias=True)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_hidden, dropout_rate)
            for _ in range(n_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(d_model, output_dim)

    def create_padding_matrix(self, input_tensor):
        # print(input_tensor.size(), flush=True)
        batch_size, seq_length, _ = input_tensor.size()
        row_sums = input_tensor.sum(dim=-1).to(input_tensor.device)
        padding_mask = (row_sums != 0).float().to(input_tensor.device)
        return padding_mask

    def forward(self, x):
        # Create masks
        padding_mask = self.create_padding_matrix(x)
        
        # Input projection
        x = self.input_projection(x)
        
        # Permute for compatibility with nn.MultiheadAttention
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        
        # Pass through Transformer layers
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)
        
        # Compute scores for each item
        output = self.output_layer(x)  # (seq_len, batch_size, 1)
                                                                                                    
        return output.permute(1, 0, 2)  # (batch_size, seq_len, 1)

def make_Encoder_model(output_dim) -> Encoder:

    return Encoder(d_model=512, 
                n_heads=2, 
                n_layers=2, 
                ffn_hidden=512, 
                input_dim=136, 
                dropout_rate = 0.15,
                output_dim=output_dim
                ).to('cpu')


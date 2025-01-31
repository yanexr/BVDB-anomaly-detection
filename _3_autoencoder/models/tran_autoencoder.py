import torch
import torch.nn as nn
import math
from .base_autoencoder import BaseAutoencoder

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=True, device='cuda'):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.device = device

        pe = torch.zeros(max_len, d_model, device=device)   # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * 
                           (-math.log(10000.0) / d_model)) # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if self.batch_first:
            pe = pe.unsqueeze(0)    # [1, max_len, d_model]
        else:
            pe = pe.unsqueeze(1)    # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        if self.batch_first:
            x = x + self.pe[:, pos:pos + x.size(1), :]
        else:
            x = x + self.pe[pos:pos + x.size(0), :]
        return self.dropout(x)

class TransformerAutoencoder(BaseAutoencoder):
    def build(self):
        self.n_feats = self.config['n_feats']
        self.seq_len = self.config['seq_len']
        self.scale = self.config.get('scale', 16)
        self.model_dim = self.scale * self.n_feats

        self.linear_layer = nn.Linear(self.n_feats, self.model_dim).to(self.device)
        self.output_layer = nn.Linear(self.model_dim, self.n_feats).to(self.device)
        self.pos_encoder = PositionalEncoding(self.model_dim, dropout=0.1, max_len=self.seq_len, batch_first=True, device=self.device).to(self.device)
        nhead = self.config.get('nhead', 8)
        num_layers = self.config.get('num_layers', 1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=nhead, batch_first=True, dim_feedforward=256, dropout=0.1).to(self.device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(self.device)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.model_dim, nhead=nhead, batch_first=True, dim_feedforward=256, dropout=0.1).to(self.device)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers).to(self.device)

    def forward(self, x):
        # input shape: (batch_size, seq_len, channels)
        x = x.to(self.device)
        
        src = self.linear_layer(x)
        src = src * math.sqrt(self.model_dim)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)  # Shape: (batch_size, seq_len, model_dim)
        z = torch.mean(memory, dim=1, keepdim=True)  # Shape: (batch_size, 1, model_dim)

        tgt = self.linear_layer(x)
        tgt = tgt * math.sqrt(self.model_dim)
        x = self.transformer_decoder(tgt, z)
        x = self.output_layer(x)
        return x


class TransformerAutoencoder2(BaseAutoencoder):
    def build(self):
        self.n_feats = self.config['n_feats']
        self.seq_len = self.config['seq_len']

        self.pos_encoder = PositionalEncoding(self.n_feats, dropout=0.1, max_len=self.seq_len, batch_first=True, device=self.device).to(self.device)

        nhead = self.config.get('nhead', 8)
        num_layers = self.config.get('num_layers', 1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_feats, nhead=nhead, batch_first=True, dim_feedforward=256, dropout=0.1).to(self.device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(self.device)

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.n_feats, nhead=nhead, batch_first=True, dim_feedforward=256, dropout=0.1).to(self.device)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers).to(self.device)
        
    def forward(self, x):
        # input shape: (batch_size, seq_len, channels)
        tgt = x.to(self.device)

        # mask values below 98th percentile
        percentile_threshold = torch.quantile(tgt[:, :, 0], 0.98, dim=1, keepdim=True)  # Shape: (batch_size, 1)
        threshold_expanded = percentile_threshold.unsqueeze(-1).expand(-1, -1, tgt.shape[1])  # Shape: (batch_size, seq_len)
        mask = (tgt[:, :, 0] >= threshold_expanded.squeeze()).float()  # Shape: (batch_size, seq_len)    

        src = self.pos_encoder(tgt) # (batch_size, seq_len, channels)
        memory = self.transformer_encoder(src, src_key_padding_mask=~mask.bool())
        x = self.transformer_decoder(tgt, memory, tgt_key_padding_mask=~mask.bool()) # (batch_size, seq_len, channels)

        x = x[:, :, 0:1] # (batch_size, seq_len, 1)
        return x  
'''
from  https://github.com/gzerveas/mvts_transformer/tree/master
'''

import torch
import torch.nn as nn
import math
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer
from torch.nn import functional as F

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
    
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return self.dropout(x)

class Transformer1D(nn.Module):
    def __init__(self, input_size, nhead, num_layers, output_size=1):
        super(Transformer1D, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model=input_size)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.fc = nn.Linear(input_size, output_size)


    def forward(self, x):
        print(x.shape)
        x = self.pos_encoder(x)
        print(x.shape)
        out = self.transformer(x)
        print(out.shape)
        #print(out.shape)
        #out = out[:, -1, :]  
        out = self.fc(out)   # Pass it through the output layer
        print(out.shape)
        return out.squeeze(-1)
    
class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TSTransformerEncoderClassiregressor(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, activation='gelu'):
        super(TSTransformerEncoderClassiregressor, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = LearnablePositionalEncoding(d_model, dropout=dropout, max_len=max_len)


        encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout, activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.act = F.gelu

        self.dropout1 = nn.Dropout(dropout)

        self.output_layer =  nn.Linear(d_model, num_classes)
        for name, param in encoder_layer.named_parameters():
            if 'weight' in name and param.data.dim() == 2:
                nn.init.kaiming_uniform_(param)

    def forward(self, X):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """
        #print(X.shape)
        X = X.squeeze(1)
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(2, 0, 1)
        #print(X.shape, inp.shape)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        #print('a', inp, inp.shape)
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp)  # (seq_length, batch_size, d_model)
        #print(output.shape)
        #print('b', output, output.shape)
        #output = self.act(output[-1, :, :])  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(torch.sum(output, 0))
        #output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        #print('b1', output, output.shape)
        #output = self.dropout1(output)
        #print('b2', output, output.shape)
        # Output
        #output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        #output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        #print('b3', output, output.shape)
        #print(output)
        output = self.output_layer(output)  # (batch_size, num_classes)
        #print(output)
        #print('c', output, output.shape)
        return output
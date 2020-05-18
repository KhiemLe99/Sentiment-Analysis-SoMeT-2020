import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout_prob=0.1):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(1024, embedding_size)
        position = torch.arange(0, 1024, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, embedding_matrix, num_layers=6, num_heads=8, maxlen=100, dropout_prob=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.maxlen = maxlen
        self.dropout_prob = dropout_prob
        self.embedding_matrix = embedding_matrix

        self.embedding = nn.Embedding(self.embedding_matrix.shape[0], self.embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(self.embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.position_encoder = PositionalEncoding(self.embedding_matrix.shape[1], self.dropout_prob)
        self.transformer_encoder_layers = nn.TransformerEncoderLayer(self.embedding_matrix.shape[1], self.num_heads, 256, self.dropout_prob)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layers, self.num_layers)

        self.drop = nn.Dropout(self.dropout_prob)
        self.fc   = nn.Linear(self.maxlen * self.embedding_matrix.shape[1], 2)

    def forward(self, x):
        embedding = self.embedding(x)
        embedding = torch.squeeze(torch.unsqueeze(embedding, 0))

        embedding = embedding * math.sqrt(self.embedding_matrix.shape[1])
        position_encoded = self.position_encoder(embedding)
        transformer_encoded = self.transformer_encoder(position_encoded, None)
        transformer_encoded = self.drop(transformer_encoded.view((transformer_encoded.shape[0], -1)))

        out = self.fc(transformer_encoded)

        return out
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=128, dropout_prob=0.1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.embedding_matrix = embedding_matrix
        
        self.embedding = nn.Embedding(self.embedding_matrix.shape[0], self.embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(self.embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        self.lstm = nn.LSTM(self.embedding_matrix.shape[1], self.hidden_size, bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(self.dropout_prob)
        self.fc   = nn.Linear(6*self.hidden_size, 2)
        
    def forward(self, x):
        embedding = self.embedding(x)
        embedding = torch.squeeze(torch.unsqueeze(embedding, 0))

        lstm, _ = self.lstm(embedding)
        avg_pool = torch.mean(lstm, 1)
        max_pool, _ = torch.max(lstm, 1)
        
        cat = torch.cat((avg_pool, max_pool, lstm[:,-1,:]), 1)
        cat = self.drop(cat)
        cat = self.fc(cat)

        return cat

class GRU(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=128, dropout_prob=0.1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.embedding_matrix = embedding_matrix
        
        self.embedding = nn.Embedding(self.embedding_matrix.shape[0], self.embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(self.embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        self.gru  = nn.GRU(self.embedding_matrix.shape[1], self.hidden_size, bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(self.dropout_prob)
        self.fc   = nn.Linear(6*self.hidden_size, 2)
        
    def forward(self, x):
        embedding = self.embedding(x)
        embedding = torch.squeeze(torch.unsqueeze(embedding, 0))

        gru, _ = self.gru(embedding)
        avg_pool = torch.mean(gru, 1)
        max_pool, _ = torch.max(gru, 1)

        cat = torch.cat((avg_pool, max_pool, gru[:,-1,:]), 1)
        cat = self.drop(cat)
        cat = self.fc(cat)

        return cat
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMCNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=128, dropout_prob=0.1):
        super(LSTMCNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.embedding_matrix = embedding_matrix
        
        self.embedding = nn.Embedding(self.embedding_matrix.shape[0], self.embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(self.embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(self.embedding_matrix.shape[1], self.hidden_size, bidirectional=True)
        self.fc1  = nn.Linear(self.embedding_matrix.shape[1] + 2*self.hidden_size, self.hidden_size)
        self.fc2  = nn.Linear(self.hidden_size, 2)
        self.drop = nn.Dropout(self.dropout_prob)

    def forward(self, x):
        embedding = self.embedding(x).permute(1, 0, 2)

        lstm, _ = self.lstm(embedding)
        cat     = torch.cat((lstm[:, :, :self.hidden_size], embedding, lstm[:, :, self.hidden_size:]), 2).permute(1, 0, 2)
        linear  = F.tanh(self.fc1(cat)).permute(0, 2, 1)
        pool    = F.max_pool1d(linear, linear.shape[2]).squeeze(2)
        pool    = self.drop(pool)

        out = self.fc2(pool) 
        
        return out

class GRUCNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=128, dropout_prob=0.1):
        super(GRUCNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.embedding_matrix = embedding_matrix
        
        self.embedding = nn.Embedding(self.embedding_matrix.shape[0], self.embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(self.embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.gru  = nn.GRU(self.embedding_matrix.shape[1], self.hidden_size, bidirectional=True)
        self.fc1  = nn.Linear(self.embedding_matrix.shape[1] + 2*self.hidden_size, self.hidden_size)
        self.fc2  = nn.Linear(self.hidden_size, 2)
        self.drop = nn.Dropout(self.dropout_prob)

    def forward(self, x):
        embedding = self.embedding(x).permute(1, 0, 2)

        gru, _ = self.gru(embedding)
        cat    = torch.cat((gru[:, :, :self.hidden_size], embedding, gru[:, :, self.hidden_size:]), 2).permute(1, 0, 2)
        linear = F.tanh(self.fc1(cat)).permute(0, 2, 1)
        pool   = F.max_pool1d(linear, linear.shape[2]).squeeze(2)
        pool   = self.drop(pool)

        out = self.fc2(pool) 
        
        return out
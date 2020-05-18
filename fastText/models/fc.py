import torch
import torch.nn as nn

class FC(nn.Module):
    def __init__(self, embedding_matrix, dropout_prob=0.1):
        super(FC, self).__init__()
        self.dropout_prob = dropout_prob
        self.embedding_matrix = embedding_matrix
        
        self.embedding = nn.Embedding(self.embedding_matrix.shape[0], self.embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(self.embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.drop = nn.Dropout(self.dropout_prob)
        self.fc   = nn.Linear(self.embedding_matrix.shape[1], 2)

    def forward(self, x):
        embedding = self.embedding(x)
        embedding = torch.squeeze(torch.unsqueeze(embedding, 0))

        out = self.drop(embedding[:, 0])
        out = self.fc(out)

        return out
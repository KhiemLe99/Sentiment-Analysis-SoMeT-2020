import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, embedding_matrix, n_filters=128, kernel_sizes=[1, 3, 5], dropout_prob=0.1):
        super(TextCNN, self).__init__()
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.dropout_prob = dropout_prob
        self.embedding_matrix = embedding_matrix

        self.embedding = nn.Embedding(self.embedding_matrix.shape[0], self.embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(self.embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList([nn.Conv2d(1, self.n_filters, (K, self.embedding_matrix.shape[1])) for K in self.kernel_sizes])
        self.drop  = nn.Dropout(self.dropout_prob)
        self.fc    = nn.Linear(self.n_filters * len(self.kernel_sizes), 2)

    def forward(self, x):
        embedding = self.embedding(x)
        embedding = embedding.unsqueeze(1)  

        convs = [conv(embedding) for conv in self.convs] 
        convs = [F.relu(conv).squeeze(3) for conv in convs]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  

        cat = torch.cat(pools, 1)
        cat = self.drop(cat)  
        out = self.fc(cat)  

        return out
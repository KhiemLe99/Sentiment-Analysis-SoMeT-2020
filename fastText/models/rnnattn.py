import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMAttn(nn.Module):
    def __init__(self, embedding_matrix, attention_type='dot', hidden_size=128, dropout_prob=0.1):
        super(LSTMAttn, self).__init__()
        self.attention_type = attention_type
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.embedding_matrix = embedding_matrix
        
        self.embedding = nn.Embedding(self.embedding_matrix.shape[0], self.embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(self.embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        self.lstm = nn.LSTM(self.embedding_matrix.shape[1], self.hidden_size, bidirectional=True, batch_first=True)
        self.fc1  = nn.Linear(2*self.hidden_size, 2*self.hidden_size)
        self.fc2  = nn.Linear(6*self.hidden_size, 2)
        self.drop = nn.Dropout(self.dropout_prob)
    
    def attention(self, lstm, final_hidden_state):
        if self.attention_type == 'dot':
            attention_weights = torch.bmm(lstm, final_hidden_state.unsqueeze(2)).squeeze(2)
        elif self.attention_type == 'general':
            attention_weights = torch.bmm(self.fc1(lstm), final_hidden_state.unsqueeze(2)).squeeze(2)
        attention_weights = F.softmax(attention_weights, 1)
        
        attention = torch.bmm(lstm.transpose(1, 2), attention_weights.unsqueeze(2)).squeeze(2)
        return attention

    def forward(self, x):
        embedding = self.embedding(x)
        embedding = torch.squeeze(torch.unsqueeze(embedding, 0))

        lstm, (hn, _) = self.lstm(embedding)
        final_hn_layer = hn.view(self.lstm.num_layers, self.lstm.bidirectional+1, hn.shape[1], self.hidden_size)[-1, :, :, :]
        final_hidden_state = torch.cat([final_hn_layer[i, :, :] for i in range(final_hn_layer.shape[0])], dim=1)
        attention = self.attention(lstm, final_hidden_state)
        avg_pool = torch.mean(lstm, 1)
        max_pool, _ = torch.max(lstm, 1)
        
        cat = torch.cat((avg_pool, max_pool, attention), 1)
        cat = self.drop(cat)
        cat = self.fc2(cat)

        return cat

class GRUAttn(nn.Module):
    def __init__(self, embedding_matrix, attention_type='dot', hidden_size=128, dropout_prob=0.1):
        super(GRUAttn, self).__init__()
        self.attention_type = attention_type
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.embedding_matrix = embedding_matrix
        
        self.embedding = nn.Embedding(self.embedding_matrix.shape[0], self.embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(self.embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        self.gru  = nn.GRU(self.embedding_matrix.shape[1], self.hidden_size, bidirectional=True, batch_first=True)
        self.fc1  = nn.Linear(2*self.hidden_size, 2*self.hidden_size)
        self.fc2  = nn.Linear(6*self.hidden_size, 2)
        self.drop = nn.Dropout(self.dropout_prob)
    
    def attention(self, gru, final_hidden_state):
        if self.attention_type == 'dot':
            attention_weights = torch.bmm(gru, final_hidden_state.unsqueeze(2)).squeeze(2)
        elif self.attention_type == 'general':
            attention_weights = torch.bmm(self.fc1(gru), final_hidden_state.unsqueeze(2)).squeeze(2)
        attention_weights = F.softmax(attention_weights, 1)
        
        attention = torch.bmm(gru.transpose(1, 2), attention_weights.unsqueeze(2)).squeeze(2)
        return attention

    def forward(self, x):
        embedding = self.embedding(x)
        embedding = torch.squeeze(torch.unsqueeze(embedding, 0))

        gru, (hn) = self.gru(embedding)
        final_hn_layer = hn.view(self.gru.num_layers, self.gru.bidirectional+1, hn.shape[1], self.hidden_size)[-1, :, :, :]
        final_hidden_state = torch.cat([final_hn_layer[i, :, :] for i in range(final_hn_layer.shape[0])], dim=1)
        attention = self.attention(gru, final_hidden_state)
        avg_pool = torch.mean(gru, 1)
        max_pool, _ = torch.max(gru, 1)
        
        cat = torch.cat((avg_pool, max_pool, attention), 1)
        cat = self.drop(cat)
        cat = self.fc2(cat)

        return cat
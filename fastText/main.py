import time
import tqdm
import warnings
import numpy as np
import pandas as pd
import sklearn.model_selection as model_selection
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as utils
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import *
from fastText.models.fc import FC
from fastText.models.rnn import LSTM, GRU
from fastText.models.cnn import TextCNN
from fastText.models.rcnn import LSTMCNN, GRUCNN
from fastText.models.rnnattn import LSTMAttn, GRUAttn
from fastText.models.transformer import Transformer
from fastText.data import load_data, load_embedding

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def onehot_labels(labels):
    label_encoder = LabelEncoder()
    label_encoded = label_encoder.fit_transform(labels)
    label_encoded = label_encoded.reshape(len(label_encoded), 1)

    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(label_encoded)

    return onehot_encoded

def evaluate(model, criterion, dataset, batch_size):
    dataloader = utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    model.to(device)

    running_loss = 0.0
    running_corrects = 0
    total_scores = []
    total_preds = []
    total_labels = []
    for inputs, labels in tqdm.tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        scores = F.softmax(outputs, dim=1)
        preds = torch.max(outputs, 1)[1]
        loss = criterion(outputs.view(-1, 2), labels.view(-1))
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        total_scores += list(scores.cpu().detach().numpy())
        total_preds += list(preds.cpu().numpy())
        total_labels += list(labels.data.cpu().numpy())

    total_loss = running_loss / len(dataset)
    total_acc = running_corrects.double() / len(dataset)
    precision = precision_score(total_labels, total_preds)
    recall = recall_score(total_labels, total_preds)
    f1 = f1_score(total_labels, total_preds)
    auc = roc_auc_score(onehot_labels(total_labels), total_scores)
    print('[TEST]  loss:{:.4f} - acc:{:.4f} - precision:{:.4f} - recall:{:.4f} - f1:{:.4f} - auc:{:.4f}'
          .format(total_loss, total_acc, precision, recall, f1, auc))

def train(head_model, dataset, use_dataset, num_epochs, batch_size, learning_rate):
    train = pd.read_csv('fastText/dataset/' + dataset + '/train.csv')
    if use_dataset == -1:    
        train, valid = model_selection.train_test_split(train, stratify=train['mapped_rating'], test_size=0.2, random_state=2020)
    else:
        train, _ = model_selection.train_test_split(train, stratify=train['mapped_rating'], train_size=use_dataset, random_state=2020)
        train, valid = model_selection.train_test_split(train, stratify=train['mapped_rating'], test_size=0.1, random_state=2020)

    test = pd.read_csv('fastText/dataset/' + dataset + '/test.csv')
    X_train, y_train, X_valid, y_valid, X_test, y_test, word_index = load_data(train, valid, test, num_words=100000, maxlen=100)

    X_train, y_train = torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long)
    X_valid, y_valid = torch.tensor(X_valid, dtype=torch.long), torch.tensor(y_valid, dtype=torch.long)
    X_test, y_test = torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

    train_dataset = utils.data.TensorDataset(X_train, y_train)
    valid_dataset = utils.data.TensorDataset(X_valid, y_valid)
    test_dataset = utils.data.TensorDataset(X_test, y_test)
    print("\nNumber of samples\n" + "=================")
    print("[TRAIN]", len(train_dataset))
    print("[VALID]", len(valid_dataset))
    print("[TEST] ", len(test_dataset))

    train_loader = utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    embedding_path = 'fastText/embedding/cc.vi.300.vec'
    print("\nLoad Embedding ...\n" + '==================\n')
    embedding_matrix = load_embedding(embedding_path, embed_size=300, word_index=word_index)

    if head_model == 'fc':
        model_name = 'fc'
        model = FC(embedding_matrix, dropout_prob=0.1)
    elif head_model == 'lstm':
        model_name = 'lstm'
        model = LSTM(embedding_matrix, hidden_size=128, dropout_prob=0.1)
    elif head_model == 'gru':
        model_name = 'gru'
        model = GRU(embedding_matrix, hidden_size=128, dropout_prob=0.1)
    elif head_model == 'lstm-attn':
        model_name = 'lstm_attn'
        model = LSTMAttn(embedding_matrix, hidden_size=128, dropout_prob=0.1, attention_type='general')
    elif head_model == 'gru-attn':
        model_name = 'gru_attn'
        model = GRUAttn(embedding_matrix, hidden_size=128, dropout_prob=0.1, attention_type='general')
    elif head_model == 'lstm-cnn':
        model_name = 'lstm_cnn'
        model = LSTMCNN(embedding_matrix, hidden_size=128, dropout_prob=0.1)
    elif head_model == 'gru-cnn':
        model_name = 'gru_cnn'
        model = GRUCNN(embedding_matrix, hidden_size=128, dropout_prob=0.1)
    elif head_model == 'cnn':
        model_name = 'cnn'
        model = TextCNN(embedding_matrix, n_filters=128, kernel_sizes=[1, 3, 5], dropout_prob=0.1)
    elif head_model == 'transformer':
        model_name = 'transformer'
        model = Transformer(embedding_matrix, num_layers=2, num_heads=6, maxlen=100, dropout_prob=0.1)
    
    model = model.to(device)
    print("\nModel Architecture\n" + "==================")
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    since = time.time()
    history = {
        'train': {'loss': [], 'acc': []},
        'valid': {'loss': [], 'acc': []},
        'lr': []
    }
    best_acc = 0.0
    best_loss = 0.0
    best_epoch = 1

    print("\nStart training ...\n" + "==================\n")
    num_epochs = num_epochs
    for epoch in range(1, num_epochs + 1):
        head = 'epoch {:2}/{:2}'.format(epoch, num_epochs)
        print(head + '\n' + '-'*(len(head)))

        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm.tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            preds = torch.max(outputs, 1)[1]
            loss = criterion(outputs.view(-1, 2), labels.view(-1))

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        history['train']['loss'].append(epoch_loss)
        history['train']['acc'].append(epoch_acc.item())
        print('{} - loss: {:.4f} acc: {:.4f}'.format('train', epoch_loss, epoch_acc))

        model.eval()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm.tqdm(valid_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            preds = torch.max(outputs, 1)[1]
            loss = criterion(outputs.view(-1, 2), labels.view(-1))
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(valid_dataset)
        epoch_acc = running_corrects.double() / len(valid_dataset)
        history['valid']['loss'].append(epoch_loss)
        history['valid']['acc'].append(epoch_acc.item())
        print('{} - loss: {:.4f} acc: {:.4f}'.format('valid', epoch_loss, epoch_acc)) 

        history['lr'].append(optimizer.param_groups[0]['lr'])

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_loss = epoch_loss
            best_epoch = epoch
            evaluate(model, criterion=nn.CrossEntropyLoss(), dataset=test_dataset, batch_size=batch_size)
            torch.save(model.state_dict(), 'fastText/logs/' + dataset + '/' + model_name + '.pth')

    time_elapsed = time.time() - since
    print('\nTraining time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('[VALID] epoch:{:2} - loss:{:.4f} - acc:{:.4f}'
          .format(best_epoch, best_loss, best_acc))
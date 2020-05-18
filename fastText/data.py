import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def load_data(train, valid, test, num_words, maxlen):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(list(train['discriptions'].astype(str).values) + list(valid['discriptions'].astype(str).values) + list(test['discriptions'].astype(str).values))
    
    train_tokenized = tokenizer.texts_to_sequences(train['discriptions'].astype(str))
    valid_tokenized = tokenizer.texts_to_sequences(valid['discriptions'].astype(str))
    test_tokenized = tokenizer.texts_to_sequences(test['discriptions'].astype(str))
    
    X_train = pad_sequences(train_tokenized, maxlen=maxlen)
    X_valid = pad_sequences(valid_tokenized, maxlen=maxlen)
    X_test = pad_sequences(test_tokenized, maxlen=maxlen)
    
    y_train = train['mapped_rating'].values
    y_valid = valid['mapped_rating'].values
    y_test = test['mapped_rating'].values
        
    return X_train, y_train, X_valid, y_valid, X_test, y_test, tokenizer.word_index

def prepare_labels(labels):
    label_encoder = LabelEncoder()
    label_encoded = label_encoder.fit_transform(labels)
    label_encoded = label_encoded.reshape(len(label_encoded), 1)

    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(label_encoded)

    return onehot_encoded

def load_embedding(embedding_path, embed_size, word_index):
    def get_coefs(word,*arr): 
        return word, np.asarray(arr, dtype='float32')

    embedding_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_path, encoding='utf-8', errors='ignore'))
    
    all_embeds = np.stack(embedding_index.values())
    embed_mean, embed_std = all_embeds.mean(), all_embeds.std()
    embedding_matrix = np.random.normal(embed_mean, embed_std, (len(word_index) + 1, embed_size))

    for word, i in tqdm.tqdm(word_index.items()):
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector

    return embedding_matrix
# Adapted from https://github.com/pytorch/pytorch/issues/39849#issue-636837996
# Requires "Reviews.csv" file, downloaded from here: https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import os
import re
import spacy

from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import random
import numpy as np 
script_dir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description='PyTorch CINIC10 Training')
parser.add_argument('--seed',type=int,  default=60)
parser.add_argument('--epochs',type=int,  default=50)

parser.add_argument('--save_path',type=str,
    default=os.path.join(script_dir, "save_path/"))
parser.add_argument('--batch_size',type=int,  default=1024)

best_acc = 0


class ReviewsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx][1]

def tokenize(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')  # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    return [token.text for token in tok.tokenizer(nopunct)]
def encode_sentence(text, vocab2index, N=70):
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length

def train_model(model, epochs, lr=0.001):
    global best_acc
    stats = []
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for batch_idx,(x, y, l) in enumerate(train_dl):
            x = x.long().cuda()
            y = y.long().cuda()
            l = l.cuda()
            y_pred = model(x,l)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item() * y.shape[0]
            total += y.shape[0]
        val_loss, val_acc  = validation_metrics(model, val_dl)
        if val_acc > best_acc:
           best_acc = val_acc
        # if i % 5 == 1:
        #     print("    train loss %.3f, val loss %.3f, val accuracy %.3f" % (
        #     sum_loss / total, val_loss, val_acc))
        stats.append([sum_loss/ total, val_loss, val_acc])
    return stats


def validation_metrics(model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    for x, y, l in valid_dl:
        x = x.long().cuda()
        y = y.long().cuda()
        l = l.cuda()
        y_hat = model(x, l)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item() * y.shape[0]
        

    return sum_loss / total, correct / total


class LSTM_fixed_len(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 5)

    def forward(self, x, l):
        x = self.embeddings(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])

def run_test(args):
    global tok
    global train_dl
    global val_dl

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)

    csv_file = os.path.join(script_dir, "Reviews.csv")
    reviews = pd.read_csv(csv_file)
    #print(reviews.shape)
    #Replacing Nan values
    reviews['Title'] = reviews['Title'].fillna('')
    reviews['Review Text'] = reviews['Review Text'].fillna('')

    reviews['review'] = reviews['Title'] + ' ' + reviews['Review Text']
    reviews = reviews[['review', 'Rating']]
    reviews.columns = ['review', 'rating']
    reviews['review_length'] = reviews['review'].apply(lambda x: len(x.split()))
    # changing ratings to 0-numbering
    zero_numbering = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    reviews['rating'] = reviews['rating'].apply(lambda x: zero_numbering[x])
    np.mean(reviews['review_length'])
    tok = spacy.load('en_core_web_sm')

    counts = Counter()
    for index, row in reviews.iterrows():
        counts.update(tokenize(row['review']))
    for word in list(counts):
        if counts[word] < 2:
            del counts[word]
    vocab2index = {"":0, "UNK":1}
    words = ["", "UNK"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)

    reviews['encoded'] = reviews['review'].apply(lambda x: np.array(encode_sentence(x,vocab2index )))
    reviews.head()
    Counter(reviews['rating'])
    X = list(reviews['encoded'])
    y = list(reviews['rating'])
    #haluka = int(len(X)*0.8)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=args.seed)#X[:haluka],X[haluka:],y[:haluka],y[haluka:]#
    train_ds = ReviewsDataset(X_train, y_train)
    valid_ds = ReviewsDataset(X_valid, y_valid)

     
    vocab_size = len(words)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(valid_ds, batch_size=args.batch_size)
    model_fixed = LSTM_fixed_len(vocab_size, 50, 50)

    stats = train_model(model_fixed.cuda(), epochs=args.epochs, lr=0.01)
    return best_acc, stats

def check_determinism(args):
    prev_accuracy = None
    prev_stats = None
    for _ in range(5):
        accuracy, stats = run_test(args)
        # print("  %f" % accuracy)
        if prev_accuracy is None:
            prev_accuracy = accuracy
            prev_stats = stats
        else:
            if prev_accuracy != accuracy or prev_stats != stats:
                return 'not deterministic'
    return 'possibly deterministic'

if __name__ == "__main__":
    args = parser.parse_args()
    
    if not os.path.exists(args.save_path):
      os.mkdir(args.save_path)

    print("Before setting deterministic: %s" % check_determinism(args))
    torch.set_deterministic(True)
    print("After setting deterministic: %s" % check_determinism(args))
    torch.set_deterministic(False)
    print("After unsetting deterministic: %s" % check_determinism(args))



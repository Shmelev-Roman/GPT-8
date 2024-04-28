
from transformers import AutoModel, BertTokenizer, BertForSequenceClassification
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

import torch
import transformers
import torch.nn as nn
from transformers import AutoModel, BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

num_class = 30
num_categorical_features = 10

class BERT_Arch(nn.Module):

    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(768, 512)
        self.fc_cat = nn.Linear(num_categorical_features, 50)
        self.fc2 = nn.Linear(512 + 50, num_class)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, sent_id, mask, categorical_features):
        _, cls_hs = self.bert(sent_id, attention_mask = mask, return_dict = False)
        x = self.fc1(cls_hs)
        x = self.relu(x)

        x_cat = self.fc_cat(categorical_features)
        x_cat = self.relu(x_cat)

        x = torch.cat((x, x_cat), dim=1)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.softmax(x)
        return x

def inference(str):
    bert = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased-sentence')
    tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-sentence')

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    X_test = pd.DataFrame([str])

    ppl_cat = Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True, stop_words=None)),
            ('classifier', LogisticRegression(random_state=42, C=5, max_iter=1000))
    ])
    ppl_cat = load('./log_reg_model.joblib') 
    ohe = load('./ohe.joblib')

    y_pred_test = ppl_cat.predict(X_test[0])

    model = BERT_Arch(bert)
    model.load_state_dict(torch.load('./saved_weights.pt', map_location=torch.device(device)))
    model = model.to(device)
    test_categories = ohe.transform(y_pred_test.reshape(-1, 1))

    tokens_test = tokenizer.batch_encode_plus(X_test[0],
                                              max_length = 50,
                                              padding = 'max_length',
                                              truncation = True)

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])

    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.float().to(device), torch.tensor(test_categories).float().to(device))
        preds = preds.detach().cpu().numpy()

    return preds.argmax(axis=1)[0]
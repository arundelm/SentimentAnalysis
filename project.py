#analyze tweet sentiment and use that to determine trends of the S&P 500
#use tweepy as twitter api
 
"""
#Read in twitter dev access keys and tokens
login = pd.read_csv("keys.csv")
consumerKey = login['key'][0]
consumerSecret = login['key'][1]
accessToken = login['key'][2]
accessTokenSecret = login['key'][3]

#Create authentication object
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)

#Set access token and token secret
authenticate.set_access_token(accessToken, accessTokenSecret)

#Create tweepy api object
api = tweepy.API(authenticate, wait_on_rate_limit=True)

x_posts = tweepy.Cursor(api.search_tweets, q="#S&P500",count=100, lang ="en",since="2019-1-1", tweet_mode="extended").items()
data=pd.DataFrame(data=[[post_info.created_at.date(),post_info.full_text]for post_info in x_posts],columns=['Date','Tweets'])

"""

####use sentiment 140 dataset to train BERT for sentiment analysis


import numpy as np
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

#Load and process the sentiment140 dataset which contains pre-processed tweets to train BERT
columns = ['polarity', 'id', 'date', 'query', 'user', 'text']
df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding='latin-1', names=columns)
df.columns=['Sentiment', 'id', 'Date', 'Query', 'User', 'Tweet']
df = df.drop(columns=['id', 'Date', 'Query', 'User'], axis=1)
#Make polarity binary (negative = 0, positive = 1)
df['Sentiment'] = df.Sentiment.replace(4, 1)

#Adjust each tweet and handl accordingly by removing hashtags, mentions, and urls and replacing them with neutral words
hashtags = re.compile(r"^#\S+|\s#\S+")
mentions = re.compile(r"^@\S+|\s@\S+")
urls = re.compile(r"https?://\S+")

def process_text(text):
    text = re.sub(r'http\S+', '', text)
    text = hashtags.sub(' hashtag', text)
    text = mentions.sub(' entity', text)

    return text.strip().lower()
   
df['Tweet'] = df.Tweet.apply(process_text)





from transformers import BertTokenizer,BertForSequenceClassification
from torch.utils.data import DataLoader,SequentialSampler,RandomSampler,TensorDataset,random_split

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case = True)

encoder_train = tokenizer.batch_encode_plus(df.Tweet.values,
                                           add_special_tokens = True,
                                           return_attention_mask = True,
                                           padding=True,
                                           return_tensors = 'pt')


input_ids = encoder_train['input_ids']
attention_masks = encoder_train["attention_mask"]
labels = torch.tensor(df.Sentiment.values)

dataset = TensorDataset(input_ids,attention_masks,labels)
train_size = int(0.80*len(dataset))
test_size = len(dataset) - train_size

train_dataset,test_dataset = random_split(dataset,[train_size,test_size])

print('Training Size - ',train_size)
print('Testing Size - ',test_size)

train_dl = DataLoader(train_dataset,sampler = RandomSampler(train_dataset),
                     batch_size = 32)
test_dl = DataLoader(test_dataset,sampler = SequentialSampler(test_dataset),
                     batch_size = 32)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False)

from transformers import get_linear_schedule_with_warmup, AdamW

optimizer = AdamW(model.parameters(),lr = 1e-5,eps = 1e-8)

epochs  = 1
scheduler = get_linear_schedule_with_warmup(
            optimizer,
    num_warmup_steps = 0,
   num_training_steps = len(train_dl)*epochs 
)

from sklearn.metrics import f1_score 

def f1_score_func(preds,labels):
    preds_flat = np.argmax(preds,axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat,preds_flat,average = 'weighted')

dict_label = {"negative" : 0, "positive" : 1}
def accuracy_per_class(preds,labels):
    label_dict_reverse = {v:k for k,v in dict_label.items()}
    
    preds_flat = np.argmax(preds,axis=1).flatten()
    labels_flat = labels.flatten()
    
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f"Class:{label_dict_reverse}")
        print(f"Accuracy:{len(y_preds[y_preds==label])}/{len(y_true)}\n")

import random

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"Loading:{device}")

from tqdm import tqdm

def evaluate(dataloader_val):
    model.eval()
    
    loss_val_total = 0
    predictions,true_vals = [],[]
    
    for batch in tqdm(dataloader_val):
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':  batch[0],
                  'attention_mask':batch[1],
                  'labels': batch[2]
                 }
        with torch.no_grad():
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total +=loss.item()
        
        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total/len(dataloader_val)  
    
    predictions = np.concatenate(predictions,axis=0)
    true_vals = np.concatenate(true_vals,axis=0) 
    return loss_val_avg,predictions,true_vals

for epoch in tqdm(range(1,epochs+1)):
    model.train()
    
    loss_train_total=0
    
    progress_bar = tqdm(train_dl,desc = "Epoch: {:1d}".format(epoch),leave = False,disable = False)
    
    
    for batch in progress_bar:
        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        inputs = {
            "input_ids":batch[0],
            "attention_mask":batch[1],
            "labels":batch[2]
            
        }
        outputs = model(**inputs)
        
        loss = outputs[0]
#         logits = outputs[1]
        loss_train_total +=loss.item()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm(model.parameters(),1.0)
        
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix({'training_loss':'{:.3f}'.format(loss.item()/len(batch))})

    tqdm.write('\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(train_dl)
    tqdm.write(f'Training Loss: {loss_train_avg}')
    val_loss,predictions,true_vals = evaluate(test_dl)
    test_score = f1_score_func(predictions,true_vals)
    tqdm.write(f'Val Loss:{val_loss}\n Test Score:{test_score}')
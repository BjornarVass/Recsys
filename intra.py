import glob
import unicodedata
import string
import torch
import random
import time
import math

from datahandler import PlainRNNDataHandler
from tester import Tester

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


reddit = "subreddit"
lastfm = "lastfm"

dataset = lastfm

dataset_path = "datasets/" + dataset + "/4_train_test_split.pickle"

BATCHSIZE = 100
HIDDEN_SIZE = 50
N_ITEMS = -1

if dataset == reddit:
    HIDDEN_SIZE = 50
    lr = 0.001
    dropout = 0.0
elif dataset == lastfm:
    HIDDEN_SIZE = 100
    lr = 0.001
    dropout = 0.36 # approximate equal to 2 dropouts of 0.2

#setting of parameters
N_LAYERS = 1
SEQLEN = 20-1
EMBEDDING_SIZE = HIDDEN_SIZE
MAX_EPOCHS = 100
TOP_K = 20

USE_CUDA = True
USE_CUDA_EMBED = True
GPU = 1



dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
torch.manual_seed(42)

datahandler = PlainRNNDataHandler(dataset_path, BATCHSIZE)
N_ITEMS = datahandler.get_num_items()
N_SESSIONS = datahandler.get_num_training_sessions()


class Intra_RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_size, output_size, dropout_rate):
        super(Intra_RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        if(USE_CUDA_EMBED):
            self.embed = nn.Embedding(output_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size)
        self.gru_dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        if(USE_CUDA_EMBED):
            embeddings = self.embed(input)
        else:
            embeddings = input
        out, hidden = self.gru(embeddings, hidden)
        
        out = self.gru_dropout(out)
        out = self.linear(out)
        
        return out, hidden

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if USE_CUDA:
            hidden = hidden.cuda()
        return hidden


#setting up help structures, network and optimizer

#embeddings
#embedding_matrix = nn.Embedding(N_ITEMS, EMBEDDING_SIZE)
#embedding_matrix.weight.data.copy_(torch.zeros(N_ITEMS,EMBEDDING_SIZE).uniform_(-1,1))
#if(USE_CUDA_EMBED):
#   embedding_matrix = embedding_matrix.cuda()

embedding_matrix = None
if(not USE_CUDA_EMBED):
    embedding_matrix = nn.Embedding(N_ITEMS, EMBEDDING_SIZE)
rnn = Intra_RNN(EMBEDDING_SIZE, HIDDEN_SIZE, N_ITEMS, dropout)
if USE_CUDA:
    rnn = rnn.cuda()
if(not USE_CUDA_EMBED):
    embed_optimizer = torch.optim.Adam(embedding_matrix.parameters(), lr=lr)
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)


criterion = nn.CrossEntropyLoss()

#CUSTOM CROSS ENTROPY LOSS(Replace as soon as pytorch has implemented an option for non-summed losses)
#https://github.com/pytorch/pytorch/issues/264
def custom(y_hat, y):
    logp = -F.log_softmax(y_hat)
    logpy = torch.gather(logp,1,y.view(-1,1))
    #logpy = logpy*y.float().sign().view(-1,1) #TEST THIS: new_var = Variable(y.data.float().sign().view(-1,1));logpy = logpy*new_var
    new_var = Variable(y.data.float().sign().view(-1,1))
    logpy = logpy*new_var
    return logpy.view(-1)

def process_batch(xinput, targetvalues):
    training_batch = torch.LongTensor(xinput)
    training_batch = Variable(training_batch.view(SEQLEN, BATCHSIZE))
    targets = torch.LongTensor(targetvalues)
    targets = Variable(targets.view(SEQLEN, BATCHSIZE))
    if(USE_CUDA_EMBED):
        training_batch = training_batch.cuda()
        embedded_data = training_batch
        targets = targets.cuda()
    else:
        embedded_data = embedding_matrix(training_batch)
        if(USE_CUDA):
            targets = targets.cuda()
            embedded_data = embedded_data.cuda()
    return embedded_data, targets

def train_on_batch(xinput, targetvalues, sl):
    optimizer.zero_grad()
    if(not USE_CUDA_EMBED):
        embed_optimizer.zero_grad()
    X, Y = process_batch(xinput, targetvalues)
    hidden = rnn.init_hidden(BATCHSIZE)
    output, _ = rnn(X, hidden)
    reshaped_Y = Y.view(-1)
    reshaped_output = output.view(-1,N_ITEMS)
    reshaped_loss = custom(reshaped_output, reshaped_Y)
    sum_loss = reshaped_loss.sum(0)
    divident = Variable(torch.FloatTensor(1))
    divident[0] = sum(sl)
    if(USE_CUDA):
        divident = divident.cuda()
    mean_loss = sum_loss/divident
    #reshaped_loss.backward(reshaped_Y.data.float().sign())
    mean_loss.backward() #TEST THIS:reshaped_loss.backward(reshaped_Y.data.float().sign())
    optimizer.step()
    if(not USE_CUDA_EMBED):
        embed_optimizer.step()
    return sum_loss.data[0]/sum(sl)

def predict_on_batch(xinput, targetvalues, sl):
    X, Y = process_batch(xinput, targetvalues)
    hidden = rnn.init_hidden(BATCHSIZE)
    output = rnn(X, hidden)
    k_values, k_predictions = torch.topk(output[0], TOP_K)
    return k_predictions

epoch_nr = 0
start_time = time.time()
num_training_batches = datahandler.get_num_training_batches()
num_test_batches = datahandler.get_num_test_batches()
epoch_loss = 0
while epoch_nr < MAX_EPOCHS:
    print("Starting epoch #" + str(epoch_nr))
    start_time_epoch = time.time()
    datahandler.reset_user_batch_data()
    xinput, targetvalues, sl = datahandler.get_next_train_batch()
    batch_nr = 0
    while(len(xinput) > int(BATCHSIZE/2)): #batch_nr < 30 len(xinput) > int(BATCHSIZE/2)
        #Training
        batch_start_time = time.time()
        batch_loss = train_on_batch(xinput, targetvalues, sl)
        epoch_loss += batch_loss
        batch_runtime = time.time() - batch_start_time
        xinput, targetvalues, sl = datahandler.get_next_train_batch()
        if batch_nr%100 == 0:
            print("Batch: " + str(batch_nr) + "/" + str(num_training_batches) + " loss: " + str(batch_loss))
            eta = (batch_runtime*(num_training_batches-batch_nr))/60
            eta = "%.2f" % eta
            print(" | ETA:", eta, "minutes.")
        batch_nr += 1
    print("Epoch loss: " + str(epoch_loss/batch_nr))
    print("Starting testing")
    tester = Tester()
    datahandler.reset_user_batch_data()
    xinput, targetvalues, sl = datahandler.get_next_test_batch()
    #xinput, targetvalues, sl = datahandler.get_next_train_batch()
    batch_nr = 0
    while(len(xinput) > int(BATCHSIZE/2)): #batch_nr < 30 len(xinput) > int(BATCHSIZE/2)
        batch_nr += 1
        batch_start_time = time.time()
        k_predictions = predict_on_batch(xinput, targetvalues, sl)
        k_predictions = k_predictions.view(BATCHSIZE, SEQLEN, TOP_K)
        tester.evaluate_batch(k_predictions, targetvalues, sl)
        xinput, targetvalues, sl = datahandler.get_next_test_batch()
        #xinput, targetvalues, sl = datahandler.get_next_train_batch()
        batch_runtime = time.time() - batch_start_time
        if batch_nr%100 == 0:
            print("Batch: " + str(batch_nr) + "/" + str(num_test_batches))
            eta = (batch_runtime*(num_test_batches-batch_nr))/60
            eta = "%.2f" % eta
            print(" | ETA:", eta, "minutes.")
        
    # Print final test stats for epoch
    test_stats, current_recall5, current_recall20 = tester.get_stats_and_reset()
    print("Recall@5 = " + str(current_recall5))
    print("Recall@20 = " + str(current_recall20))
    print("Epoch #" + str(epoch_nr) + " Time: " + str(time.time()-start_time_epoch))
    epoch_nr += 1
    epoch_loss = 0
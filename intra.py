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

#datasets
reddit = "subreddit"
lastfm = "lastfm"

#set current dataset here
dataset = lastfm
dataset_path = "datasets/" + dataset + "/4_train_test_split.pickle"

#universal settings
BATCHSIZE = 100
N_LAYERS = 1 #currently not used
SEQLEN = 20-1
MAX_EPOCHS = 100
TOP_K = 20

#gpu settings
USE_CUDA = True
USE_CUDA_EMBED = True
GPU = 1 #currently not used

#dataset dependent settings
if dataset == reddit:
    HIDDEN_SIZE = 50
    lr = 0.001
    dropout = 0.0
elif dataset == lastfm:
    HIDDEN_SIZE = 100
    lr = 0.001
    dropout = 0.2 # approximate equal to 2 dropouts of 0.2 TODO: Not really, look at this

EMBEDDING_SIZE = HIDDEN_SIZE

#setting of seed
torch.manual_seed(42) #seed CPU

#loading of dataset into datahandler and getting relevant iformation about the dataset
datahandler = PlainRNNDataHandler(dataset_path, BATCHSIZE)
N_ITEMS = datahandler.get_num_items()
N_SESSIONS = datahandler.get_num_training_sessions()


#intra session RNN module
class Intra_RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_size, output_size, dropout_rate):
        super(Intra_RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        if(USE_CUDA_EMBED):
            self.embed = nn.Embedding(output_size, embedding_dim)
        self.gru_dropout1 = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(embedding_dim, hidden_size)
        self.gru_dropout2 = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        if(USE_CUDA_EMBED):
            embeddings = self.embed(input)
        else:
            embeddings = input
        embeddings = self.gru_dropout1(embeddings)
        out, hidden = self.gru(embeddings, hidden)
        
        out = self.gru_dropout2(out)
        out = self.linear(out)
        
        return out, hidden

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if USE_CUDA:
            hidden = hidden.cuda()
        return hidden


#setting up embedding matrix, network and optimizer
embedding_matrix = None
if(not USE_CUDA_EMBED):
    embedding_matrix = nn.Embedding(N_ITEMS, EMBEDDING_SIZE)
    embed_optimizer = torch.optim.Adam(embedding_matrix.parameters(), lr=lr) #need to cover the parameters in the embedding matrix as well if this is outside the RNN module
#model
rnn = Intra_RNN(EMBEDDING_SIZE, HIDDEN_SIZE, N_ITEMS, dropout)
if USE_CUDA:
    rnn = rnn.cuda()
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)

#library option that does not allow us to mask away 0-paddings and returns a mean by default
criterion = nn.CrossEntropyLoss()

#CUSTOM CROSS ENTROPY LOSS(Replace as soon as pytorch has implemented an option for non-summed losses)
#https://github.com/pytorch/pytorch/issues/264
def masked_cross_entropy_loss(y_hat, y):
    logp = -F.log_softmax(y_hat)
    logpy = torch.gather(logp,1,y.view(-1,1))
    mask = Variable(y.data.float().sign().view(-1,1))
    logpy = logpy*mask
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
            embedded_data = embedded_data.cuda()
            targets = targets.cuda()
    return embedded_data, targets

def train_on_batch(xinput, targetvalues, sl):
	#zero gradients
    optimizer.zero_grad()
    if(not USE_CUDA_EMBED):
        embed_optimizer.zero_grad()

    #get batch from datahandler and turn into tensors of expected format, embed input if embedding not in module (not done on GPU)
    X, Y = process_batch(xinput, targetvalues)

    #get initial hidden state of gru layer and call forward on the module
    hidden = rnn.init_hidden(BATCHSIZE)
    output, _ = rnn(X, hidden)

    #prepare tensors for loss evaluation
    reshaped_Y = Y.view(-1)
    reshaped_output = output.view(-1,N_ITEMS) #[SEQLEN*BATCHSIZE,N_items]

    #call loss function on reshaped data
    reshaped_loss = masked_cross_entropy_loss(reshaped_output, reshaped_Y)

    #get mean loss based on actual number of valid events in batch
    sum_loss = reshaped_loss.sum(0)
    divident = Variable(torch.FloatTensor(1))
    divident[0] = sum(sl)
    if(USE_CUDA):
        divident = divident.cuda()
    mean_loss = sum_loss/divident

    #calculate gradients
    sum_loss.backward()

    #update parameters by using the gradients and optimizers
    optimizer.step()
    if(not USE_CUDA_EMBED):
        embed_optimizer.step()
    return mean_loss.data[0]

def predict_on_batch(xinput, targetvalues, sl):
    X, Y = process_batch(xinput, targetvalues)
    hidden = rnn.init_hidden(BATCHSIZE)
    output = rnn(X, hidden)
    k_values, k_predictions = torch.topk(output[0], TOP_K)
    return k_predictions


#setting up or training
epoch_nr = 0
start_time = time.time()
num_training_batches = datahandler.get_num_training_batches()
num_test_batches = datahandler.get_num_test_batches()
epoch_loss = 0
#epoch loop
while epoch_nr < MAX_EPOCHS:
    print("Starting epoch #" + str(epoch_nr))
    start_time_epoch = time.time()

    #reset state of datahandler and get first training batch
    datahandler.reset_user_batch_data()
    xinput, targetvalues, sl = datahandler.get_next_train_batch()
    batch_nr = 0
    while(len(xinput) > int(BATCHSIZE/2)): #Why is the stopping condition this?
      	#batch training
        batch_start_time = time.time()

        #training call
        batch_loss = train_on_batch(xinput, targetvalues, sl)
        epoch_loss += batch_loss
        batch_runtime = time.time() - batch_start_time

        #get next training batch
        xinput, targetvalues, sl = datahandler.get_next_train_batch()

        #print batch loss and ETA occationally
        if batch_nr%100 == 0:
            print("Batch: " + str(batch_nr) + "/" + str(num_training_batches) + " loss: " + str(batch_loss))
            eta = (batch_runtime*(num_training_batches-batch_nr))/60
            eta = "%.2f" % eta
            print(" | ETA:", eta, "minutes.")
        batch_nr += 1
    #finished training in epoch
    print("Epoch loss: " + str(epoch_loss/batch_nr))
    print("Starting testing")

    #initialize trainer
    tester = Tester()

    #reset state of datahandler and get first test batch
    datahandler.reset_user_batch_data()
    xinput, targetvalues, sl = datahandler.get_next_test_batch()
    batch_nr = 0
    while(len(xinput) > int(BATCHSIZE/2)):
    	#batch testing
        batch_nr += 1
        batch_start_time = time.time()

        #run predictions on test batch
        k_predictions = predict_on_batch(xinput, targetvalues, sl)
        k_predictions = k_predictions.view(BATCHSIZE, SEQLEN, TOP_K)

        #evaluate results
        tester.evaluate_batch(k_predictions, targetvalues, sl)

        #get next test batch
        xinput, targetvalues, sl = datahandler.get_next_test_batch()
        batch_runtime = time.time() - batch_start_time

        #print progress and ETA occationally
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
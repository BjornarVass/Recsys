import glob
import unicodedata
import string
import torch
import random
import time
import math

from datahandler_simple import Simple_RNNDataHandler
from tester import Tester

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from statistics import mean
from logger import Logger

#datasets
reddit = "subreddit"
lastfm = "lastfm"

#set current dataset here
dataset = lastfm
use_hidden = True
dataset_path = "datasets/" + dataset + "/4_train_test_split.pickle"

#universal settings
BATCHSIZE = 100
N_LAYERS = 1 #currently not used
SEQLEN = 20-1
TOP_K = 20
MAX_SESSION_REPRESENTATIONS = 15

#gpu settings
USE_CUDA = True
USE_CUDA_EMBED = True
GPU = 1 #currently not used

#dataset dependent settings
if dataset == reddit:
    HIDDEN_SIZE = 50
    lr = 0.001
    dropout = 0.0
    MAX_EPOCHS = 31
elif dataset == lastfm:
    HIDDEN_SIZE = 100
    lr = 0.001
    dropout = 0.2 # approximate equal to 2 dropouts of 0.2 TODO: Not really, look at this
    MAX_EPOCHS = 50

EMBEDDING_SIZE = HIDDEN_SIZE
INTER_HIDDEN = HIDDEN_SIZE

#setting of seed
torch.manual_seed(0) #seed CPU

#loading of dataset into datahandler and getting relevant iformation about the dataset
datahandler = Simple_RNNDataHandler(dataset_path, BATCHSIZE, MAX_SESSION_REPRESENTATIONS, INTER_HIDDEN)
N_ITEMS = datahandler.get_num_items()
N_SESSIONS = datahandler.get_num_training_sessions()

"""
train, test = datahandler.get_times()
train_full = []
test_full = []
print("Train")
for times in train:
    train_full.extend(times)
    #print(str(min(times)) + "," + str(mean(times)) + "," + str(max(times)))

print("Test")
for times in test:
    test_full.extend(times)
    print(len(times))
    print(str(min(times)) + "," + str(mean(times)) + "," + str(max(times)))
    print(times[0])

print(mean(train_full))
print(mean(test_full))
train_full.extend(test_full)
print(mean(train_full))
"""
#embedding
class Embed(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(Embed, self).__init__()
        self.embedding_table = nn.Embedding(input_size, embedding_size)
        self.embedding_table.weight.data.copy_(torch.zeros(input_size,embedding_size).uniform_(-1,1))
        self.embedding_table.weight.data[0] = torch.zeros(embedding_size) #ensure that the representation of paddings are tensors of zeros, which then easily can be used in an average rep
    
    def forward(self, input):
        output = self.embedding_table(input)
        return output


#inter session RNN module
class Inter_RNN(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(Inter_RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.gru_dropout1 = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.gru_dropout2 = nn.Dropout(dropout_rate)
        self.rate = nn.Parameter(torch.cuda.FloatTensor([0.00001]))
    
    def forward(self, input, hidden, rep_indicies, gaps):
        input = self.gru_dropout1(input)
        output, _ = self.gru(input, hidden)   

        hidden_indices = rep_indicies.view(-1,1,1).expand(output.size(0), 1, output.size(2))
        hidden_out = torch.gather(output,1,hidden_indices)
        hidden_out = hidden_out.squeeze()
        hidden_out = hidden_out.unsqueeze(0)
        hidden_out = self.gru_dropout2(hidden_out)
        hidden_out = torch.exp(-self.rate*gaps.view(-1,1))*hidden_out#*(-rate)
        return hidden_out

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if USE_CUDA:
            hidden = hidden.cuda()
        return hidden

    def get_rate(self):
        return self.rate

#intra session RNN module
class Intra_RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_size, output_size, dropout_rate):
        super(Intra_RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru_dropout1 = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.gru_dropout2 = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden, lengths):
        input = self.gru_dropout1(input)
        out, h = self.gru(input, hidden)
        
        output = self.gru_dropout2(out)
        output = self.linear(output)
        hidden_indices = lengths.view(-1,1,1).expand(out.size(0), 1, out.size(2))
        hidden_out = torch.gather(out,1,hidden_indices)
        hidden_out = hidden_out.squeeze()
        hidden_out = hidden_out.unsqueeze(0)
        return output, hidden_out


#setting up embedding matrix, network and optimizer
embed = Embed(N_ITEMS, EMBEDDING_SIZE)
if(USE_CUDA_EMBED):
    embed = embed.cuda()
embed_optimizer = torch.optim.Adam(embed.parameters(), lr=lr)

#models
inter_rnn = Inter_RNN(HIDDEN_SIZE, dropout)
if(USE_CUDA):
    inter_rnn = inter_rnn.cuda()
inter_optimizer = torch.optim.Adam(inter_rnn.parameters(), lr=lr)


intra_rnn = Intra_RNN(EMBEDDING_SIZE, HIDDEN_SIZE, N_ITEMS, dropout)
if USE_CUDA:
    intra_rnn = intra_rnn.cuda()
intra_optimizer = torch.optim.Adam(intra_rnn.parameters(), lr=lr)

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

def process_batch(xinput, targetvalues, session_reps, sr_sl, session_gaps):
    training_batch = torch.LongTensor(xinput)
    training_batch = Variable(training_batch)
    targets = torch.LongTensor(targetvalues)
    targets = Variable(targets)
    sessions = torch.FloatTensor(session_reps)
    sessions = Variable(sessions)
    gaps = torch.FloatTensor(session_gaps)
    gaps = Variable(gaps)
    if(USE_CUDA_EMBED):
        training_batch = training_batch.cuda()
    if(USE_CUDA):
        targets = targets.cuda()
        sessions = sessions.cuda()
        gaps = gaps.cuda()

    return training_batch, targets, sessions, gaps

def train_on_batch(xinput, targetvalues, sl, session_reps, sr_sl, user_list, session_gaps):
	#zero gradients
    inter_optimizer.zero_grad()
    embed_optimizer.zero_grad()
    intra_optimizer.zero_grad()

    #get batch from datahandler and turn into tensors of expected format, embed input if embedding not in module (not done on GPU)
    X, Y, S, T = process_batch(xinput, targetvalues, session_reps, sr_sl, session_gaps)

    #get initial hidden state of inter gru layer and call forward on the module
    rep_indicies = Variable(torch.LongTensor(sr_sl)) - 1
    if(USE_CUDA):
        rep_indicies = rep_indicies.cuda()
    inter_hidden = inter_rnn.init_hidden(S.size(0))
    hidden = inter_rnn(S, inter_hidden, rep_indicies, T)

    #get embeddings
    embedded_X = embed(X)
    lengths = Variable(torch.FloatTensor(sl).view(-1,1)) #by reshaping the length to this, it can be broadcasted and used for division
    if(USE_CUDA):
        lengths = lengths.cuda()
        if(not USE_CUDA_EMBED):
            embedded_X = embedded_X.cuda()
    sum_X = embedded_X.sum(1)
    mean_X = sum_X.div(lengths)

    #call forward on intra gru layer with hidden state from inter
    lengths = lengths.long() - 1  #length to indices
    output, hidden_out = intra_rnn(embedded_X, hidden, lengths)

    if(use_hidden):
        datahandler.store_user_session_representations(hidden_out.data[0], user_list)
    else:
        datahandler.store_user_session_representations(mean_X.data, user_list)

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
    mean_loss = reshaped_loss.mean(0)#sum_loss/divident

    #calculate gradients
    mean_loss.backward()

    #update parameters by using the gradients and optimizers
    embed_optimizer.step()
    inter_optimizer.step()
    intra_optimizer.step()
    return mean_loss.data[0]

def predict_on_batch(xinput, targetvalues, sl, session_reps, sr_sl, user_list, session_gaps):
    X, Y, S, T = process_batch(xinput, targetvalues, session_reps, sr_sl, session_gaps)

    rep_indicies = Variable(torch.LongTensor(sr_sl)) - 1
    if(USE_CUDA):
        rep_indicies = rep_indicies.cuda()
    inter_hidden = inter_rnn.init_hidden(S.size(0))
    hidden = inter_rnn(S, inter_hidden, rep_indicies, T)
    #get embeddings
    embedded_X = embed(X)
    lengths = Variable(torch.FloatTensor(sl).view(-1,1))
    if(USE_CUDA):
        lengths = lengths.cuda()
        if(not USE_CUDA_EMBED):
            embedded_X = embedded_X.cuda()
    sum_X = embedded_X.sum(1)
    mean_X = sum_X.div(lengths)

    lengths = lengths.long() - 1
    output, hidden_out = intra_rnn(embedded_X, hidden, lengths)

    if(use_hidden):
        datahandler.store_user_session_representations(hidden_out.data[0], user_list)
    else:
        datahandler.store_user_session_representations(mean_X.data, user_list)
    
    k_values, k_predictions = torch.topk(output, TOP_K)
    return k_predictions


#setting up or training
epoch_nr = 0
start_time = time.time()
num_training_batches = datahandler.get_num_training_batches()
num_test_batches = datahandler.get_num_test_batches()
epoch_loss = 0

#tensorboard logger
logger = Logger('./logs')

#epoch loop
while epoch_nr < MAX_EPOCHS:
    print("Starting epoch #" + str(epoch_nr))
    start_time_epoch = time.time()

    #reset state of datahandler and get first training batch
    datahandler.reset_user_batch_data()
    datahandler.reset_user_session_representations()
    xinput, targetvalues, sl, session_reps, sr_sl, user_list, session_gaps = datahandler.get_next_train_batch()
    batch_nr = 0
    intra_rnn.train()
    inter_rnn.train()
    while(len(xinput) > int(BATCHSIZE/2)): #Why is the stopping condition this?
      	#batch training
        batch_start_time = time.time()

        #training call
        batch_loss = train_on_batch(xinput, targetvalues, sl, session_reps, sr_sl, user_list, session_gaps)
        epoch_loss += batch_loss
        batch_runtime = time.time() - batch_start_time

        #get next training batch
        xinput, targetvalues, sl, session_reps, sr_sl, user_list, session_gaps = datahandler.get_next_train_batch()

        #print batch loss and ETA occasionally
        if batch_nr%100 == 0:
            #print("Batch: " + str(batch_nr) + "/" + str(num_training_batches) + " loss: " + str(batch_loss))
            eta = (batch_runtime*(num_training_batches-batch_nr))/60
            eta = "%.2f" % eta
            #print(" | ETA:", eta, "minutes.")
            logger.histo_summary("rate", inter_rnn.get_rate().data.cpu().numpy(), epoch_nr*5)
            logger.histo_summary("rate/grad", inter_rnn.get_rate().grad.data.cpu().numpy(), epoch_nr*5)
            print("Grad: " + str(inter_rnn.get_rate().grad.data.cpu().numpy()))
        batch_nr += 1
    #finished training in epoch
    print("Epoch loss: " + str(epoch_loss/batch_nr))
    print("Starting testing")

    #initialize trainer
    tester = Tester()

    #reset state of datahandler and get first test batch
    datahandler.reset_user_batch_data()
    xinput, targetvalues, sl, session_reps, sr_sl, user_list, session_gaps = datahandler.get_next_test_batch()
    batch_nr = 0
    intra_rnn.eval()
    inter_rnn.eval()
    while(len(xinput) > int(BATCHSIZE/2)):
    	#batch testing
        batch_nr += 1
        batch_start_time = time.time()

        #run predictions on test batch
        k_predictions = predict_on_batch(xinput, targetvalues, sl, session_reps, sr_sl, user_list, session_gaps)

        #evaluate results
        tester.evaluate_batch(k_predictions, targetvalues, sl)

        #get next test batch
        xinput, targetvalues, sl, session_reps, sr_sl, user_list, session_gaps = datahandler.get_next_test_batch()
        batch_runtime = time.time() - batch_start_time

        #print progress and ETA occationally
        if batch_nr%400 == 0:
            #print("Batch: " + str(batch_nr) + "/" + str(num_test_batches))
            eta = (batch_runtime*(num_test_batches-batch_nr))/60
            eta = "%.2f" % eta
            #print(" | ETA:", eta, "minutes.")
        
    # Print final test stats for epoch
    test_stats, current_recall5, current_recall20 = tester.get_stats_and_reset()
    print("Recall@5 = " + str(current_recall5))
    print("Recall@20 = " + str(current_recall20))
    print(test_stats)
    print("Epoch #" + str(epoch_nr) + " Time: " + str(time.time()-start_time_epoch))
    epoch_nr += 1
    epoch_loss = 0
    info = {
        "recall@5": current_recall5,
        "recall@20": current_recall20
    }
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch_nr)
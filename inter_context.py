import glob
import unicodedata
import string
import torch
import random
import time
import math

from datahandler_temporal import RNNDataHandler
from tester_context import Tester

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

#datasets
reddit = "subreddit"
lastfm = "lastfm"
lastfm2 = "lastfm2"
lastfm3 = "lastfm3"

#set current dataset here
dataset = lastfm2
use_hidden = True
dataset_path = "datasets/" + dataset + "/4_train_test_split.pickle"

#universal settings
BATCHSIZE = 100
N_LAYERS = 1 #currently not used
SEQLEN = 20-1
TOP_K = 20
MAX_SESSION_REPRESENTATIONS = 15
TIME_RESOLUTION = 500
TIME_HIDDEN = 20
USER_HIDDEN = 30
USE_DAY = True

#gpu settings
USE_CUDA = True
USE_CUDA_EMBED = True
GPU = 1
torch.cuda.set_device(GPU)

#dataset dependent settings
if dataset == reddit:
    EMBEDDING_SIZE = 50
    lr = 0.001
    dropout = 0.2
    MAX_EPOCHS = 30
    min_time = 1.0
elif dataset == lastfm or dataset == lastfm2:
    EMBEDDING_SIZE = 100
    lr = 0.001
    dropout = 0.2
    MAX_EPOCHS = 25
    min_time = 0.5
elif dataset == lastfm3:
    EMBEDDING_SIZE = 100
    lr = 0.001
    dropout = 0.2
    MAX_EPOCHS = 25
    min_time = 4.0

INTRA_HIDDEN = EMBEDDING_SIZE+TIME_HIDDEN+USER_HIDDEN
if(use_hidden):
    INTER_HIDDEN = INTRA_HIDDEN+TIME_HIDDEN+USER_HIDDEN
    REP_SIZE = INTRA_HIDDEN
else:
    INTER_HIDDEN = INTRA_HIDDEN
    REP_SIZE = EMBEDDING_SIZE

print("Time res: " + str(TIME_RESOLUTION))
print("Hidden: " + str(INTRA_HIDDEN))
print("Time hidden: " + str(TIME_HIDDEN))

#setting of seed
torch.manual_seed(0) #seed CPU

#loading of dataset into datahandler and getting relevant iformation about the dataset
datahandler = RNNDataHandler(dataset_path, BATCHSIZE, MAX_SESSION_REPRESENTATIONS, REP_SIZE, TIME_RESOLUTION, USE_DAY, min_time)
N_ITEMS = datahandler.get_num_items()
N_SESSIONS = datahandler.get_num_training_sessions()
N_USERS = datahandler.get_num_users()

time_threshold = torch.FloatTensor([min_time])
if(USE_DAY):
    time_threshold = time_threshold/24
if(USE_CUDA):
    time_threshold = time_threshold.cuda(GPU)

#embedding
class Embed(nn.Module):
    def __init__(self, input_size, embedding_size, time):
        super(Embed, self).__init__()
        self.embedding_table = nn.Embedding(input_size, embedding_size)
        if(not time):
            self.embedding_table.weight.data[0] = torch.zeros(embedding_size) #ensure that the representation of paddings are tensors of zeros, which then easily can be used in an average rep
    
    def forward(self, input):
        output = self.embedding_table(input)
        return output


#inter session RNN module
class Inter_RNN(nn.Module):
    def __init__(self, hidden_size, dropout_rate, output_size):
        super(Inter_RNN, self).__init__()
        
        self.cat_size = hidden_size
        self.gru_dropout1 = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(self.cat_size, self.cat_size, batch_first=True)
        self.gru_dropout2 = nn.Dropout(dropout_rate)
    
    def forward(self, input, hidden, rep_indicies):
        input = self.gru_dropout1(input)
        output, _ = self.gru(input, hidden)

        hidden_indices = rep_indicies.view(-1,1,1).expand(output.size(0), 1, output.size(2))
        hidden_cat = torch.gather(output,1,hidden_indices)
        hidden_cat = hidden_cat.squeeze()
        hidden_cat = hidden_cat.unsqueeze(0)
        hidden_cat = self.gru_dropout2(hidden_cat)

        return hidden_cat

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(1, batch_size, self.cat_size))
        if USE_CUDA:
            hidden = hidden.cuda(GPU)
        return hidden


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
        out, _ = self.gru(input, hidden)
        
        output = self.gru_dropout2(out)
        output = self.linear(output)
        hidden_indices = lengths.view(-1,1,1).expand(out.size(0), 1, out.size(2))
        hidden_out = torch.gather(out,1,hidden_indices)
        hidden_out = hidden_out.squeeze()
        hidden_out = hidden_out.unsqueeze(0)
        return output, hidden_out


#setting up embedding matrices, networks and optimizers
session_embed = Embed(N_ITEMS, EMBEDDING_SIZE, False)
if(USE_CUDA_EMBED):
    session_embed = session_embed.cuda(GPU)
session_embed_optimizer = torch.optim.Adam(session_embed.parameters(), lr=lr)

time_embed = Embed(TIME_RESOLUTION, TIME_HIDDEN, True)
if(USE_CUDA):
    time_embed = time_embed.cuda(GPU)
time_embed_optimizer = torch.optim.Adam(time_embed.parameters(), lr=lr)

user_embed = Embed(N_USERS, USER_HIDDEN, True)
if(USE_CUDA):
    user_embed = user_embed.cuda(GPU)
user_embed_optimizer = torch.optim.Adam(user_embed.parameters(), lr=lr)

#models
inter_rnn = Inter_RNN(INTER_HIDDEN, dropout, N_ITEMS)
if(USE_CUDA):
    inter_rnn = inter_rnn.cuda(GPU)
inter_optimizer = torch.optim.Adam(inter_rnn.parameters(), lr=lr)

intra_linear = nn.Linear(INTER_HIDDEN,INTRA_HIDDEN)
if(USE_CUDA):
    intra_linear = intra_linear.cuda(GPU)
intra_linear_optimizer = torch.optim.Adam(intra_linear.parameters(), lr=lr)


intra_rnn = Intra_RNN(EMBEDDING_SIZE, INTRA_HIDDEN, N_ITEMS, dropout)
if USE_CUDA:
    intra_rnn = intra_rnn.cuda(GPU)
intra_optimizer = torch.optim.Adam(intra_rnn.parameters(), lr=lr)

#library option that does not allow us to mask away 0-paddings and returns a mean by default
criterion = nn.CrossEntropyLoss()

#CUSTOM CROSS ENTROPY LOSS(Replace as soon as pytorch has implemented an option for non-summed losses)
#https://github.com/pytorch/pytorch/issues/264
def masked_cross_entropy_loss(y_hat, y):
    logp = -F.log_softmax(y_hat, dim=1)
    logpy = torch.gather(logp,1,y.view(-1,1))
    mask = Variable(y.data.float().sign().view(-1,1))
    logpy = logpy*mask
    return logpy.view(-1)

def process_batch(xinput, targetvalues, session_reps, sess_time_reps, first_Y):
    training_batch = torch.LongTensor(xinput)
    training_batch = Variable(training_batch)
    targets = torch.LongTensor(targetvalues)
    targets = Variable(targets)
    sessions = torch.FloatTensor(session_reps)
    sessions = Variable(sessions)
    sess_times = torch.LongTensor(sess_time_reps)
    sess_times = Variable(sess_times)
    first = torch.LongTensor(first_Y)
    first = Variable(first)
    if(USE_CUDA_EMBED):
        training_batch = training_batch.cuda(GPU)
    if(USE_CUDA):
        targets = targets.cuda(GPU)
        sessions = sessions.cuda(GPU)
        sess_times = sess_times.cuda(GPU)
        first = first.cuda(GPU)

    return training_batch, targets, sessions, sess_times, first

def train_on_batch(xinput, targetvalues, sl, session_reps, sr_sl, user_list, sess_time_reps, time_targets, first_Y):
	#zero gradients
    inter_optimizer.zero_grad()
    session_embed_optimizer.zero_grad()
    time_embed_optimizer.zero_grad()
    user_embed_optimizer.zero_grad()
    intra_linear_optimizer.zero_grad()
    intra_optimizer.zero_grad()

    #get batch from datahandler and turn into tensors of expected format, embed input if embedding not in module (not done on GPU)
    X, Y, S, T, First = process_batch(xinput, targetvalues, session_reps, sess_time_reps, first_Y)

    #get embedded times
    embedded_T = time_embed(T)

    #get embedded user
    U = Variable(torch.LongTensor(user_list.tolist()))
    if(USE_CUDA):
        U = U.cuda(GPU)
    embedded_U = user_embed(U)
    embedded_U = embedded_U.unsqueeze(1)
    embedded_U = embedded_U.expand(embedded_U.size(0), embedded_T.size(1), embedded_U.size(2))

    #get initial hidden state of inter gru layer and call forward on the module
    rep_indicies = Variable(torch.LongTensor(sr_sl)) - 1
    if(USE_CUDA):
        rep_indicies = rep_indicies.cuda(GPU)
    inter_hidden = inter_rnn.init_hidden(S.size(0))
    hidden_cat = inter_rnn(torch.cat((S, embedded_T, embedded_U),2), inter_hidden, rep_indicies)
    hidden = intra_linear(hidden_cat)

    #get embeddings
    embedded_X = session_embed(X)
    lengths = Variable(torch.FloatTensor(sl).view(-1,1)) #by reshaping the length to this, it can be broadcasted and used for division
    if(USE_CUDA):
        lengths = lengths.cuda(GPU)
        if(not USE_CUDA_EMBED):
            embedded_X = embedded_X.cuda(GPU)
    sum_X = embedded_X.sum(1)
    mean_X = sum_X.div(lengths)

    #call forward on intra gru layer with hidden state from inter
    lengths = lengths.long()-1
    output, hidden_out = intra_rnn(embedded_X, hidden, lengths)

    if(use_hidden):
        datahandler.store_user_session_representations(hidden_out.data[0], user_list, time_targets)
    else:
        datahandler.store_user_session_representations(mean_X.data, user_list, time_targets)

    #prepare tensors for loss evaluation
    reshaped_Y = Y.view(-1)
    reshaped_output = output.view(-1,N_ITEMS) #[SEQLEN*BATCHSIZE,N_items]

    #call loss function on reshaped data
    reshaped_loss = masked_cross_entropy_loss(reshaped_output, reshaped_Y)


    #get mean loss based on actual number of valid events in batch
    sum_loss = reshaped_loss.sum(0)
    divident = Variable(torch.FloatTensor([sum(sl)]))
    if(USE_CUDA):
        divident = divident.cuda(GPU)
    mean_loss = sum_loss/divident

    #calculate gradients
    combined_loss = mean_loss
    combined_loss.backward()

    #update parameters by using the gradients and optimizers
    intra_optimizer.step() 
    intra_linear_optimizer.step()
    session_embed_optimizer.step()
    time_embed_optimizer.step()
    user_embed_optimizer.step()
    inter_optimizer.step()

    return mean_loss.data[0]

def predict_on_batch(xinput, targetvalues, sl, session_reps, sr_sl, user_list, sess_time_reps, time_targets, first_Y):
    X, Y, S, T, First = process_batch(xinput, targetvalues, session_reps, sess_time_reps, first_Y)
    
    #get embedded times
    embedded_T = time_embed(T)

    U = Variable(torch.LongTensor(user_list.tolist()))
    if(USE_CUDA):
        U = U.cuda(GPU)
    embedded_U = user_embed(U)
    embedded_U = embedded_U.unsqueeze(1)
    embedded_U = embedded_U.expand(embedded_U.size(0), embedded_T.size(1), embedded_U.size(2))

    rep_indicies = Variable(torch.LongTensor(sr_sl)) - 1
    if(USE_CUDA):
        rep_indicies = rep_indicies.cuda(GPU)

    inter_hidden = inter_rnn.init_hidden(S.size(0))
    hidden_cat = inter_rnn(torch.cat((S, embedded_T, embedded_U),2), inter_hidden, rep_indicies)
    hidden = intra_linear(hidden_cat)

    #get embeddings
    embedded_X = session_embed(X)
    lengths = Variable(torch.FloatTensor(sl).view(-1,1))
    if(USE_CUDA):
        lengths = lengths.cuda(GPU)
        if(not USE_CUDA_EMBED):
            embedded_X = embedded_X.cuda(GPU)
    sum_X = embedded_X.sum(1)
    mean_X = sum_X.div(lengths)

    lengths = lengths.long()-1
    output, hidden_out = intra_rnn(embedded_X, hidden, lengths)

    if(use_hidden):
        datahandler.store_user_session_representations(hidden_out.data[0], user_list, time_targets)
    else:
        datahandler.store_user_session_representations(mean_X.data, user_list, time_targets)
    
    k_values, k_predictions = torch.topk(output, TOP_K)
    return k_predictions





#setting up or training
epoch_nr = 0
start_time = time.time()
num_training_batches = datahandler.get_num_training_batches()
num_test_batches = datahandler.get_num_test_batches()
epoch_loss = 0
#epoch loop


#training loop
while epoch_nr < MAX_EPOCHS:
    #Training scheduler:
    print("Starting epoch #" + str(epoch_nr))
    start_time_epoch = time.time()

    #reset state of datahandler and get first training batch
    datahandler.reset_user_batch_data()
    datahandler.reset_user_session_representations()
    xinput, targetvalues, sl, session_reps, sr_sl, user_list, sess_time_reps, time_targets, first_predictions = datahandler.get_next_train_batch() #why twice?
    batch_nr = 0
    intra_rnn.train()
    inter_rnn.train()
    while(len(xinput) > int(BATCHSIZE/2)): #Why is the stopping condition this?
      	#batch training
        batch_start_time = time.time()

        #training call
        batch_loss = train_on_batch(xinput, targetvalues, sl, session_reps, sr_sl, user_list, sess_time_reps, time_targets, first_predictions)
        epoch_loss += batch_loss
        batch_runtime = time.time() - batch_start_time

        #get next training batch
        xinput, targetvalues, sl, session_reps, sr_sl, user_list, sess_time_reps, time_targets, first_predictions = datahandler.get_next_train_batch()

        #print batch loss and ETA occationally
        if batch_nr%3000 == 0:
            print("Batch: " + str(batch_nr) + "/" + str(num_training_batches) + " batch_loss: " + str(batch_loss))
            eta = (batch_runtime*(num_training_batches-batch_nr))/60
            eta = "%.2f" % eta
            print(" | ETA:", eta, "minutes.")
            #print("a: weights: " + str(sorted(list(zip(times.data.cpu().numpy(),time_targets)))))
            #print(time_linear.weight.grad.data.cpu().numpy())
            """
            for tag, value in time_linear.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), epoch_nr*5+batch_nr%300)
                logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch_nr*5+batch_nr%300)
            """

        batch_nr += 1
    #finished training in epoch
    print("Epoch loss: " + str(epoch_loss/batch_nr))
    print("Starting testing")

    #initialize trainerst
    tester = Tester(seslen = SEQLEN)

    #reset state of datahandler and get first test batch
    datahandler.reset_user_batch_data()
    xinput, targetvalues, sl, session_reps, sr_sl, user_list, sess_time_reps, time_targets, first_predictions = datahandler.get_next_test_batch()
    batch_nr = 0
    intra_rnn.eval()
    inter_rnn.eval()
    while(len(xinput) > int(BATCHSIZE/2)):
    	#batch testing
        batch_nr += 1
        batch_start_time = time.time()

        #run predictions on test batch
        k_predictions = predict_on_batch(xinput, targetvalues, sl, session_reps, sr_sl, user_list, sess_time_reps, time_targets, first_predictions)

        #evaluate results
        tester.evaluate_batch(k_predictions, targetvalues, sl)

        #get next test batch
        xinput, targetvalues, sl, session_reps, sr_sl, user_list, sess_time_reps, time_targets, first_predictions = datahandler.get_next_test_batch()
        batch_runtime = time.time() - batch_start_time

        #print progress and ETA occationally
        if batch_nr%800 == 0:
            print("Batch: " + str(batch_nr) + "/" + str(num_test_batches))
            eta = (batch_runtime*(num_test_batches-batch_nr))/60
            eta = "%.2f" % eta
            print(" | ETA:", eta, "minutes.")
        
    # Print final test stats for epoch
    individual_stats, test_stats, current_recall5, current_recall20 = tester.get_stats_and_reset()
    print("Recall@5 = " + str(current_recall5))
    print("Recall@20 = " + str(current_recall20))
    print(test_stats)
    print("\nIdividual scores")
    print(individual_stats)
    print("Epoch #" + str(epoch_nr) + " Time: " + str(time.time()-start_time_epoch))
    epoch_nr += 1
    epoch_loss = 0

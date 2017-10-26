import glob
import unicodedata
import string
import torch
import random
import time
import math

from datahandlerIII import IIIRNNDataHandler
from tester import Tester

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

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
TIME_RESOLUTION = 10000
ALPHA = 0.5

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
datahandler = IIIRNNDataHandler(dataset_path, BATCHSIZE, MAX_SESSION_REPRESENTATIONS, INTER_HIDDEN)
N_ITEMS = datahandler.get_num_items()
N_SESSIONS = datahandler.get_num_training_sessions()
SMALLEST_GAP, GREATEST_GAP = datahandler.get_gap_range()
GAP_STEP = (GREATEST_GAP-SMALLEST_GAP)/TIME_RESOLUTION

SMALLEST_GAP = torch.FloatTensor([SMALLEST_GAP])
GAP_STEP = torch.FloatTensor([GAP_STEP])


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
        
        self.hidden_size = hidden_size*2
        self.gru_dropout1 = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.gru_dropout2 = nn.Dropout(dropout_rate)
        self.linear_hidden = nn.Linear(self.hidden_size, hidden_size)
        self.linear_class = nn.Linear(self.hidden_size, output_size)
        self.linear_time = nn.Linear(self.hidden_size, 1)
        self.b = nn.Parameter(torch.FloatTensor([-0.1]))
    
    def forward(self, input, hidden, rep_indicies):
        input = self.gru_dropout1(input)
        output, _ = self.gru(input, hidden)

        hidden_indices = rep_indicies.view(-1,1,1).expand(output.size(0), 1, output.size(2))
        hidden_cat = torch.gather(output,1,hidden_indices)
        hidden_cat = hidden_cat.squeeze()
        hidden_cat = hidden_cat.unsqueeze(0)
        hidden_cat = self.gru_dropout2(hidden_cat)

        hidden_out = self.linear_hidden(hidden_cat)
        classes = self.linear_class(hidden_cat)
        time = self.linear_time(hidden_cat)
        time = time+self.b
        return classes.squeeze(), time.squeeze(), hidden_out

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if USE_CUDA:
            hidden = hidden.cuda(GPU)
        return hidden

class Time_Loss(nn.Module):
    def __init__(self):
        super(Time_Loss, self).__init__() 
        self.w = nn.Parameter(torch.FloatTensor([-0.1]))
    
    def forward(self, time, target):
        time_exp = torch.exp(time)
        w_target = self.w*target
        exps = (time_exp*(1-torch.exp(w_target)))/self.w
        output = time+w_target+exps
        return -output

    def get_w(self):
        return self.w


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


#setting up embedding matrices, networks and optimizers
session_embed = Embed(N_ITEMS, EMBEDDING_SIZE, False)
if(USE_CUDA_EMBED):
    session_embed = session_embed.cuda(GPU)
session_embed_optimizer = torch.optim.Adam(session_embed.parameters(), lr=lr)

time_embed = Embed(TIME_RESOLUTION, EMBEDDING_SIZE, True)
if(USE_CUDA):
    time_embed = time_embed.cuda(GPU)
time_embed_optimizer = torch.optim.Adam(time_embed.parameters(), lr=lr)

#models
inter_rnn = Inter_RNN(HIDDEN_SIZE, dropout, N_ITEMS)
if(USE_CUDA):
    inter_rnn = inter_rnn.cuda(GPU)
inter_optimizer = torch.optim.Adam(inter_rnn.parameters(), lr=lr)

time_loss_func = Time_Loss()
if(USE_CUDA):
    time_loss_func = time_loss_func.cuda(GPU)
time_optimizer = torch.optim.Adam(time_loss_func.parameters(), lr=0.0001)


intra_rnn = Intra_RNN(EMBEDDING_SIZE, HIDDEN_SIZE, N_ITEMS, dropout)
if USE_CUDA:
    intra_rnn = intra_rnn.cuda(GPU)
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

def process_batch(xinput, targetvalues, session_reps, sess_time_reps, first_Y):
    training_batch = torch.LongTensor(xinput)
    training_batch = Variable(training_batch)
    targets = torch.LongTensor(targetvalues)
    targets = Variable(targets)
    sessions = torch.FloatTensor(session_reps)
    sessions = Variable(sessions)
    sess_times = (torch.FloatTensor(sess_time_reps)-SMALLEST_GAP)/GAP_STEP #normalise,!: add extra penalty if some returns negative values
    sess_times = Variable(sess_times.long())
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
    time_optimizer.zero_grad()
    intra_optimizer.zero_grad()

    #get batch from datahandler and turn into tensors of expected format, embed input if embedding not in module (not done on GPU)
    X, Y, S, T, F = process_batch(xinput, targetvalues, session_reps, sess_time_reps, first_Y)

    #get embedded times
    embedded_T = time_embed(T)

    #get initial hidden state of inter gru layer and call forward on the module
    rep_indicies = Variable(torch.LongTensor(sr_sl)) - 1
    if(USE_CUDA):
        rep_indicies = rep_indicies.cuda(GPU)
    inter_hidden = inter_rnn.init_hidden(BATCHSIZE)
    first_predictions, times, hidden = inter_rnn(torch.cat((S, embedded_T),2), inter_hidden, rep_indicies)

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
    first_loss = masked_cross_entropy_loss(first_predictions, F)


    #get mean loss based on actual number of valid events in batch
    sum_loss = reshaped_loss.sum(0)
    sum_first_loss = first_loss.sum(0)
    divident = Variable(torch.FloatTensor([sum(sl)]))
    if(USE_CUDA):
        divident = divident.cuda(GPU)
    mean_loss = sum_loss/divident
    mean_first_loss = sum_first_loss/mean_X.size(0)


    time_targets = Variable(torch.FloatTensor(time_targets))
    if(USE_CUDA):
        time_targets = time_targets.cuda(GPU)
    time_loss = time_loss_func(times, time_targets)
    time_loss = time_loss.mean(0)

    #calculate gradients
    combined_loss = ALPHA*time_loss+(1-ALPHA)*(mean_loss+mean_first_loss)
    combined_loss.backward()

    w, b = time_loss_func.get_w_and_b()
    print("w: weight: " + str(w.data.cpu().numpy()) + " grad: " + str(w.grad.data.cpu().numpy()) + " b: weight: " + str(b.data.cpu().numpy()) + " grad: " + str(b.grad.data.cpu().numpy()))

    #update parameters by using the gradients and optimizers
    session_embed_optimizer.step()
    time_embed_optimizer.step()
    time_optimizer.step()
    inter_optimizer.step()
    intra_optimizer.step()

    return mean_first_loss.data[0], mean_loss.data[0], time_loss.data[0]

def predict_on_batch(xinput, targetvalues, sl, session_reps, sr_sl, user_list, sess_time_reps, time_targets, first_Y):
    X, Y, S, T, F = process_batch(xinput, targetvalues, session_reps, sess_time_reps, first_Y)
    
    #get embedded times
    embedded_T = time_embed(T)

    rep_indicies = Variable(torch.LongTensor(sr_sl)) - 1
    if(USE_CUDA):
        rep_indicies = rep_indicies.cuda(GPU)

    inter_hidden = inter_rnn.init_hidden(BATCHSIZE)
    first_predictions, times, hidden = inter_rnn(torch.cat((S, embedded_T),2), inter_hidden, rep_indicies)
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
    
    k_values, k_predictions = torch.topk(torch.cat((first_predictions.unsqueeze(1),output),1), TOP_K)
    return k_predictions

def step_val(t, time_exp, w): # exp_a = exp(a): exp(time+w*t + exp(time)-exp(time+w*t)/w) = exp_time_w*exp((exp_time-exp_time_w)/w)
    time_w_exp = time_exp*torch.exp(t*w)
    exp_2 = torch.exp((time_exp-time_w_exp)/w)
    return t*time_w_exp*exp_2

def time_prediction_cuda(time, w): #simpson
    precision = 10000
    n = 2 * precision
    T = 1000 #time units
    dt = T/n
    time_exp = torch.exp(time)
    time_preds = 4*step_val(torch.cuda.FloatTensor([T]),time_exp, w)
    for i in range(1,precision):
        t = (2*i-1)*dt
        time_preds += 4*step_val(torch.cuda.FloatTensor([t]),time_exp, w)
        time_preds += 2*step_val(torch.cuda.FloatTensor([t+dt]),time_exp, w)
    time_preds += 4*step_val(torch.cuda.FloatTensor([T-dt]),time_exp, w)
    time_preds *= torch.cuda.FloatTensor([dt/3])
    return time_preds




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
    datahandler.reset_user_session_representations()
    xinput, targetvalues, sl, session_reps, sr_sl, user_list, sess_time_reps, time_targets, first_predictions = datahandler.get_next_train_batch()
    batch_nr = 0
    while(len(xinput) > int(BATCHSIZE/2)): #Why is the stopping condition this?
      	#batch training
        batch_start_time = time.time()

        #training call
        batch_first_loss, batch_loss, batch_time_loss = train_on_batch(xinput, targetvalues, sl, session_reps, sr_sl, user_list, sess_time_reps, time_targets, first_predictions)
        epoch_loss += batch_loss
        batch_runtime = time.time() - batch_start_time

        #get next training batch
        xinput, targetvalues, sl, session_reps, sr_sl, user_list, sess_time_reps, time_targets, first_predictions = datahandler.get_next_train_batch()

        #print batch loss and ETA occationally
        if batch_nr%100 == 0:
            #print("Batch: " + str(batch_nr) + "/" + str(num_training_batches) + " first loss: " + str(batch_first_loss) + " batch_loss: " + str(batch_loss) + " time loss: " + str(batch_time_loss))
            eta = (batch_runtime*(num_training_batches-batch_nr))/60
            eta = "%.2f" % eta
            #print(" | ETA:", eta, "minutes.")
        batch_nr += 1
    #finished training in epoch
    print("Epoch loss: " + str(epoch_loss/batch_nr))
    print("Starting testing")

    #initialize trainer
    tester = Tester()

    #reset state of datahandler and get first test batch
    datahandler.reset_user_batch_data()
    xinput, targetvalues, sl, session_reps, sr_sl, user_list, sess_time_reps, time_targets, first_predictions = datahandler.get_next_test_batch()
    batch_nr = 0
    while(len(xinput) > int(BATCHSIZE/2)):
    	#batch testing
        batch_nr += 1
        batch_start_time = time.time()

        #run predictions on test batch
        k_predictions = predict_on_batch(xinput, targetvalues, sl, session_reps, sr_sl, user_list, sess_time_reps, time_targets, first_predictions)

        #evaluate results
        full_seqs = []
        for i in range(len(first_predictions)):
            full_seqs.append([first_predictions[i]])
            full_seqs[i].extend(targetvalues[i])
        tester.evaluate_batch(k_predictions, full_seqs, sl)

        #get next test batch
        xinput, targetvalues, sl, session_reps, sr_sl, user_list, sess_time_reps, time_targets, first_predictions = datahandler.get_next_test_batch()
        batch_runtime = time.time() - batch_start_time

        #print progress and ETA occationally
        if batch_nr%100 == 0:
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
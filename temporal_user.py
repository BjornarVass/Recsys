import glob
import unicodedata
import string
import torch
import random
import time
import math

from datahandler_temporal import RNNDataHandler
from tester_temporal import Tester

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
#from logger import Logger

#datasets
reddit = "subreddit"
lastfm = "lastfm"
lastfm2 = "lastfm2"
lastfm3 = "lastfm3"

#set current dataset here
dataset = reddit
use_hidden = True
timeless = False
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
ALPHA = 1.0
BETA = 0.05
USE_DAY = True

#log_name = "2018_dayless1"

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
    MAX_EPOCHS = 29
    min_time = 1.0
    freeze = False
elif dataset == lastfm or dataset == lastfm2:
    EMBEDDING_SIZE = 100
    lr = 0.001
    dropout = 0.2
    MAX_EPOCHS = 25
    min_time = 0.5
    freeze = False
elif dataset == lastfm3:
    EMBEDDING_SIZE = 120
    lr = 0.001
    dropout = 0.2
    MAX_EPOCHS = 25
    min_time = 4.0
    freeze = True

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
print("ALPHA: " + str(ALPHA))
print("BETA: " + str(BETA))

#setting of seed
torch.manual_seed(1) #seed CPU

#loading of dataset into datahandler and getting relevant iformation about the dataset
datahandler = RNNDataHandler(dataset_path, BATCHSIZE, MAX_SESSION_REPRESENTATIONS, REP_SIZE, TIME_RESOLUTION, USE_DAY, min_time)
N_ITEMS = datahandler.get_num_items()
N_SESSIONS = datahandler.get_num_training_sessions()
N_USERS = datahandler.get_num_users()

integration_acc = torch.cuda.FloatTensor([0])
integration_count = 0
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
    
    def forward(self, input, hidden, context, rep_indicies):
        input = self.gru_dropout1(input)
        output, _ = self.gru(input, hidden)

        hidden_indices = rep_indicies.view(-1,1,1).expand(output.size(0), 1, output.size(2))
        hidden_cat = torch.gather(output,1,hidden_indices)
        hidden_cat = hidden_cat.squeeze()
        hidden_cat = hidden_cat.unsqueeze(0)
        hidden_cat = self.gru_dropout2(hidden_cat)

        return F.relu(hidden_cat)

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(1, batch_size, self.cat_size))
        if USE_CUDA:
            hidden = hidden.cuda(GPU)
        return hidden

class Time_Loss(nn.Module):
    def __init__(self):
        super(Time_Loss, self).__init__() 
        self.w = nn.Parameter(torch.FloatTensor([-0.1]))
        #self.w.data.uniform_(-0.1,0.1)
    
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

time_linear = nn.Linear(INTER_HIDDEN,1)
if(USE_CUDA):
    time_linear = time_linear.cuda(GPU)
time_linear_optimizer = torch.optim.Adam(time_linear.parameters(), lr=lr)

first_linear = nn.Linear(INTER_HIDDEN,N_ITEMS)
if(USE_CUDA):
    first_linear = first_linear.cuda(GPU)
first_linear_optimizer = torch.optim.Adam(first_linear.parameters(), lr=lr)

intra_linear = nn.Linear(INTER_HIDDEN,INTRA_HIDDEN)
if(USE_CUDA):
    intra_linear = intra_linear.cuda(GPU)
intra_linear_optimizer = torch.optim.Adam(intra_linear.parameters(), lr=lr)

time_loss_func = Time_Loss()
if(USE_CUDA):
    time_loss_func = time_loss_func.cuda(GPU)
time_optimizer = torch.optim.Adam(time_loss_func.parameters(), lr=0.1*lr)


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

def step_val(t, time_exp, w, dt): # exp_a = exp(a): exp(time+w*t + exp(time)-exp(time+w*t)/w) = exp_time_w*exp((exp_time-exp_time_w)/w)
    global integration_acc
    time_w_exp = time_exp*torch.exp(t*w)
    exp_2 = torch.exp((time_exp-time_w_exp)/w)
    prob = time_w_exp*exp_2
    integration_acc += prob.mean(0)[0]*dt
    return t*prob

def time_prediction_cuda(time, w): #simpson
    global integration_count
    integration_count += 1
    precision = 3000
    T = 700 #time units
    part1 = 100
    part2 = 600
    if(USE_DAY):
        T = T/24
        part1 = part1/24
        part2 = part2/24
    T = torch.cuda.FloatTensor([T])
    dt1 = torch.cuda.FloatTensor([part1/precision])
    dt2 = torch.cuda.FloatTensor([part2/precision])
    part1 = torch.cuda.FloatTensor([part1])
    time_exp = torch.exp(time)
    time_preds1 = step_val(part1,time_exp, w, dt1)
    time_preds2 = step_val(T,time_exp, w, dt2) + time_preds1
    for i in range(1,precision//2):
        t = (2*i-1)*dt1
        time_preds1 += 4*step_val(t,time_exp, w, dt1)
        time_preds1 += 2*step_val(t+dt1,time_exp, w, dt1)
    time_preds1 += 4*step_val(part1-dt1,time_exp, w, dt1)
    for i in range(1,precision//2):
        t = (2*i-1)*dt2 + part1
        time_preds2 += 4*step_val(t,time_exp, w, dt2)
        time_preds2 += 2*step_val(t+dt2,time_exp, w, dt2)
    time_preds2 += 4*step_val(T-dt2,time_exp,w,dt2)
    time_preds1 *= dt1/3
    time_preds2 *= dt2/3
    return time_preds1+time_preds2


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
    time_linear_optimizer.zero_grad()
    first_linear_optimizer.zero_grad()
    intra_linear_optimizer.zero_grad()
    time_optimizer.zero_grad()
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
    inter_context = inter_rnn.init_hidden(S.size(0))
    hidden_cat = inter_rnn(torch.cat((S, embedded_T, embedded_U),2), inter_hidden, inter_context, rep_indicies)
    times = time_linear(hidden_cat).squeeze()
    hidden = intra_linear(hidden_cat)
    first_predictions = first_linear(hidden_cat).squeeze()

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
    first_loss = masked_cross_entropy_loss(first_predictions, First)


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
    mask = Variable(time_targets.data.ge(time_threshold).float())
    time_loss = time_loss*mask
    non_zero_count = 0
    for sign in mask.data:
        if (sign != 0):
            non_zero_count += 1
    time_loss_discount = Variable(torch.FloatTensor([max(non_zero_count,1)]))
    if(USE_CUDA):
        time_loss_discount = time_loss_discount.cuda(GPU)
    mean_time_loss = time_loss.sum(0)/time_loss_discount

    #calculate gradients
    combined_loss = (1-ALPHA)*((1-BETA)*mean_loss+BETA*mean_first_loss)+ALPHA*mean_time_loss
    combined_loss.backward()

    #update parameters by using the gradients and optimizers
    if(train_time):
        time_optimizer.step()
        time_linear_optimizer.step()
    if(train_first):
        first_linear_optimizer.step()
    if(train_all):
        intra_optimizer.step() 
        intra_linear_optimizer.step()
        session_embed_optimizer.step()
        time_embed_optimizer.step()
        user_embed_optimizer.step()
        inter_optimizer.step()

    return mean_first_loss.data[0], mean_loss.data[0], mean_time_loss.data[0], times

def predict_on_batch(xinput, targetvalues, sl, session_reps, sr_sl, user_list, sess_time_reps, time_targets, first_Y, time_error):
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
    inter_context = inter_rnn.init_hidden(S.size(0))
    hidden_cat = inter_rnn(torch.cat((S, embedded_T, embedded_U),2), inter_hidden, inter_context, rep_indicies)
    times = time_linear(hidden_cat).squeeze()
    hidden = intra_linear(hidden_cat)
    first_predictions = first_linear(hidden_cat).squeeze()

    if(time_error):
        w = time_loss_func.get_w()
        time_predictions = time_prediction_cuda(times.data, w.data)
        tester.evaluate_batch_time(time_predictions, time_targets)

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





#setting up or training
epoch_nr = 0
start_time = time.time()
num_training_batches = datahandler.get_num_training_batches()
num_test_batches = datahandler.get_num_test_batches()
epoch_loss = 0
#epoch loop

#tensorboard logger
#logger = Logger('./tensorboard/'+log_name)

train_time = True
train_first = True
train_all = True

#training loop
while epoch_nr < MAX_EPOCHS:
    #Training scheduler:
    if(timeless):
        if(epoch_nr == 0):
            ALPHA = 0.0
            BETA = 0.0
        if(epoch_nr == 5):
            BETA = 1.0
        if(epoch_nr == 7):
            BETA = 0.3
        if(epoch_nr == 21):
            train_all = False
            BETA = 1.0
    else:
        if(epoch_nr == 4):
            ALPHA = 0.0
            BETA = 0.0
        if(epoch_nr == 8):
            BETA = 1.0
        if(epoch_nr == 10):
            ALPHA = 0.5
        if(epoch_nr == 11):
            BETA = 0.0
        if(epoch_nr == 12):
            ALPHA = 0.5
            BETA = 0.3
        if(freeze):
            if(epoch_nr == 21):
                train_all = False
                train_first = False
                ALPHA = 1.0
            if(epoch_nr == 24):
                train_first = True
                train_time = False
                ALPHA = 0.0
                BETA = 1.0
    print("Starting epoch #" + str(epoch_nr))
    start_time_epoch = time.time()

    #reset state of datahandler and get first training batch
    datahandler.reset_user_batch_data()
    datahandler.reset_user_session_representations()
    xinput, targetvalues, sl, session_reps, sr_sl, user_list, sess_time_reps, time_targets, first_predictions = datahandler.get_next_train_batch()
    xinput, targetvalues, sl, session_reps, sr_sl, user_list, sess_time_reps, time_targets, first_predictions = datahandler.get_next_train_batch() #why twice?
    batch_nr = 0
    intra_rnn.train()
    inter_rnn.train()
    while(len(xinput) > int(BATCHSIZE/2)): #Why is the stopping condition this?
        #batch training
        batch_start_time = time.time()

        #training call
        batch_first_loss, batch_loss, batch_time_loss, times = train_on_batch(xinput, targetvalues, sl, session_reps, sr_sl, user_list, sess_time_reps, time_targets, first_predictions)
        epoch_loss += batch_loss
        batch_runtime = time.time() - batch_start_time

        #get next training batch
        xinput, targetvalues, sl, session_reps, sr_sl, user_list, sess_time_reps, time_targets, first_predictions = datahandler.get_next_train_batch()

        #print batch loss and ETA occationally
        if batch_nr%1500 == 0:
            print("Batch: " + str(batch_nr) + "/" + str(num_training_batches) + " first loss: " + str(batch_first_loss) + " batch_loss: " + str(batch_loss) + " time loss: " + str(batch_time_loss))
            eta = (batch_runtime*(num_training_batches-batch_nr))/60
            eta = "%.2f" % eta
            print(" | ETA:", eta, "minutes.")
            w = time_loss_func.get_w()
            #print("w: weight: " + str(w.data.cpu().numpy()) + " grad: " + str(w.grad.data.cpu().numpy()))
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

    #initialize trainer
    tester = Tester(seqlen = SEQLEN, use_day = USE_DAY, min_time = min_time)

    #reset state of datahandler and get first test batch
    datahandler.reset_user_batch_data()
    xinput, targetvalues, sl, session_reps, sr_sl, user_list, sess_time_reps, time_targets, first_predictions = datahandler.get_next_test_batch()
    batch_nr = 0
    if( epoch_nr != 0 and epoch_nr%12 == 0):
        time_error = True
    else:
        time_error = False
    intra_rnn.eval()
    inter_rnn.eval()
    while(len(xinput) > int(BATCHSIZE/2)):
        #batch testing
        batch_nr += 1
        batch_start_time = time.time()

        #run predictions on test batch
        k_predictions = predict_on_batch(xinput, targetvalues, sl, session_reps, sr_sl, user_list, sess_time_reps, time_targets, first_predictions, time_error)

        #evaluate results
        tester.evaluate_batch(k_predictions[:,1:], targetvalues, sl, k_predictions[:,0], first_predictions)

        #get next test batch
        xinput, targetvalues, sl, session_reps, sr_sl, user_list, sess_time_reps, time_targets, first_predictions = datahandler.get_next_test_batch()
        batch_runtime = time.time() - batch_start_time

        #print progress and ETA occationally
        if batch_nr%600 == 0:
            print("Batch: " + str(batch_nr) + "/" + str(num_test_batches))
            eta = (batch_runtime*(num_test_batches-batch_nr))/60
            eta = "%.2f" % eta
            print(" | ETA:", eta, "minutes.")
        
    # Print final test stats for epoch
    test_stats, current_recall5, current_recall20, time_stats, time_output = tester.get_stats_and_reset()
    print("Recall@5 = " + str(current_recall5))
    print("Recall@20 = " + str(current_recall20))
    print(test_stats)
    print("\n")
    print(time_stats)
    print("Epoch #" + str(epoch_nr) + " Time: " + str(time.time()-start_time_epoch))
    print("Integration accuracy:" + str(integration_acc[0]/max(integration_count,1)))
    epoch_nr += 1
    epoch_loss = 0
    integration_acc = torch.cuda.FloatTensor([0])
    integration_count = 0
    """
    info = {
        "recall@5": current_recall5,
        "recall@20": current_recall20
    }
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch_nr)

    if(time_error):
        logger.scalar_summary("mae", time_output, epoch_nr)
    """
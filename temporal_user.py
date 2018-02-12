import glob
import unicodedata
import string
import torch
import random
import time
import math

from datahandler_temporal import RNNDataHandler
from tester_temporal import Tester
from modules import Embed, Intra_RNN, Inter_RNN, Time_Loss

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
flags = {}
flags["use_hidden"] = True
flags["context"] = False
flags["temporal"] = False
dataset_path = "datasets/" + dataset + "/4_train_test_split.pickle"

if(flags["temporal"]):
    model = "temporal"
elif(flags["context"]):
    model = "context"
else:
    model = "inter"

#universal settings
BATCHSIZE = 100
N_LAYERS = 1 #currently not used
SEQLEN = 20-1
TOP_K = 20
MAX_SESSION_REPRESENTATIONS = 15
TIME_RESOLUTION = 500
TIME_HIDDEN = 20
USER_HIDDEN = 30

params = {}
params["ALPHA"] = 1.0
params["BETA"] = 0.05
flags["use_day"] = True

#gpu and seed settings
SEED = 1
GPU = 1
torch.cuda.set_device(GPU)
torch.manual_seed(SEED)

#log name
log_name = model + "_" + dataset + str(SEED)

#dataset dependent settings
if dataset == reddit:
    EMBEDDING_DIM = 50
    lr = 0.001
    dropout = 0.2 if flags["context"] else 0.0
    MAX_EPOCHS = 29
    min_time = 1.0
    flags["freeze"] = False
elif dataset == lastfm or dataset == lastfm2:
    EMBEDDING_DIM = 100
    lr = 0.001
    dropout = 0.2
    MAX_EPOCHS = 25
    min_time = 0.5
    flags["freeze"] = False
elif dataset == lastfm3:
    EMBEDDING_DIM = 100
    lr = 0.001
    dropout = 0.2
    MAX_EPOCHS = 25
    min_time = 4.0
    flags["freeze"] = True

#dimensionalities
INTRA_HIDDEN = EMBEDDING_DIM
if(flags["context"]):
    INTER_INPUT_DIM = INTRA_HIDDEN + TIME_HIDDEN + USER_HIDDEN
else:
    INTER_INPUT_DIM = INTRA_HIDDEN

INTER_HIDDEN = INTRA_HIDDEN + TIME_HIDDEN + USER_HIDDEN

#loading of dataset into datahandler and getting relevant iformation about the dataset
datahandler = RNNDataHandler(dataset_path, BATCHSIZE, MAX_SESSION_REPRESENTATIONS, INTRA_HIDDEN, TIME_RESOLUTION, flags["use_day"], min_time)
N_ITEMS = datahandler.get_num_items()
N_SESSIONS = datahandler.get_num_training_sessions()
N_USERS = datahandler.get_num_users()

if(flags["temporal"]):
    integration_acc = torch.cuda.FloatTensor([0])
    integration_count = torch.cuda.FloatTensor([0])
    time_threshold = torch.cuda.FloatTensor([min_time])
else:
    integration_acc = 0
    integration_count = 0
    time_threshold = 0


#initialize lists to contain the parameters in two sub-nets
inter_intra_params = []
time_params = []

#setting up embedding matrices
item_embed = Embed(N_ITEMS, EMBEDDING_DIM, False)
item_embed = item_embed.cuda(GPU)
inter_intra_params += list(item_embed.parameters())

if(flags["context"]):
    time_embed = Embed(TIME_RESOLUTION, TIME_HIDDEN, True)
    time_embed = time_embed.cuda(GPU)
    inter_intra_params += list(time_embed.parameters())

    user_embed = Embed(N_USERS, USER_HIDDEN, True)
    user_embed = user_embed.cuda(GPU)
    inter_intra_params += list(user_embed.parameters())


#setting up models with optimizers
inter_rnn = Inter_RNN(INTER_INPUT_DIM, INTER_HIDDEN, dropout)
inter_rnn = inter_rnn.cuda(GPU)
inter_intra_params += list(inter_rnn.parameters())

intra_rnn = Intra_RNN(EMBEDDING_DIM, INTRA_HIDDEN, N_ITEMS, dropout)
intra_rnn = intra_rnn.cuda(GPU)
inter_intra_params += list(intra_rnn.parameters())

#setting up linear layers for the time loss, first recommendation loss and inter RNN
if(flags["temporal"]):
    time_linear = nn.Linear(INTER_HIDDEN,1)
    time_linear = time_linear.cuda(GPU)
    time_params += [{"params": time_linear.parameters()}]

    first_linear = nn.Linear(INTER_HIDDEN,N_ITEMS)
    first_linear = first_linear.cuda(GPU)

intra_linear = nn.Linear(INTER_HIDDEN,INTRA_HIDDEN)
intra_linear = intra_linear.cuda(GPU)
inter_intra_params += list(intra_linear.parameters())

#setting up time loss model
if(flags["temporal"]):
    time_loss_func = Time_Loss()
    time_loss_func = time_loss_func.cuda(GPU)
    time_params += [{"params": time_loss_func.parameters(), "lr": 0.1*lr}]

#setting up optimizers
inter_intra_optimizer = torch.optim.Adam(inter_intra_params, lr=lr)
if(flags["temporal"]):
    time_optimizer = torch.optim.Adam(time_params, lr=lr)
    first_rec_optimizer = torch.optim.Adam(first_linear.parameters(), lr=lr)

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

#step function implementing the equantion:
#exp(time+w*t + exp(time)-exp(time+w*t)/w) = exp_time_w*exp((exp_time-exp_time_w)/w)
def step_val(t, time_exp, w, dt, integration_acc): 
    time_w_exp = time_exp*torch.exp(t*w)
    exp_2 = torch.exp((time_exp-time_w_exp)/w)
    prob = time_w_exp*exp_2
    integration_acc += prob.mean(0)[0]*dt
    return t*prob

#simpson numerical integration with higher resolution in the first 100 hours
def time_prediction(time, w, integration_count, integration_acc, flags):
    #integration settings
    integration_count += 1
    precision = 3000
    T = 700 #time units
    part1 = 100
    part2 = 600
    if(flags["use_day"]):
        T = T/24
        part1 = part1/24
        part2 = part2/24

    #moving data structures to the GPU for efficiency
    T = torch.cuda.FloatTensor([T])
    dt1 = torch.cuda.FloatTensor([part1/precision])
    dt2 = torch.cuda.FloatTensor([part2/precision])
    part1 = torch.cuda.FloatTensor([part1])
    
    #integration loops
    time_exp = torch.exp(time)
    time_preds1 = step_val(part1,time_exp, w, dt1, integration_acc)
    time_preds2 = step_val(T,time_exp, w, dt2, integration_acc) + time_preds1
    for i in range(1,precision//2):#high resolution loop
        t = (2*i-1)*dt1
        time_preds1 += 4*step_val(t,time_exp, w, dt1, integration_acc)
        time_preds1 += 2*step_val(t+dt1,time_exp, w, dt1, integration_acc)
    time_preds1 += 4*step_val(part1-dt1,time_exp, w, dt1, integration_acc)
    for i in range(1,precision//2):#rough resolution loop
        t = (2*i-1)*dt2 + part1
        time_preds2 += 4*step_val(t,time_exp, w, dt2, integration_acc)
        time_preds2 += 2*step_val(t+dt2,time_exp, w, dt2, integration_acc)
    time_preds2 += 4*step_val(T-dt2,time_exp,w,dt2, integration_acc)

    #division moved to the end for efficiency
    time_preds1 *= dt1/3
    time_preds2 *= dt2/3

    return time_preds1+time_preds2

#create GPU variables of the batch data
def process_batch_inputs(items, session_reps, sess_time_reps, user_list):
    sessions = Variable(torch.cuda.FloatTensor(session_reps))
    items = Variable(torch.cuda.LongTensor(items))
    sess_gaps = Variable(torch.cuda.LongTensor(sess_time_reps))
    users = Variable(torch.cuda.LongTensor(user_list.tolist()))
    return items, sessions, sess_gaps, users

def process_batch_targets( item_targets, time_targets, first_rec_targets):
    item_targets = Variable(torch.cuda.LongTensor(item_targets))
    time_targets = Variable(torch.cuda.FloatTensor(time_targets)) 
    first = Variable(torch.cuda.LongTensor(first_rec_targets))
    return item_targets, time_targets, first

def train_on_batch(items, session_reps, sess_time_reps, user_list, item_targets, time_targets, first_rec_targets, session_lengths, session_rep_lengths, flags, params, time_threshold):
    #zero gradients before each epoch
    inter_intra_optimizer.zero_grad()
    if(flags["temporal"]):
        time_optimizer.zero_grad()
        first_rec_optimizer.zero_grad()

    #get batch from datahandler and turn into variables
    X, S, S_gaps, U = process_batch_inputs(items, session_reps, sess_time_reps, user_list)
    Y, T_targets, First_targets = process_batch_targets(item_targets, time_targets, first_rec_targets)

    if(flags["context"]):
        #get embedded times
        embedded_S_gaps = time_embed(S_gaps)

        #get embedded user
        embedded_U = user_embed(U)
        embedded_U = embedded_U.unsqueeze(1)
        embedded_U = embedded_U.expand(embedded_U.size(0), embedded_S_gaps.size(1), embedded_U.size(2))

    #get the index of the last session representation of each user by subtracting 1 from each lengths, move to GPU for efficiency
    rep_indicies = Variable(torch.cuda.LongTensor(session_rep_lengths)) - 1

    #get initial hidden state of inter gru layer and call forward on the module
    inter_hidden = inter_rnn.init_hidden(S.size(0))
    if(flags["context"]):
        inter_last_hidden = inter_rnn(torch.cat((S, embedded_S_gaps, embedded_U),2), inter_hidden, rep_indicies)
    else:
        inter_last_hidden = inter_rnn(S, inter_hidden, rep_indicies)

    #get initial hidden for inter RNN, time scores and first prediction scores from the last hidden state of the inter RNN
    hidden = intra_linear(inter_last_hidden)
    if(flags["temporal"]):
        times = time_linear(inter_last_hidden).squeeze()
        first_predictions = first_linear(inter_last_hidden).squeeze()

    #get item embeddings
    embedded_X = item_embed(X)

    #create average pooling session representation using the item embeddings and the lenght of each sequence
    lengths = Variable(torch.cuda.FloatTensor(session_lengths).view(-1,1)) #reshape the lengths in order to broadcast and use it for division
    sum_X = embedded_X.sum(1)
    mean_X = sum_X.div(lengths)

    #subtract 1 from the lengths to get the index of the last item in each sequence
    lengths = lengths.long()-1

    #call forward on the inter RNN
    recommendation_output, hidden_out = intra_rnn(embedded_X, hidden, lengths)

    #store the new session representation based on the current scheme
    if(flags["use_hidden"]):
        datahandler.store_user_session_representations(hidden_out.data[0], user_list, time_targets)
    else:
        datahandler.store_user_session_representations(mean_X.data, user_list, time_targets)

    # LOSSES
    #prepare tensors for recommendation loss evaluation
    reshaped_Y = Y.view(-1)
    reshaped_rec_output = recommendation_output.view(-1,N_ITEMS) #[SEQLEN*BATCHSIZE,N_items]

    #calculate recommendation losses
    reshaped_rec_loss = masked_cross_entropy_loss(reshaped_rec_output, reshaped_Y)
    #get mean losses based on actual number of valid events in batch
    sum_loss = reshaped_rec_loss.sum(0)
    divident = Variable(torch.cuda.FloatTensor([sum(session_lengths)]))
    mean_loss = sum_loss/divident

    if(flags["temporal"]):
        first_loss = masked_cross_entropy_loss(first_predictions, First_targets)
        sum_first_loss = first_loss.sum(0)
        mean_first_loss = sum_first_loss/embedded_X.size(0)


        #calculate the time loss
        time_loss = time_loss_func(times, T_targets)

        #mask out "fake session time-gaps" from time loss
        mask = Variable(T_targets.data.ge(time_threshold).float())
        time_loss = time_loss*mask

        #find number of non-ignored time gaps
        non_zero_count = 0
        for sign in mask.data:
            if (sign != 0):
                non_zero_count += 1
        time_loss_divisor = Variable(torch.cuda.FloatTensor([max(non_zero_count,1)]))
        mean_time_loss = time_loss.sum(0)/time_loss_divisor

        #calculate gradients
        combined_loss = (1-params["ALPHA"])*((1-params["BETA"])*mean_loss+params["BETA"]*mean_first_loss)+params["ALPHA"]*mean_time_loss
        combined_loss.backward()

        #update parameters through BPTT, options for freezing parts of the network
        if(flags["train_time"]):
            time_optimizer.step()
        if(flags["train_first"]):
            first_rec_optimizer.step()
        if(flags["train_all"]):
            inter_intra_optimizer.step()
    else:
        mean_loss.backward()
        inter_intra_optimizer.step()
    

    return mean_loss.data[0]

def predict_on_batch(items, session_reps, sess_time_reps, user_list, item_targets, time_targets, first_rec_targets, session_lengths, session_rep_lengths, flags, integration_count, integration_acc):

    #get batch from datahandler and turn into variables
    X, S, S_gaps, U = process_batch_inputs(items, session_reps, sess_time_reps, user_list)

    #get embedded times
    if(flags["context"]):
        #get embedded times
        embedded_S_gaps = time_embed(S_gaps)

        #get embedded user
        embedded_U = user_embed(U)
        embedded_U = embedded_U.unsqueeze(1)
        embedded_U = embedded_U.expand(embedded_U.size(0), embedded_S_gaps.size(1), embedded_U.size(2))

    #get the index of the last session representation of each user by subtracting 1 from each lengths, move to GPU for efficiency
    rep_indicies = Variable(torch.cuda.LongTensor(session_rep_lengths)) - 1

    #get initial hidden state of inter gru layer and call forward on the module
    inter_hidden = inter_rnn.init_hidden(S.size(0))
    if(flags["context"]):
        inter_last_hidden = inter_rnn(torch.cat((S, embedded_S_gaps, embedded_U),2), inter_hidden, rep_indicies)
    else:
        inter_last_hidden = inter_rnn(S, inter_hidden, rep_indicies)

    #get initial hidden for inter RNN, time scores and first prediction scores from the last hidden state of the inter RNN
    hidden = intra_linear(inter_last_hidden)
    if(flags["temporal"]):
        times = time_linear(inter_last_hidden).squeeze()
        first_predictions = first_linear(inter_last_hidden).squeeze()

        #calculate time error if this is desired
        if(time_error):
            w = time_loss_func.get_w()
            time_predictions = time_prediction(times.data, w.data, integration_count, integration_acc, flags)
            tester.evaluate_batch_time(time_predictions, time_targets)

    #get item embeddings
    embedded_X = item_embed(X)

    #create average pooling session representation using the item embeddings and the lenght of each sequence
    lengths = Variable(torch.cuda.FloatTensor(session_lengths).view(-1,1)) #reshape the lengths in order to broadcast and use it for division
    sum_X = embedded_X.sum(1)
    mean_X = sum_X.div(lengths)

    #subtract 1 from the lengths to get the index of the last item in each sequence
    lengths = lengths.long()-1

    #call forward on the inter RNN
    recommendation_output, hidden_out = intra_rnn(embedded_X, hidden, lengths)

    #store the new session representation based on the current scheme
    if(flags["use_hidden"]):
        datahandler.store_user_session_representations(hidden_out.data[0], user_list, time_targets)
    else:
        datahandler.store_user_session_representations(mean_X.data, user_list, time_targets)
    
    if(flags["temporal"]):
        k_values, k_predictions = torch.topk(torch.cat((first_predictions.unsqueeze(1),recommendation_output),1), TOP_K)
    else:
        k_values, k_predictions = torch.topk(recommendation_output, TOP_K)
    return k_predictions

#setting up for training
epoch_nr = 0
start_time = time.time()
num_training_batches = datahandler.get_num_training_batches()
num_test_batches = datahandler.get_num_test_batches()
epoch_loss = 0

#initially no part of the network is frozen
flags["train_time"] = True
flags["train_first"] = True
flags["train_all"] = True

#scedule updater
def update_settings(epoch_nr, flags, params):
    if(not flags["temporal"]):
        return
    else:
        if(epoch_nr == 4):
            params["ALPHA"] = 0.0
            params["BETA"] = 0.0
        if(epoch_nr == 8):
            params["BETA"] = 1.0
        if(epoch_nr == 10):
            params["ALPHA"] = 0.5
        if(epoch_nr == 11):
            params["BETA"] = 0.0
        if(epoch_nr == 12):
            params["ALPHA"] = 0.5
            params["BETA"] = 0.3
        if(flags["freeze"]):
            if(epoch_nr == 21):
                flags["train_all"] = False
                flags["train_first"] = False
                params["ALPHA"] = 1.0
            if(epoch_nr == 24):
                flags["train_first"] = True
                flags["train_time"] = False
                params["ALPHA"] = 0.0
                params["BETA"] = 1.0

#**********************************************************************TRAINING LOOP********************************************************************************************
while epoch_nr < MAX_EPOCHS:
    #print start of new epoch and save start time   
    print("Starting epoch #" + str(epoch_nr))
    start_time_epoch = time.time()

    update_settings(epoch_nr, flags, params)
    #reset state of datahandler and get first training batch
    datahandler.reset_user_batch_data_train()
    datahandler.reset_user_session_representations()
    items, item_targets, session_lengths, session_reps, session_rep_lengths, user_list, sess_time_reps, time_targets, first_rec_targets = datahandler.get_next_train_batch()
    batch_nr = 0

    #set model to train-mode, effectivly turning on all dropouts
    intra_rnn.train()
    inter_rnn.train()

    #loop until new batches are too sparse because most users have exhausted all their training sessions
    while(len(items) > int(BATCHSIZE/2)):
        #save start time of training epoch
        batch_start_time = time.time()

        #training call
        batch_loss = train_on_batch(items, session_reps, sess_time_reps, user_list, item_targets, time_targets, first_rec_targets, session_lengths, session_rep_lengths, flags, params, time_threshold)
        epoch_loss += batch_loss

        #total time spent on mini-batch
        batch_runtime = time.time() - batch_start_time

        #get next training batch
        items, item_targets, session_lengths, session_reps, session_rep_lengths, user_list, sess_time_reps, time_targets, first_rec_targets = datahandler.get_next_train_batch()

        #print batch loss and ETA occationally
        if batch_nr%1500 == 0:
            print("Batch: " + str(batch_nr) + "/" + str(num_training_batches) + " batch_loss: " + str(batch_loss))
            eta = (batch_runtime*(num_training_batches-batch_nr))/60
            eta = "%.2f" % eta
            print(" | ETA:", eta, "minutes.")

        #increment bath number
        batch_nr += 1
    #print mean recommendation loss in epoch
    print("Epoch loss: " + str(epoch_loss/batch_nr))


#********************************************************************************TESTING******************************************************************************************
    
    #test the model in some epochs, no need for testing in every epoch for most experiments
    if(epoch_nr > 0 and (epoch_nr%6 == 0 or epoch_nr == MAX_EPOCHS-1)):
        print("Starting testing")

        #initialize trainer
        tester = Tester(seqlen = SEQLEN, use_day = flags["use_day"], min_time = min_time, model_info = log_name, temporal = flags["temporal"])

        #reset state of datahandler and get first test batch
        datahandler.reset_user_batch_data_test()
        items, item_targets, session_lengths, session_reps, session_rep_lengths, user_list, sess_time_reps, time_targets, first_rec_targets = datahandler.get_next_test_batch()

        #set flag in order to only perform the expensive time prediction if necessary
        if( flags["temporal"] and epoch_nr == MAX_EPOCHS-1):
            time_error = True
        else:
            time_error = False

        #set model in evaluation mode, effectivly turing of dropouts and scaling affected weights accordingly
        intra_rnn.eval()
        inter_rnn.eval()

        batch_nr = 0
        while(len(items) > int(BATCHSIZE/2)):
            #batch testing
            batch_nr += 1
            batch_start_time = time.time()

            #run predictions on test batch
            k_predictions = predict_on_batch(items, session_reps, sess_time_reps, user_list, item_targets, time_targets, first_rec_targets, session_lengths, session_rep_lengths, flags, integration_count, integration_acc)

            #evaluate results
            if(flags["temporal"]):
                tester.evaluate_batch_temporal(k_predictions[:,1:], item_targets, session_lengths, k_predictions[:,0], first_rec_targets)
            else:
                tester.evaluate_batch_rec(k_predictions, item_targets, session_lengths)

            #get next test batch
            items, item_targets, session_lengths, session_reps, session_rep_lengths, user_list, sess_time_reps, time_targets, first_rec_targets = datahandler.get_next_test_batch()
            batch_runtime = time.time() - batch_start_time

            #print progress and ETA occationally
            if batch_nr%600 == 0:
                print("Batch: " + str(batch_nr) + "/" + str(num_test_batches))
                eta = (batch_runtime*(num_test_batches-batch_nr))/60
                eta = "%.2f" % eta
                print(" | ETA:", eta, "minutes.")
            
        # Print final test stats for epoch
        test_stats, current_recall5, current_recall20, time_stats, time_output, individual_scores = tester.get_stats_and_reset(get_time = time_error)
        print("Recall@5 = " + str(current_recall5))
        print("Recall@20 = " + str(current_recall20))
        print(test_stats)
        print("\n")
        print(individual_scores)

        #only print time stats if available
        if(time_error):
            print("\n")
            print(time_stats)
            print("Integration accuracy:" + str(integration_acc[0]/max(integration_count[0],1)))
            integration_acc = torch.cuda.FloatTensor([0])
            integration_count = torch.cuda.FloatTensor([0])

    #end of epoch, print total time, increment counter and reset epoch loss
    print("Epoch #" + str(epoch_nr) + " Time: " + str(time.time()-start_time_epoch))
    epoch_nr += 1
    epoch_loss = 0
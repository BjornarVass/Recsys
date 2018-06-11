import glob
import unicodedata
import string
import torch
import random
import time
import math
import os

from datahandler_temporal import RNNDataHandler
from tester_dynamic import Tester
from model import RecommenderModel

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
#from logger import Logger

#datasets
reddit = "subreddit"
lastfm = "lastfm"
reddit_time = "subreddit_time"
lastfm_time = "lastfm_time"
lastfm_simple = "lastfm_sim"
lastfm3 = "lastfm3"
instacart = "instacart"
nowplaying = "nowplaying"

#runtime settings
flags = {}
dataset = lastfm
flags["context"] = True
flags["temporal"] = True
SEED = 2
GPU = 0
gap_strat = ""
add = "_" if gap_strat != "" else ""
directory = "/data/stud/bjorva/logs/dim/" + gap_strat + add
debug = False

torch.manual_seed(SEED)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU)

#universal "static" settings
dims = {}
params = {}
BATCHSIZE = 100
N_LAYERS = 1 #currently not used
SEQLEN = 20-1
params["TOP_K"] = 20
MAX_SESSION_REPRESENTATIONS = 15
dims["TIME_RESOLUTION"] = 500
dims["TIME_HIDDEN"] = 5
dims["USER_HIDDEN"] = 10
flags["train_time"] = True
flags["train_first"] = True
flags["train_all"] = True
flags["use_hidden"] = True


params["ALPHA"] = 0.45
params["BETA"] = 0.45
params["GAMMA"] = 0.1
params["EPSILON"] = 1.0
flags["use_day"] = True

#data path and log/model-name
dataset_path = "/data/stud/bjorva/datasets/" + dataset + "/4_train_test_split.pickle"
if(flags["temporal"]):
    model = "temporal"
elif(flags["context"]):
    model = "context"
else:
    model = "inter"
log_name = directory + model + "_" + dataset + str(SEED)
txt_file_name = log_name+".txt"
with open(txt_file_name,'w+') as txt_file:
    txt_file.write("New experiment\n")

#dataset dependent settings
if dataset == reddit or dataset == reddit_time:
    dims["EMBEDDING_DIM"] = 50
    params["lr"] = 0.001
    params["dropout"] = 0.2 if flags["context"] else 0.0
    MAX_EPOCHS = 30
    if model == "inter":
        MAX_EPOCHS -= 10
    min_time = 1.0
    flags["freeze"] = False
elif dataset == lastfm or dataset == lastfm_simple or dataset == lastfm_time or dataset == nowplaying:
    dims["EMBEDDING_DIM"] = 100
    params["lr"] = 0.001
    params["dropout"] = 0.2
    MAX_EPOCHS = 24
    if model == "inter":
        MAX_EPOCHS -= 3
    min_time = 0.5
    flags["freeze"] = False
elif dataset == instacart:
    dims["EMBEDDING_DIM"] = 80
    params["lr"] = 0.001
    params["dropout"] = 0.2
    MAX_EPOCHS = 30
    min_time = 0.5
    flags["freeze"] = False
#additional parameter
time_threshold = torch.cuda.FloatTensor([min_time])
if(flags["use_day"]):
    time_threshold /= 24

#dimensionalities
dims["INTRA_HIDDEN"] = dims["EMBEDDING_DIM"]
if(flags["context"]):
    dims["INTER_INPUT_DIM"] = dims["INTRA_HIDDEN"] + dims["TIME_HIDDEN"] + dims["USER_HIDDEN"]
else:
    dims["INTER_INPUT_DIM"] = dims["INTRA_HIDDEN"]

dims["INTER_HIDDEN"] = dims["INTRA_HIDDEN"]

#loading of dataset into datahandler and getting relevant iformation about the dataset
datahandler = RNNDataHandler(dataset_path, BATCHSIZE, MAX_SESSION_REPRESENTATIONS, dims["INTRA_HIDDEN"], dims["TIME_RESOLUTION"], flags["use_day"], min_time, gap_strat)
dims["N_ITEMS"] = datahandler.get_num_items()
N_SESSIONS = datahandler.get_num_training_sessions()
dims["N_USERS"] = datahandler.get_num_users()

#initialize tester
tester = Tester(seqlen = SEQLEN, use_day = flags["use_day"], min_time = min_time, model_info = log_name, temporal = flags["temporal"])

#initialize model
model = RecommenderModel(dims, params, flags, datahandler, tester, time_threshold)

#setting up for training
epoch_nr = 0
start_time = time.time()
num_training_batches = datahandler.get_num_training_batches()
num_test_batches = datahandler.get_num_test_batches()
epoch_loss = 0

#**********************************************************************TRAINING LOOP********************************************************************************************
while epoch_nr < MAX_EPOCHS:
    #print start of new epoch and save start time
    if(debug):
        print("Starting epoch #" + str(epoch_nr))
    with open(txt_file_name,'a') as txt_file:
        txt_file.write("Starting epoch #" + str(epoch_nr)+"\n")
    start_time_epoch = time.time()
    """
    if(flags["temporal"]):
        model.update_loss_settings(epoch_nr)
    """

    #reset state of datahandler and get first training batch
    datahandler.reset_user_batch_data_train()
    datahandler.reset_user_session_representations()
    items, item_targets, session_lengths, session_reps, session_rep_lengths, user_list, sess_time_reps, time_targets, first_rec_targets = datahandler.get_next_train_batch()
    batch_nr = 0

    #set model to train-mode, effectivly turning on all dropouts
    model.train_mode()

    #loop until new batches are too sparse because most users have exhausted all their training sessions
    while(len(items) > int(BATCHSIZE/2)):
        #save start time of training epoch
        batch_start_time = time.time()

        #training call
        batch_loss = model.train_on_batch(items, session_reps, sess_time_reps, user_list, item_targets, time_targets, first_rec_targets, session_lengths, session_rep_lengths)
        epoch_loss += batch_loss

        #total time spent on mini-batch
        batch_runtime = time.time() - batch_start_time

        #get next training batch
        items, item_targets, session_lengths, session_reps, session_rep_lengths, user_list, sess_time_reps, time_targets, first_rec_targets = datahandler.get_next_train_batch()

        #print batch loss and ETA occationally
        if batch_nr%1500 == 0:
            eta = (batch_runtime*(num_training_batches-batch_nr))/60
            eta = "%.2f" % eta
            if(debug):
                print("Batch: " + str(batch_nr) + "/" + str(num_training_batches) + " batch_loss: " + str(batch_loss))
                print(" | ETA:", eta, "minutes.")
            with open(txt_file_name,'a') as txt_file:
                txt_file.write("Batch: " + str(batch_nr) + "/" + str(num_training_batches) + " batch_loss: " + str(batch_loss)+"\n")
                txt_file.write(" | ETA:" + str(eta) + "minutes."+"\n")

        batch_nr += 1
    #print mean recommendation loss in epoch
    if(debug):
        print("Epoch loss: " + str(epoch_loss/batch_nr))
    with open(txt_file_name,'a') as txt_file:
        txt_file.write("Epoch loss: " + str(epoch_loss/batch_nr)+"\n")

#********************************************************************************TESTING******************************************************************************************
    
    #test the model in some epochs, no need for testing in every epoch for most experiments
    if(epoch_nr == MAX_EPOCHS-1):
        if(debug):
            print("Starting testing")
        with open(txt_file_name,'a') as txt_file:
            txt_file.write("Starting testing"+"\n")
        #reset state of datahandler and get first test batch
        datahandler.reset_user_batch_data_test()
        items, item_targets, session_lengths, session_reps, session_rep_lengths, user_list, sess_time_reps, time_targets, first_rec_targets = datahandler.get_next_test_batch()

        #set flag in order to only perform the expensive time prediction if necessary
        if( flags["temporal"] and epoch_nr == MAX_EPOCHS-1):
            time_error = True
        else:
            time_error = False

        #set model in evaluation mode, effectivly turing off dropouts and scaling affected weights accordingly
        model.eval_mode()

        batch_nr = 0
        while(len(items) > int(BATCHSIZE/2)):
            #batch testing
            batch_start_time = time.time()

            #run predictions on test batch
            k_predictions = model.predict_on_batch(items, session_reps, sess_time_reps, user_list, item_targets, time_targets, first_rec_targets, session_lengths, session_rep_lengths, time_error)

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
                eta = (batch_runtime*(num_test_batches-batch_nr))/60
                eta = "%.2f" % eta
                if(debug):
                    print("Batch: " + str(batch_nr) + "/" + str(num_test_batches))
                    print(" | ETA:", eta, "minutes.")
                with open(txt_file_name,'a') as txt_file:
                    txt_file.write("Batch: " + str(batch_nr) + "/" + str(num_test_batches)+"\n")
                    txt_file.write(" | ETA:" + str(eta) + "minutes."+"\n")

            batch_nr += 1
            
        # Print final test stats for epoch
        test_stats, time_stats, individual_stats = tester.get_stats_and_reset(get_time = time_error, store = epoch_nr == MAX_EPOCHS-1)
        if(debug):
            print(test_stats)
            print("\n")
            print(individual_stats)
        with open(txt_file_name,'a') as txt_file:
            txt_file.write(test_stats+"\n\n")
            txt_file.write(individual_stats + "\n\n")

        #only print time stats if available
        if(time_error):
            if(debug):
                print("\n")
                print(time_stats)
            with open(txt_file_name,'a') as txt_file:
                txt_file.write(str(model.get_w().data))
                txt_file.write("\n")
                txt_file.write(time_stats + "\n\n")

    #end of epoch, print total time, increment counter and reset epoch loss
    if(debug):
        print("Epoch #" + str(epoch_nr) + " Time: " + str(time.time()-start_time_epoch))
    with open(txt_file_name,'a') as txt_file:
        txt_file.write("Epoch #" + str(epoch_nr) + " Time: " + str(time.time()-start_time_epoch)+"\n")
    epoch_nr += 1
    epoch_loss = 0

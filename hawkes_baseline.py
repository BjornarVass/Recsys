from hawkes import MHP
from hawkes_datahandler import DataHandler
import numpy as np
import pickle

#datasets
reddit = "subreddit"
reddit_std = "subreddit_std_eight"
lastfm = "lastfm"
lastfm_simple = "lastfm_sim"
instacart = "instacart"

#global settings
USE_DAY = True
dataset = lastfm
n_decimals = 4

#parameters
if(dataset == lastfm or dataset == lastfm_simple or dataset == instacart):
    min_time = 0.5
elif(dataset == reddit or dataset == reddit_std):
    min_time = 1.0

#switchable
full_hist = False
gap_strat = ""

add = "_" if gap_strat != "" else ""
pickle_path = "hawkes_regular_" + dataset + add + gap_strat + "4.pickle"
omega = 8
history_length = 15
future_length = 1
sample_size = 100
time_buckets = [2, 12, 36, 60, 84, 108, 132, 156, 180, 204, 228, 252, 276, 300, 348, 396, 444, 500, 501]
if(USE_DAY):
    for i in range(len(time_buckets)):
        time_buckets[i] /=24

if(gap_strat == ""):
    #loading of data
    dataset_path = "/data/stud/bjorva/datasets/" + dataset + "/4_train_test_split.pickle"
    datahandler = DataHandler(dataset_path, USE_DAY, min_time)

    data = datahandler.get_times()
    user_gaps = datahandler.get_gaps()
else:
    times_path = "/data/stud/bjorva/datasets/" + dataset + "/gaps_" + gap_strat + ".pickle"
    raw_data = pickle.load(open(times_path,"rb"))
    data = {}
    for user in raw_data["train"].keys():
        if (len(raw_data["train"][user])>0):
            times = [raw_data["train"][user][0]]
            start_train = 1
            start_test = 0
        else:
            times = [raw_data["test"][user][0]]
            start_train = 0
            start_test = 1
        for i in range(start_train,len(raw_data["train"][user])):
            if raw_data["train"][user][i] != 0:
                times.append(raw_data["train"][user][i]+times[-1])
        for i in range(start_test,len(raw_data["test"][user])):
            if raw_data["test"][user][i] != 0:
                times.append(raw_data["test"][user][i]+times[-1])
        data[user] = times

#setting up data structures for keeping track of the results
mae = np.zeros((future_length,len(time_buckets)))
percentage_errors = np.zeros((future_length,len(time_buckets)))
no_predictions = np.zeros((future_length,len(time_buckets)))

#user loop
for user in range(len(data)):
    #setting of user specific data structures
    history = history_length
    future = future_length
    avg_gap = (data[user][-1]-data[user][0])/len(data[user])
    split_index = int(len(data[user])*0.8)

    #handling of users with short sequences of sessions
    if(split_index < history_length):
        history = split_index
    if(len(data[user])-split_index < future_length):
        future = len(data[user])-split_index
    if(len(data[user])< 3):
        continue

    if full_hist:
        seq = []
        for i in range(split_index):
            seq.append([data[user][i]-data[user][0],0])
        seq = np.array(seq)
        P = MHP()
        mhat = np.random.uniform(0,1, size=1)
        ahat = np.random.uniform(0,1, size=(1,1))
        alpha, mu = P.EM(ahat, mhat, omega, seq, verbose=False)

        P = MHP(alpha, mu, omega)
    #testing loop, fits a hawkes point process on a pre-set number of observations
    #then "predicts" by sampling said point process and stores the scores
    i = split_index-history
    while(i < len(data[user])-history-future+1):
        #create a sequence in the form [normalized time, user/dim], we use only one user/dim
        seq = []
        for j in range(i, i+history):
            seq.append([data[user][j]-data[user][i],0])
        seq = np.array(seq)
        if(not full_hist):
            #init a hawkes object
            P = MHP()

            #get fitted parameters based on inputted sequence and randomized initial values
            mhat = np.random.uniform(0,1, size=1)
            ahat = np.random.uniform(0,1, size=(1,1))
            alpha, mu = P.EM(ahat, mhat, omega, seq, verbose=False)

            #set the parameters of the point process to those found
            P = MHP(alpha, mu, omega)

        #sample loop
        results = np.zeros(future_length)
        for sample in range(sample_size):
            #sample by simulation using the sequence used for fitting to get initial rates
            result = P.generate_seq(future_length, init_rates = P.get_init_rates(seq))
            results += result[1:,0]
        results/=sample_size
        start = i+history

        #loop for splitting the final values into correct buckets, and position in future events, based on gap size
        for j in range(future_length):
            gap = data[user][start+j]-data[user][start+j-1]
            for k in range(len(time_buckets)):
                if gap < time_buckets[k]:
                    diff = abs(results[j]-gap)
                    mae[j][k] += diff
                    percentage_errors[j][k] += (diff/gap)*100
                    no_predictions[j][k] += 1
                    break

        #either the history or the start index is incremented
        if(history < history_length):
            history += 1
        else:
            i += 1


#preparing output messages
time_messages = []
for i in range(future_length):
    time_message = "\t\tMAE\tPercent\t"
    cumulative_count = 0
    cumulative_error = 0
    cumulative_percent = 0
    prefix = ""
    if(USE_DAY):
        prefix = "\ndays<="
    else:
        prefix = "\nhours<="
    for j in range(len(time_buckets)):
        time_message += prefix+str(round(time_buckets[j],1))+"\t"
        error = mae[i][j]/max(no_predictions[i][j],1)
        percent = percentage_errors[i][j]/max(no_predictions[i][j],1)
        if(j != len(time_buckets)-1):
            cumulative_count += no_predictions[i][j]
            cumulative_percent += percentage_errors[i][j]
            cumulative_error += mae[i][j]
        time_message += str(round(error, n_decimals))+'\t'
        time_message += str(round(percent, n_decimals))+'\t'
    cumulative_count = max(cumulative_count,1)
    #time_output = cumulative_error/cumulative_count
    time_message += "\ntotal-last\t" + str(round(cumulative_error/cumulative_count, n_decimals))+'\t' + str(round(cumulative_percent/cumulative_count, n_decimals))+'\t'
    last = len(time_buckets)-1
    cumulative_count += no_predictions[i][last]
    cumulative_error += mae[i][last]
    cumulative_percent += percentage_errors[i][last]
    time_message += "\ntotal\t" + str(round(cumulative_error/cumulative_count, n_decimals))+'\t' + str(round(cumulative_percent/cumulative_count, n_decimals))+'\t'
    time_messages.append(time_message)
"""
for msg in time_messages:
    print(msg)
    print("\n")
"""
pickle_dict = {}
pickle_dict["mae"] = mae
pickle_dict["count"] = no_predictions
pickle_dict["buckets"] = time_buckets
pickle_dict["percent"] = percentage_errors

pickle.dump(pickle_dict, open(pickle_path, 'wb'))
import collections
import datetime
import logging
import math
import numpy as np
import os
import pickle
import time
from datetime import datetime

class RNNDataHandler:
    
    def __init__(self, dataset_path, batch_size, max_sess_reps, lt_internalsize, time_resolution, use_day, min_time, gap_strat):
        # LOAD DATASET
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        #print("Loading dataset")
        load_time = time.time()
        dataset = pickle.load(open(self.dataset_path, 'rb'))
        #print("|- dataset loaded in", str(time.time()-load_time), "s")

        self.trainset = dataset['trainset']
        self.testset = dataset['testset']
        self.train_session_lengths = dataset['train_session_lengths']
        self.test_session_lengths = dataset['test_session_lengths']
        
        self.num_users = len(self.trainset)
        if len(self.trainset) != len(self.testset):
            raise Exception("""Testset and trainset have different 
                    amount of users.""")

        # II_RNN stuff
        self.MAX_SESSION_REPRESENTATIONS = max_sess_reps
        self.LT_INTERNALSIZE = lt_internalsize
    
        # batch control
        self.gap_strat = gap_strat
        self.time_resolution = time_resolution
        self.use_day = use_day
        self.time_factor = 24 if self.use_day else 1
        self.min_time = min_time/self.time_factor
        self.divident = 3600*self.time_factor
        self.init_user_times()
        self.reset_user_batch_data_train()


    def init_user_times(self):
        self.user_train_times = [None]*self.num_users
        self.user_test_times = [None]*self.num_users
        self.max_time = 500/self.time_factor
        self.max_exp = 50
        self.scale = 1#np.log(self.max_exp+1)
        self.delta = self.scale/self.time_resolution
        self.scale += 0.01 #overflow handling

        #add gap-times based on first timestamp in new session and the last timestamp in the last session
        #gaps that are less than provided minimum threshold indicate that the two sessions involved originally was a single session, thus the gap should be ignored
        
        if(self.gap_strat == ""):
            for k in self.trainset.keys():
                times = []
                #add initial gap of zero if the user has a session in the trainset
                if(len(self.trainset[k]) > 0):
                    times.append(0)
                #add gaps within trainset
                for session_index in range(1,len(self.trainset[k])):
                    gap = self.real_gap(self.trainset[k][session_index][0][0],self.trainset[k][session_index-1][self.train_session_lengths[k][session_index-1]][0])
                    times.append(gap)
                self.user_train_times[k] = times
                
                times = []
                #add gap between the last session in train and the first session in test, if the user has sessions in both sets, add gap of 0 if user only has sessions in testset
                if(len(self.trainset[k]) > 0 and len(self.testset[k]) > 0):
                    gap = self.real_gap(self.testset[k][0][0][0],self.trainset[k][-1][self.train_session_lengths[k][-1]][0])
                    times.append(gap)
                elif(len(self.testset[k]) > 0):
                    times.append(0)

                #add gaps within testset
                for session_index in range(1,len(self.testset[k])):
                    gap = self.real_gap(self.testset[k][session_index][0][0],self.testset[k][session_index-1][self.test_session_lengths[k][session_index-1]][0])
                    times.append(gap)
                self.user_test_times[k] = times
        else:
            end_index = self.dataset_path.index("4")
            pickle_path = self.dataset_path[:end_index] + "gaps_" + self.gap_strat + ".pickle"
            times = pickle.load(open(pickle_path ,"rb"))
            self.user_train_times = times["train"]
            self.user_test_times = times["test"]


    def real_gap(self, new_time, old_time):
        gap = (new_time-old_time)/self.divident
        gap = gap if gap < self.max_time else self.max_time
        return gap if gap > self.min_time else 0

    # call before training and testing
    def reset_user_batch_data(self, dataset):
        # the index of the next session(event) to retrieve for a user
        self.user_next_session_to_retrieve = [0]*self.num_users
        # list of users who have not been exhausted for sessions
        self.users_with_remaining_sessions = []
        # a list where we store the number of remaining sessions for each user. Updated for eatch batch fetch. But we don't want to create the object multiple times.
        self.num_remaining_sessions_for_user = [0]*self.num_users
        for k, v in self.trainset.items():
            # user may have 0 sessions in the relevant dataset
            if(len(dataset[k]) > 0):
                self.users_with_remaining_sessions.append(k)

    def reset_user_batch_data_train(self):
        self.reset_user_batch_data(self.trainset)

    def reset_user_batch_data_test(self):
        self.reset_user_batch_data(self.testset)

    def reset_user_session_representations(self):
        #istate = np.zeros([self.LT_INTERNALSIZE])

        # session representations for each user is stored here
        self.user_session_representations = [None]*self.num_users
        self.user_gaptime_representations = [None]*self.num_users
        # the number of (real) session representations a user has
        self.num_user_session_representations = [0]*self.num_users
        for k, v in self.trainset.items():
            self.user_session_representations[k] = collections.deque(maxlen=self.MAX_SESSION_REPRESENTATIONS)
            self.user_session_representations[k].append([0]*self.LT_INTERNALSIZE)
            self.user_gaptime_representations[k] = collections.deque(maxlen=self.MAX_SESSION_REPRESENTATIONS)
            self.user_gaptime_representations[k].append(0)

    def get_N_highest_indexes(a,N):
        return np.argsort(a)[::-1][:N]


    def add_unique_items_to_dict(self, items, dataset):
        for k, v in dataset.items():
            for session in v:
                for event in session:
                    item = event[1]
                    if item not in items:
                        items[item] = True
        return items

    def get_num_users(self):
        return self.num_users

    def get_num_items(self):
        items = {}
        items = self.add_unique_items_to_dict(items, self.trainset)
        items = self.add_unique_items_to_dict(items, self.testset)
        return len(items)

    def get_num_sessions(self, dataset):
        session_count = 0
        for k, v in dataset.items():
            session_count += len(v)
        return session_count

    def get_num_training_sessions(self):
        return self.get_num_sessions(self.trainset)
    
    # for the II-RNN this is only an estimate
    def get_num_batches(self, dataset):
        num_sessions = self.get_num_sessions(dataset)
        return math.ceil(num_sessions/self.batch_size)

    def get_num_training_batches(self):
        return self.get_num_batches(self.trainset)

    def get_num_test_batches(self):
        return self.get_num_batches(self.testset)

    def get_next_batch(self, dataset, dataset_session_lengths, time_set):
        session_batch = []
        session_lengths = []
        sess_rep_batch = []
        sess_gaptime_batch = []
        sess_rep_lengths = []
        target_times = []
        
        # Decide which users to take sessions from. First count the number of remaining sessions
        remaining_sessions = [0]*len(self.users_with_remaining_sessions)
        for i in range(len(self.users_with_remaining_sessions)):
            user = self.users_with_remaining_sessions[i]
            remaining_sessions[i] = len(dataset[user]) - self.user_next_session_to_retrieve[user]
        
        # index of users to get
        user_list = RNNDataHandler.get_N_highest_indexes(remaining_sessions, self.batch_size)
        if(len(user_list) == 0):
            return [],[],[],[],[],[],[],[],[]
        for i in range(len(user_list)):
            user_list[i] = self.users_with_remaining_sessions[user_list[i]]

        # For each user -> get the next session, and check if we should remove 
        # him from the list of users with remaining sessions
        for user in user_list:
            session_index = self.user_next_session_to_retrieve[user]
            session_batch.append(dataset[user][session_index])
            session_lengths.append(dataset_session_lengths[user][session_index])
            srl = max(self.num_user_session_representations[user],1)
            sess_rep_lengths.append(srl)
            sess_rep = list(self.user_session_representations[user]) #copy
            sess_gaptime = list(self.user_gaptime_representations[user])

            #pad session representations and corresponding contexts if not full
            if(srl < self.MAX_SESSION_REPRESENTATIONS):
                for i in range(self.MAX_SESSION_REPRESENTATIONS-srl):
                    sess_rep.append([0]*self.LT_INTERNALSIZE) #pad with zeroes after valid reps
                    sess_gaptime.append(0) #pad with zeros after valid time-gaps
            sess_rep_batch.append(sess_rep)
            sess_gaptime_batch.append(sess_gaptime)

            self.user_next_session_to_retrieve[user] += 1
            if self.user_next_session_to_retrieve[user] >= len(dataset[user]):
                # User have no more session, remove him from users_with_remaining_sessions
                self.users_with_remaining_sessions.remove(user)
            target_times.append(time_set[user][session_index]) 

        #sort batch based on seq rep len
        session_batch = [[event[1] for event in session] for session in session_batch]
        x = [session[:-1] for session in session_batch]
        y = [session[1:] for session in session_batch]
        first_predictions = [session[0] for session in session_batch]

        return x, y, session_lengths, sess_rep_batch, sess_rep_lengths, user_list, sess_gaptime_batch, target_times, first_predictions

    def get_next_train_batch(self):
        return self.get_next_batch(self.trainset, self.train_session_lengths, self.user_train_times)

    def get_next_test_batch(self):
        return self.get_next_batch(self.testset, self.test_session_lengths, self.user_test_times)

    
    def store_user_session_representations(self, sessions_representations, user_list, target_times):
        for i in range(len(user_list)):
            user = user_list[i]
            session_representation = list(sessions_representations[i])
            target_time = float(target_times[i])
            if(target_time > self.min_time):
                #if(target_time > self.max_time):
                #    target_time = 0
                target_time = min(target_time, self.max_time)/self.max_time
                #target_time = np.log2(min(target_time,self.max_time)/self.max_time*(self.max_exp)+1)
                target_time = target_time/self.scale
                target_time = int(target_time//self.delta)
            else:
                target_time = 0

            
            num_reps = self.num_user_session_representations[user]

            #self.num_user_session_representations[user] = min(self.MAX_SESSION_REPRESENTATIONS, num_reps+1)
            if(num_reps == 0):
                self.user_session_representations[user].pop() #pop dummy session representation
                self.user_gaptime_representations[user].pop() #pop dummy gap-time
            self.user_session_representations[user].append(session_representation)
            self.user_gaptime_representations[user].append(target_time)
            self.num_user_session_representations[user] = min(self.MAX_SESSION_REPRESENTATIONS, num_reps+1)


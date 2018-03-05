import collections
import datetime
import logging
import math
import numpy as np
import os
import pickle
import time
from datetime import datetime

class DataHandler:
    
    def __init__(self, dataset_path, use_day, min_time):
        # LOAD DATASET
        self.dataset_path = dataset_path
        print("Loading dataset")
        load_time = time.time()
        dataset = pickle.load(open(self.dataset_path, 'rb'))
        print("|- dataset loaded in", str(time.time()-load_time), "s")

        self.trainset = dataset['trainset']
        self.testset = dataset['testset']
        self.train_session_lengths = dataset['train_session_lengths']
        self.test_session_lengths = dataset['test_session_lengths']
        
        self.num_users = len(self.trainset)
        if len(self.trainset) != len(self.testset):
            raise Exception("""Testset and trainset have different 
                    amount of users.""")
    
        # batch control
        self.use_day = use_day
        self.time_factor = 24 if self.use_day else 1
        self.min_time = min_time/self.time_factor
        self.max_time = 500/self.time_factor
        self.divident = 3600*self.time_factor
        self.user_gap_indices = {}
        self.init_user_times()


    def init_user_times(self):
        self.user_times = [None]*self.num_users
        self.max_time = 500/self.time_factor
        for k in self.trainset.keys():
            train = self.trainset[k] 
            if(len(train) > 0):
                times = [train[0][0][0]/self.divident]
            else:
                times = []
            for session_index in range(1,len(train)):
                gap = (train[session_index][0][0]-train[session_index-1][self.train_session_lengths[k][session_index-1]][0])/self.divident
                if(gap > self.min_time):
                    times.append(train[session_index][0][0]/self.divident)
            test = self.testset[k]
            if(len(train) > 0 and len(test) > 0):
                gap = (test[0][0][0]-train[-1][self.train_session_lengths[k][-1]][0])/self.divident
                if(gap > self.min_time):
                    times.append(test[0][0][0]/self.divident)
            for session_index in range(1,len(test)):
                gap = (test[session_index][0][0]-test[session_index-1][self.test_session_lengths[k][session_index-1]][0])/self.divident
                if(gap > self.min_time):
                    times.append(test[session_index][0][0]/self.divident)
            self.user_times[k] = times
            self.remove_long_gaps(k)
        
    def remove_long_gaps(self, k):        
        long_gap_indices = []
        discrepencies = []
        for i in range(len(self.user_times[k])-1):
            if(self.user_times[k][i+1]-self.user_times[k][i] > self.max_time):
                long_gap_indices.append(i)
                discrepencies.append((self.user_times[k][i+1]-self.user_times[k][i])-self.max_time)
        remove = sum(discrepencies)
        index = len(long_gap_indices)-1
        if(index > 0):
            for i in range(len(self.user_times[k])-1,0,-1):
                if(i == long_gap_indices[index]):
                    remove -= discrepencies[index]
                    if(index != 0):
                        index -= 1
                self.user_times[k][i] = self.user_times[k][i]-remove
        gaps = np.array(long_gap_indices)
        gaps = gaps+1
        self.user_gap_indices[k] = gaps




    def get_times(self):
        return self.user_times

    def get_gaps(self):
        return self.user_gap_indices

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from modules import Embed, Intra_RNN, Inter_RNN, Time_Loss


class RecommenderModel:

    def __init__(self, dims, params, flags, datahandler, tester, time_threshold):
        self.dims = dims
        self.params = params
        self.flags = flags
        self.datahandler = datahandler
        self.tester = tester
        self.time_threshold = time_threshold
        self.init_model()

    def init_model(self):
        #initialize lists to contain the parameters in two sub-nets
        inter_intra_params = []
        time_params = []

        #setting up embedding matrices
        self.item_embed = Embed(self.dims["N_ITEMS"], self.dims["EMBEDDING_DIM"], item=True)
        self.item_embed = self.item_embed.cuda()
        inter_intra_params += list(self.item_embed.parameters())

        if(self.flags["context"]):
            self.time_embed = Embed(self.dims["TIME_RESOLUTION"], self.dims["TIME_HIDDEN"], item=False)
            self.time_embed = self.time_embed.cuda()
            inter_intra_params += list(self.time_embed.parameters())

            self.user_embed = Embed(self.dims["N_USERS"], self.dims["USER_HIDDEN"], item=False)
            self.user_embed = self.user_embed.cuda()
            inter_intra_params += list(self.user_embed.parameters())

        #setting up models with optimizers
        self.inter_rnn = Inter_RNN(self.dims["INTER_INPUT_DIM"], self.dims["INTER_HIDDEN"], self.params["dropout"])
        self.inter_rnn = self.inter_rnn.cuda()
        inter_intra_params += list(self.inter_rnn.parameters())

        self.intra_rnn = Intra_RNN(self.dims["EMBEDDING_DIM"], self.dims["INTRA_HIDDEN"], self.dims["N_ITEMS"], self.params["dropout"])
        self.intra_rnn = self.intra_rnn.cuda()
        inter_intra_params += list(self.intra_rnn.parameters())

        #setting up linear layers for the time loss, first recommendation loss and inter RNN
        if(self.flags["temporal"]):
            self.time_linear = nn.Linear(self.dims["INTER_HIDDEN"],1)
            self.time_linear = self.time_linear.cuda()
            time_params += [{"params": self.time_linear.parameters(), "lr":0.1*self.params["lr"]}]

            self.first_linear = nn.Linear(self.dims["INTER_HIDDEN"],self.dims["N_ITEMS"])
            self.first_linear = self.first_linear.cuda()
        """
        self.intra_linear = nn.Linear(self.dims["INTER_HIDDEN"],self.dims["INTRA_HIDDEN"])
        self.intra_linear = self.intra_linear.cuda()
        inter_intra_params += list(self.intra_linear.parameters())
        """
        #setting up time loss model
        if(self.flags["temporal"]):
            self.time_loss_func = Time_Loss()
            self.time_loss_func = self.time_loss_func.cuda()
            time_params += [{"params": self.time_loss_func.parameters(), "lr": 0.1*self.params["lr"]}]

        #setting up optimizers
        self.inter_intra_optimizer = torch.optim.Adam(inter_intra_params, lr=self.params["lr"])
        if(self.flags["temporal"]):
            self.time_optimizer = torch.optim.Adam(time_params, lr=self.params["lr"])
            self.first_rec_optimizer = torch.optim.Adam(self.first_linear.parameters(), lr=self.params["lr"])

    #CUSTOM CROSS ENTROPY LOSS(Replace as soon as pytorch has implemented an option for non-summed losses)
    #https://github.com/pytorch/pytorch/issues/264
    def masked_cross_entropy_loss(self, y_hat, y):
        logp = -F.log_softmax(y_hat, dim=1)
        logpy = torch.gather(logp,1,y.view(-1,1))
        mask = Variable(y.data.float().sign().view(-1,1))
        logpy = logpy*mask
        return logpy.view(-1)

    def get_w(self):
        return self.time_loss_func.get_w()

    #step function implementing the equantion:
    #exp(time+w*t + exp(time)-exp(time+w*t)/w) = exp_time_w*exp((exp_time-exp_time_w)/w)
    @staticmethod
    def step_val(t, time_exp, w, dt): 
        time_w_exp = time_exp*torch.exp(t*w)
        exp_2 = torch.exp((time_exp-time_w_exp)/w)
        prob = time_w_exp*exp_2
        return t*prob

    #simpson numerical integration with higher resolution in the first 100 hours
    def time_prediction(self, time, w):
        #integration settings
        #integration_count += 1
        precision = 3000
        T = 700 #time units
        part1 = 100
        part2 = 600
        if(self.flags["use_day"]):
            T = T/24
            part1 = part1/24
            part2 = part2/24

        #moving data structures to the  for efficiency
        T = torch.cuda.FloatTensor([T])
        dt1 = torch.cuda.FloatTensor([part1/precision])
        dt2 = torch.cuda.FloatTensor([part2/precision])
        part1 = torch.cuda.FloatTensor([part1])
        
        #integration loops
        time_exp = torch.exp(time)
        time_preds1 = self.step_val(part1,time_exp, w, dt1)
        time_preds2 = self.step_val(T,time_exp, w, dt2) + time_preds1
        for i in range(1,precision//2):#high resolution loop
            t = (2*i-1)*dt1
            time_preds1 += 4*self.step_val(t,time_exp, w, dt1)
            time_preds1 += 2*self.step_val(t+dt1,time_exp, w, dt1)
        time_preds1 += 4*self.step_val(part1-dt1,time_exp, w, dt1)
        for i in range(1,precision//2):#rough resolution loop
            t = (2*i-1)*dt2 + part1
            time_preds2 += 4*self.step_val(t,time_exp, w, dt2)
            time_preds2 += 2*self.step_val(t+dt2,time_exp, w, dt2)
        time_preds2 += 4*self.step_val(T-dt2,time_exp,w,dt2)

        #division moved to the end for efficiency
        time_preds1 *= dt1/3
        time_preds2 *= dt2/3

        return time_preds1+time_preds2

        #scedule updater
    def update_loss_settings(self, epoch_nr):
        if(not self.flags["temporal"]):
            return
        else:
            if(epoch_nr == 0):
                self.params["ALPHA"] = 1.0
                self.params["BETA"] = 0.0
                self.params["GAMMA"] = 0.0
            if(epoch_nr == 4):
                self.params["ALPHA"] = 0.0
                self.params["BETA"] = 1.0
            if(epoch_nr == 8):
                self.params["BETA"] = 0.0
                self.params["GAMMA"] = 1.0
            if(epoch_nr == 10):
                self.params["ALPHA"] = 0.5
                self.params["GAMMA"] = 0.5
            if(epoch_nr == 11):
                self.params["BETA"] = 0.5
                self.params["GAMMA"] = 0.0
            if(epoch_nr == 12):
                self.params["ALPHA"] = 0.45
                self.params["BETA"] = 0.45
                self.params["GAMMA"] = 0.1
            if(self.flags["freeze"]):
                if(epoch_nr == 21):
                    self.flags["train_all"] = False
                    self.flags["train_first"] = False
                    self.params["ALPHA"] = 1.0
                    self.params["BETA"] = 0.0
                    self.params["GAMMA"] = 0.0
                if(epoch_nr == 24):
                    self.flags["train_first"] = True
                    self.flags["train_time"] = False
                    self.params["ALPHA"] = 0.0
                    self.params["GAMMA"] = 1.0
        return

    def train_mode(self):
        self.intra_rnn.train()
        self.inter_rnn.train()
        return

    def eval_mode(self):
        self.intra_rnn.eval()
        self.inter_rnn.eval()
        return

    #move batch data to cuda tensors
    def process_batch_inputs(self, items, session_reps, sess_time_reps, user_list):
        sessions = Variable(torch.cuda.FloatTensor(session_reps))
        items = Variable(torch.cuda.LongTensor(items))
        sess_gaps = Variable(torch.cuda.LongTensor(sess_time_reps))
        users = Variable(torch.cuda.LongTensor(user_list.tolist()))
        return items, sessions, sess_gaps, users

    def process_batch_targets(self, item_targets, time_targets, first_rec_targets):
        item_targets = Variable(torch.cuda.LongTensor(item_targets))
        time_targets = Variable(torch.cuda.FloatTensor(time_targets)) 
        first = Variable(torch.cuda.LongTensor(first_rec_targets))
        return item_targets, time_targets, first

    def train_on_batch(self, items, session_reps, sess_time_reps, user_list, item_targets, time_targets, first_rec_targets, session_lengths, session_rep_lengths):
        #zero gradients before each epoch
        self.inter_intra_optimizer.zero_grad()
        if(self.flags["temporal"]):
            self.time_optimizer.zero_grad()
            self.first_rec_optimizer.zero_grad()

        #get batch from datahandler and turn into variables
        X, S, S_gaps, U = self.process_batch_inputs(items, session_reps, sess_time_reps, user_list)
        Y, T_targets, First_targets = self.process_batch_targets(item_targets, time_targets, first_rec_targets)

        if(self.flags["context"]):
            #get embedded times
            embedded_S_gaps = self.time_embed(S_gaps)

            #get embedded user
            embedded_U = self.user_embed(U)
            embedded_U = embedded_U.unsqueeze(1)
            embedded_U = embedded_U.expand(embedded_U.size(0), embedded_S_gaps.size(1), embedded_U.size(2))

        #get the index of the last session representation of each user by subtracting 1 from each lengths, move to  for efficiency
        rep_indicies = Variable(torch.cuda.LongTensor(session_rep_lengths)) - 1

        #get initial hidden state of inter gru layer and call forward on the module
        inter_hidden = self.inter_rnn.init_hidden(S.size(0))
        if(self.flags["context"]):
            inter_last_hidden = self.inter_rnn(torch.cat((S, embedded_S_gaps, embedded_U),2), inter_hidden, rep_indicies)
        else:
            inter_last_hidden = self.inter_rnn(S, inter_hidden, rep_indicies)

        #get time scores and first prediction scores from the last hidden state of the inter RNN
        if(self.flags["temporal"]):
            times = self.time_linear(inter_last_hidden).squeeze()
            first_predictions = self.first_linear(inter_last_hidden).squeeze()

        #get item embeddings
        embedded_X = self.item_embed(X)

        #create average pooling session representation using the item embeddings and the lenght of each sequence
        lengths = Variable(torch.cuda.FloatTensor(session_lengths).view(-1,1)) #reshape the lengths in order to broadcast and use it for division
        sum_X = embedded_X.sum(1)
        mean_X = sum_X.div(lengths)

        #subtract 1 from the lengths to get the index of the last item in each sequence
        lengths = lengths.long()-1

        #call forward on the inter RNN
        recommendation_output, hidden_out = self.intra_rnn(embedded_X, inter_last_hidden, lengths)

        #store the new session representation based on the current scheme
        if(self.flags["use_hidden"]):
            self.datahandler.store_user_session_representations(hidden_out.data[0], user_list, time_targets)
        else:
            self.datahandler.store_user_session_representations(mean_X.data, user_list, time_targets)

        # LOSSES
        #prepare tensors for recommendation loss evaluation
        reshaped_Y = Y.view(-1)
        reshaped_rec_output = recommendation_output.view(-1,self.dims["N_ITEMS"]) #[SEQLEN*BATCHSIZE,N_items]

        #calculate recommendation losses
        reshaped_rec_loss = self.masked_cross_entropy_loss(reshaped_rec_output, reshaped_Y)
        #get mean losses based on actual number of valid events in batch
        sum_loss = reshaped_rec_loss.sum(0)
        divident = Variable(torch.cuda.FloatTensor([sum(session_lengths)]))
        mean_loss = sum_loss/divident

        if(self.flags["temporal"]):
            first_loss = self.masked_cross_entropy_loss(first_predictions, First_targets)
            sum_first_loss = first_loss.sum(0)
            mean_first_loss = sum_first_loss/embedded_X.size(0)


            #calculate the time loss
            time_loss = self.time_loss_func(times, T_targets)

            #mask out "fake session time-gaps" from time loss
            mask = Variable(T_targets.data.ge(self.time_threshold).float())
            time_loss = time_loss*mask

            #find number of non-ignored time gaps
            non_zero_count = 0
            for sign in mask.data:
                if (sign != 0):
                    non_zero_count += 1
            time_loss_divisor = Variable(torch.cuda.FloatTensor([max(non_zero_count,1)]))
            mean_time_loss = time_loss.sum(0)/time_loss_divisor

            #calculate gradients
            combined_loss = self.params["ALPHA"]*mean_time_loss + self.params["BETA"]*mean_loss + self.params["GAMMA"]*mean_first_loss
            combined_loss.backward()

            #update parameters through BPTT, options for freezing parts of the network
            if(self.flags["train_time"]):
                self.time_optimizer.step()
            if(self.flags["train_first"]):
                self.first_rec_optimizer.step()
            if(self.flags["train_all"]):
                self.inter_intra_optimizer.step()
        else:
            mean_loss.backward()
            self.inter_intra_optimizer.step()
        return mean_loss.data[0]

    def predict_on_batch(self, items, session_reps, sess_time_reps, user_list, item_targets, time_targets, first_rec_targets, session_lengths, session_rep_lengths, time_error):
        #get batch from datahandler and turn into variables
        X, S, S_gaps, U = self.process_batch_inputs(items, session_reps, sess_time_reps, user_list)

        #get embedded times
        if(self.flags["context"]):
            #get embedded times
            embedded_S_gaps = self.time_embed(S_gaps)

            #get embedded user
            embedded_U = self.user_embed(U)
            embedded_U = embedded_U.unsqueeze(1)
            embedded_U = embedded_U.expand(embedded_U.size(0), embedded_S_gaps.size(1), embedded_U.size(2))

        #get the index of the last session representation of each user by subtracting 1 from each lengths, move to  for efficiency
        rep_indicies = Variable(torch.cuda.LongTensor(session_rep_lengths)) - 1

        #get initial hidden state of inter gru layer and call forward on the module
        inter_hidden = self.inter_rnn.init_hidden(S.size(0))
        if(self.flags["context"]):
            inter_last_hidden = self.inter_rnn(torch.cat((S, embedded_S_gaps, embedded_U),2), inter_hidden, rep_indicies)
        else:
            inter_last_hidden = self.inter_rnn(S, inter_hidden, rep_indicies)

        #get time scores and first prediction scores from the last hidden state of the inter RNN
        if(self.flags["temporal"]):
            times = self.time_linear(inter_last_hidden).squeeze()
            first_predictions = self.first_linear(inter_last_hidden).squeeze()

            #calculate time error if this is desired
            if(time_error):
                w = self.time_loss_func.get_w()
                time_predictions = self.time_prediction(times.data, w.data)
                self.tester.evaluate_batch_time(time_predictions, time_targets)

        #get item embeddings
        embedded_X = self.item_embed(X)

        #create average pooling session representation using the item embeddings and the lenght of each sequence
        lengths = Variable(torch.cuda.FloatTensor(session_lengths).view(-1,1)) #reshape the lengths in order to broadcast and use it for division
        sum_X = embedded_X.sum(1)
        mean_X = sum_X.div(lengths)

        #subtract 1 from the lengths to get the index of the last item in each sequence
        lengths = lengths.long()-1

        #call forward on the inter RNN
        recommendation_output, hidden_out = self.intra_rnn(embedded_X, inter_last_hidden, lengths)

        #store the new session representation based on the current scheme
        if(self.flags["use_hidden"]):
            self.datahandler.store_user_session_representations(hidden_out.data[0], user_list, time_targets)
        else:
            self.datahandler.store_user_session_representations(mean_X.data, user_list, time_targets)
        
        if(self.flags["temporal"]):
            k_values, k_predictions = torch.topk(torch.cat((first_predictions.unsqueeze(1),recommendation_output),1), self.params["TOP_K"])
        else:
            k_values, k_predictions = torch.topk(recommendation_output, self.params["TOP_K"])
        return k_predictions
    

    

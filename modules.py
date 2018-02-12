import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


#embedding
class Embed(nn.Module):
    def __init__(self, input_dim, embedding_dim, item = False):
        super(Embed, self).__init__()
        self.embedding_table = nn.Embedding(input_dim, embedding_dim)

        #ensure that the representation of paddings are tensors of zeros, thus, will 
        #not contribute in potential average pooling session representations
        if(item):
            self.embedding_table.weight.data[0] = torch.zeros(embedding_dim)
    
    def forward(self, input):
        output = self.embedding_table(input)
        return output


#inter session RNN module
class Inter_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(Inter_RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)
    
    def forward(self, input, hidden, rep_indicies):
        input = self.dropout(input)
        gru_output, _ = self.gru(input, hidden)

        #find the last hidden state of each sequence in the batch which are not 
        hidden_indices = rep_indicies.view(-1,1,1).expand(gru_output.size(0), 1, gru_output.size(2))
        hidden_out = torch.gather(gru_output,1,hidden_indices)
        hidden_out = hidden_out.squeeze().unsqueeze(0)
        hidden_out = self.dropout(hidden_out)

        return hidden_out

    def init_hidden(self, batch_size):
        hidden = Variable(torch.cuda.FloatTensor(1, batch_size, self.hidden_dim).fill_(0))
        return hidden

#intra session RNN module
class Intra_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(Intra_RNN, self).__init__()      
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, input, hidden, lengths):
        input = self.dropout(input)
        gru_output, _ = self.gru(input, hidden)
        gru_output = self.dropout(gru_output)
        output = self.linear(gru_output)
        hidden_indices = lengths.view(-1,1,1).expand(gru_output.size(0), 1, gru_output.size(2))
        hidden_out = torch.gather(gru_output,1,hidden_indices)
        hidden_out = hidden_out.squeeze().unsqueeze(0)
        return output, hidden_out

#time loss module
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
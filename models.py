import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import DynamicLSTM, SqueezeEmbedding, SoftAttention
from tnet_lf import TNet_LF

class LSTM(nn.Module):
    ''' Standard LSTM '''
    def __init__(self, embedding_matrix, opt):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)
    
    def forward(self, inputs):
        text = inputs[0]
        x = self.embed(text)
        x_len = torch.sum(text != 0, dim=-1)
        _, (h_n, _) = self.lstm(x, x_len)
        out = self.dense(h_n[0])
        return out

class AE_LSTM(nn.Module):
    ''' LSTM with Aspect Embedding '''
    def __init__(self, embedding_matrix, opt):
        super(AE_LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(opt.embed_dim*2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)
    
    def forward(self, inputs):
        text, aspect_text = inputs[0], inputs[1]
        x_len = torch.sum(text != 0, dim=-1)
        x_len_max = torch.max(x_len)
        aspect_len = torch.sum(aspect_text != 0, dim=-1).float()
        '''
        x = [F.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1]  
        y = [F.relu(conv(feature.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
        x = [i * j for i, j in zip(x, y)]
        '''
        x = self.embed(text)
        x = self.squeeze_embedding(x, x_len)
        aspect = self.embed(aspect_text)
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.view(aspect_len.size(0), 1))
        aspect = torch.unsqueeze(aspect_pool, dim=1).expand(-1, x_len_max, -1)
        x = torch.cat((aspect, x), dim=-1)
        
        _, (h_n, _) = self.lstm(x, x_len)
        out = self.dense(h_n[0])
        return out

class ATAE_LSTM(nn.Module):
    ''' Attention-based LSTM with Aspect Embedding '''
    def __init__(self, embedding_matrix, opt):
        super(ATAE_LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(opt.embed_dim*2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention = SoftAttention(opt.hidden_dim, opt.embed_dim)
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)
    
    def forward(self, inputs):
        text, aspect_text = inputs[0], inputs[1]
        x_len = torch.sum(text != 0, dim=-1)
        x_len_max = torch.max(x_len)
        aspect_len = torch.sum(aspect_text != 0, dim=-1).float()
        
        x = self.embed(text)
        x = self.squeeze_embedding(x, x_len)
        aspect = self.embed(aspect_text)
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.view(aspect_len.size(0), 1))
        aspect = torch.unsqueeze(aspect_pool, dim=1).expand(-1, x_len_max, -1)
        x = torch.cat((aspect, x), dim=-1)
        
        h, _ = self.lstm(x, x_len)
        hs = self.attention(h, aspect)
        out = self.dense(hs)
        return out

class PBAN(nn.Module):
    ''' Position-aware bidirectional attention network '''
    def __init__(self, embedding_matrix, opt):
        super(PBAN, self).__init__()
        self.text_embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.pos_embed = nn.Embedding(opt.max_length, opt.position_dim)

        ''' GRU '''
        self.left_gru = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, 
                                    batch_first=True, bidirectional=True, rnn_type='GRU')
        self.right_gru = DynamicLSTM(opt.embed_dim+opt.position_dim, opt.hidden_dim, num_layers=1, 
                                     batch_first=True, bidirectional=True, rnn_type='GRU')
        
        '''TNet_LF'''
        self.tnet_lf = TNet_LF(embedding_matrix, opt)
    
        self.weight_m = nn.Parameter(torch.Tensor(opt.hidden_dim*2, opt.hidden_dim*2))
        self.bias_m = nn.Parameter(torch.Tensor(1))
        self.weight_n = nn.Parameter(torch.Tensor(opt.hidden_dim*2, opt.hidden_dim*2))
        self.bias_n = nn.Parameter(torch.Tensor(1))
        self.w_r = nn.Linear(opt.hidden_dim*2, opt.hidden_dim)
        self.w_s = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    
    def forward(self, inputs):
        text, aspect_text, position_tag = inputs[0], inputs[1], inputs[2]
        ''' Sentence representation '''
        x = self.text_embed(text)
        position = self.pos_embed(position_tag)
        x_len = torch.sum(text != 0, dim=-1)
        x = torch.cat((position, x), dim=-1)

        h_x, _ = self.right_gru(x, x_len.cpu())
        ''' Aspect term representation '''
        aspect = self.text_embed(aspect_text)
        aspect_len = torch.sum(aspect_text != 0, dim=-1)
        h_t, _ = self.left_gru(aspect, aspect_len.cpu())  #64,4,400
    
        '''-----------------加入门控---------------------------'''
        # context+position 64,66,400
        WX = torch.matmul(h_x, self.weight_x)  # 64,x,400
        s_sum = torch.add(WX, self.bias_x)  # 64,--,400
        s_i = F.tanh(s_sum)  # 64,--,400
        # aspect
        VA = torch.matmul(h_t, self.weight_v)  # 64,a,400
        a_sum = torch.add(VA, self.bias_v)  # 64,a,400

        m = torch.ones(a_sum.size())  # WX
        arr = m.permute(0,2,1)
        WT = torch.bmm(WX, arr)
        WT2 = torch.bmm(WT,a_sum)  #
        # print('mSize:', m.size())
        # print('arrSize:', arr.size())
        # print('WTSize:', WT.size())
        # print('WT2Size:', WT2.size())
        # print('tempSize:', temp.size())
        a_i = F.relu(torch.add(WT2,WX))
        c_i = torch.add(s_i, a_i)
        '''-----------------------------------------------------'''
        
        '''-----------------加入TNet_LF--------------------------'''
        h_i = tnet_lf(c_i)
        print('h_i:',h_i.size())
        
        '''-----------------------------------------------------'''

#         ''' Aspect term to position-aware sentence attention '''
#         alpha = F.softmax(F.tanh(torch.add(torch.matmul(torch.matmul(a_i, self.weight_m),
#                                                  torch.transpose(s_i, 1, 2)), self.bias_m)), dim=1)
#         s_x = torch.matmul(alpha, c_i)
#         ''' Position-aware sentence attention to aspect term '''
#         s_i_pool = torch.unsqueeze(torch.div(torch.sum(s_i, dim=1), x_len.float().view(x_len.size(0), 1)), dim=1)
#         gamma = F.softmax(F.tanh(torch.add(torch.matmul(torch.matmul(s_i_pool, self.weight_n),
#                                                  torch.transpose(c_i, 1, 2)), self.bias_n)), dim=1)
#         h_r = torch.squeeze(torch.matmul(gamma, s_x), dim=1)
#         ''' Output transform '''
#         out = F.tanh(self.w_r(h_r))
#         out = self.w_s(out)
        return out

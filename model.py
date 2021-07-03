import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

MAX_LEN = 128


class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        # embedding layer
        self.embed = nn.Embedding(input_dim, embed_dim)
        # rnn (bidirectional GRU)
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers, bidirectional=True)
        # add dropout layer after the embedding layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, hidden=None):
        embedded = self.dropout(self.embed(src))

        # outputs: output features from the last layer of the GRU
        # hidden: the final hidden state for each element in the batch
        outputs, hidden = self.gru(embedded, hidden)

        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_dim] + outputs[:, :, self.hidden_dim:])
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(self.hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        # repeat the hidden decoder_hidden_states for each time period
        h = hidden.repeat(encoder_outputs.size(0), 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [T*B*H] -> [B*T*H]
        attn_energies = self.calculate_score(h, encoder_outputs)
        # a_ij = exp(e_ij) / (sigma_k(exp(e_ik)))
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def calculate_score(self, hidden, encoder_outputs):
        # e_ij = a(s_{i-1}, h_j) s: hidden, h: encoder_outputs
        # [B*T*2H]->[B*T*H]
        # energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*T*H] -> [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_dim, embed_dim)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_dim)
        self.gru = nn.GRU(hidden_dim + embed_dim, hidden_dim,
                          n_layers)
        self.out = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input, last_hidden, encoder_outputs):
        # Use the last output word as the current input word
        embedded = self.dropout(self.embed(input).unsqueeze(0))  # (1,B,N)
        # Calculate attention weights
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        # attn_weights x encoder_outputs -> weighted outputs(attention score)
        # c_i = sigma_j(a_ij * h_j)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and context
        rnn_input = torch.cat([embedded, context], 2)
        # s_i = f(s_{i-1}, y_{i-1}, c_i)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)

        context = context.squeeze(0)
        # p(y_i|y_1,...,y_{i-1}, x) = g(y_{i-1}, s_i, c_i)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)

        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)  # src: [T*B*H] (T:Length of the sentence)
        max_len = trg.size(0) if trg is not None else MAX_LEN
        vocab_size = self.decoder.output_dim
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).to(self.device)

        encoder_output, hidden = self.encoder(src)
        # hidden: [num_layers, B, H]
        hidden = hidden[:self.decoder.n_layers]  # the hidden state of the first word
        output = Variable(trg.data[0, :]) if trg is not None else Variable(src[0, :])  # sos
        # pass the decoder network one by one
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(
                output, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            # the output will be used as the input for the next stage
            output = Variable(trg.data[t] if is_teacher else top1).to(self.device)

        return outputs

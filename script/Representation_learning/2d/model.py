import torch 
import torch.nn as nn

class RecurrentEncoder(nn.Module):
    """Recurrent encoder"""
    def __init__(self, input_size, hidden_size,device):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size,num_layers=1,batch_first=True)
    def forward(self, x, hidden_state):
        _, h_n = self.rnn(x,hidden_state)
        return h_n
    def init_hidden_states(self,bsize):
        h = torch.zeros(1,bsize,self.hidden_size).float().to(self.device)
        return h

class RecurrentDecoder(nn.Module):
    """Recurrent decoder for RNN and GRU"""
    def __init__(self, hidden_size, output_size, device):
        super().__init__()
        self.output_size = output_size
        self.device = device
        self.rec_dec1 = nn.GRUCell(output_size, hidden_size)
        self.dense_dec1 = nn.Linear(hidden_size, output_size)
    def forward(self, h_0, seq_len):
        # Initialize output
        x = torch.tensor([], device = self.device)
        # Squeezing
        h_i = h_0.squeeze()
        # Reconstruct first element with encoder output
        x_i = self.dense_dec1(h_i)

        # Reconstruct remaining elements
        for i in range(0, seq_len):
            h_i = self.rec_dec1(x_i, h_i)
            x_i = self.dense_dec1(h_i)
            x = torch.cat([x, x_i], axis=1)

        return x.view(-1, seq_len, self.output_size)

class RecurrentAE(nn.Module):
    """Recurrent autoencoder"""
    def __init__(self,input_size, output_size, hidden_size, device):
        super().__init__()
        # Encoder and decoder configuration
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        # Encoder and decoder
        self.encoder = RecurrentEncoder(self.input_size, self.hidden_size, self.device).to(device)
        self.decoder = RecurrentDecoder(self.hidden_size, self.output_size, self.device).to(device)

    def forward(self, x, hidden_state):
        seq_len = x.shape[1]
        h_n = self.encoder(x,hidden_state)
        out = self.decoder(h_n, seq_len)
        return torch.flip(out, [1])
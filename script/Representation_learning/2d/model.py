import torch 
import torch.nn as nn

def get_and_init_FC_layer(din, dout):
    li = nn.Linear(din, dout)
    nn.init.xavier_uniform_(
       li.weight.data, gain=nn.init.calculate_gain('relu'))
    li.bias.data.fill_(0.)
    return li

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
        self.dense_dec1 = get_and_init_FC_layer(hidden_size, output_size)
    def forward(self, h_0, seq_len):
        # Initialize output
        x = torch.tensor([], device = self.device)
        # Squeezing
        h_i = h_0.squeeze(0)
        # Reconstruct first element with encoder output
        x_i = self.dense_dec1(h_i)
        # Reconstruct remaining elements
        for i in range(0, seq_len):
            h_i = self.rec_dec1(x_i, h_i)
            x_i = self.dense_dec1(h_i)
            x = torch.cat([x, x_i], axis=1)

        return x.view(-1, seq_len, self.output_size)

class RecurrentAE(nn.Module):
    """Recurrent autoencoder
       input: a sequence of obs with size (B,L,51) and hidden state
       output: a sequence of obs with size (B,L,49*3+2)
    """
    def __init__(self,input_size, output_size, hidden_size, device, train=True):
        super().__init__()
        # Encoder and decoder configuration
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        # Encoder and decoder
        self.encoder = RecurrentEncoder(self.input_size, self.hidden_size, self.device).to(device)
        self.decoder = RecurrentDecoder(self.hidden_size, self.output_size, self.device).to(device)
        if train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

    def forward(self, x, hidden_state):
        seq_len = x.shape[1]
        h_n = self.encoder(x,hidden_state)
        out = self.decoder(h_n, seq_len)
        return torch.flip(out, [1]), h_n

class SNAC_Lnet(nn.Module):
    """Recurrent Lnet
       input: a sequence of obs with size (B,L,51) and hidden state
       output: a sequence of obs with size (B,L,49*3+2)
    """
    def __init__(self,input_size, hidden_size, device, Loss_type="L2"):
        super().__init__()
        # Encoder and decoder configuration
        self.hidden_size = hidden_size
        self.input_size = input_size
        if Loss_type == "L2": 
            self.output_size = 2
        else:
            self.output_size = 26*26 
        self.device = device
        self.Loss_type = Loss_type
        # Encoder and decoder
        self.rnn = nn.LSTM(self.input_size, self.hidden_size,batch_first=True).to(device)
        if Loss_type == "L2":
            self.MLP = nn.Sequential(
                        get_and_init_FC_layer(self.hidden_size,64),
                        nn.ReLU(),
                        get_and_init_FC_layer(64,16),
                        nn.ReLU(),
                        get_and_init_FC_layer(16,self.output_size),
                        nn.ReLU())
        else: 
            self.MLP = nn.Sequential(
                        get_and_init_FC_layer(self.hidden_size,256),
                        nn.ReLU(),
                        get_and_init_FC_layer(256,512),
                        nn.ReLU(),
                        get_and_init_FC_layer(512,self.output_size),
                        nn.LogSoftmax(dim=2))

    def forward(self, x, pos, hidden_state, cell_state):
        """
        x: size (B,L,K), K = 51+51+1 (two obs + action)
        pos: size (B,L,2)
        """
        seq_len = x.shape[1]
        B_size = x.shape[0]
        predicted_pos = []
        input_pos = pos[:,0:1,:]
        for i in range(0, seq_len):
            output, (hidden_state,cell_state) = self.rnn(torch.cat((x[:,i:i+1,:],input_pos),dim=2),(hidden_state,cell_state)) ### output shape = (B,1,hidden size)
            if self.Loss_type == "L2":
                next_pos = self.MLP(output) ## next pos size (B,1,2)
                predicted_pos.append(next_pos)
                input_pos = next_pos
            else:
                next_pos = self.MLP(output) ## next pos size (B,1,576), 26**2
                next_pos = next_pos.view(B_size,1,26,26)
                predicted_pos.append(next_pos)
                if i >= (seq_len-1):
                    break
                input_pos = pos[:,i+1:i+2,:]
        return predicted_pos, hidden_state, cell_state
        
    def init_hidden_states(self,bsize):
        h = torch.zeros(1,bsize,self.hidden_size).float().to(self.device)
        c = torch.zeros(1,bsize,self.hidden_size).float().to(self.device)
        return h,c

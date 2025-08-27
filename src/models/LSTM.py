import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional, dropout_rnn=0.3,  dropout=0.3,):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout)
        # RNN layer
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout_rnn)
        
        # Output layer
        if self.bidirectional:
            self.fc = nn.Linear(hidden_size*2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid() 
        self.rnn.apply(self.init_weights)   
        
        
    def init_weights(self, layer):
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        #print(x.shape)
        if len(x.shape)<3:
            x = x.unsqueeze(-1)
        elif len(x.shape)>3:
            x = x.squeeze(1)
        else:  # in case of stft
            x = x.permute(0, 2, 1)
        # Initialize hidden state
        if self.bidirectional:
            h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        
        # Forward propagate the RNN
        out, _ = self.rnn(x, (h0, c0))
        # Get the output of the last time step
        out = out[:, -1, :]  # Get the last time step's output
        out = self.dropout(out)
        out = self.fc(out)   # Pass it through the output layer
        out = self.sigmoid(out)
        out = out.squeeze(-1)
        return out
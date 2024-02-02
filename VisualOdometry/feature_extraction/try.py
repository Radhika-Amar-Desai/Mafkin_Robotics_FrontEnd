import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class CNN_RNN(nn.Module):
    def __init__(self, cnn_hidden_size, rnn_hidden_size, num_classes):
        super(CNN_RNN, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the input size for the RNN
        # Assuming input image size is 32x32
        self.rnn_input_size = 64 * 8 * 8  # output size after conv2 and pooling
        
        # RNN layers
        self.rnn = nn.GRU(input_size=self.rnn_input_size, hidden_size=rnn_hidden_size, num_layers=1, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(rnn_hidden_size, num_classes)
        
    def forward(self, x):
        print("Input shape:", x.size())
        
        batch_size, seq_length, c, h, w = x.size()
        
        # CNN forward
        x = x.view(batch_size * seq_length, c, h, w)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        print("After conv1 and pool:", x.size())
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        print("After conv2 and pool:", x.size())
        
        # Reshape for RNN
        x = x.view(batch_size, seq_length, -1)  # reshape (batch_size, seq_length, -1)
        
        # RNN forward
        _, h_n = self.rnn(x)
        
        # Take the last hidden state of the RNN
        h_n = h_n.squeeze(0)
        
        # Fully connected layer
        out = self.fc(h_n)
        print("Output shape:", out.size())
        
        return out

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # Define two instances of the CNN_RNN network (sub-networks)
        self.cnn_rnn1 = CNN_RNN(cnn_hidden_size=64, 
                                rnn_hidden_size=128, 
                                num_classes=10)
        self.cnn_rnn2 = CNN_RNN(cnn_hidden_size=64, 
                                rnn_hidden_size=128, 
                                num_classes=10)
        
    def forward(self, input1, input2):
        # Forward pass through both branches of the Siamese network
        output1 = self.cnn_rnn1(input1)
        output2 = self.cnn_rnn2(input2)
        return utils.euclidean_distance ( output1, output2 )

# Example usage
# Create an instance of the SiameseNetwork model
# model = SiameseNetwork()
# print(model)

model = SiameseNetwork()

# Create sample tensors of appropriate dimensions
batch_size = 1
channels = 3
height = 32
width = 32

# Create input tensors with sequences of length 2
input1 = torch.randn(batch_size, 2, channels, height, width)
input2 = torch.randn(batch_size, 2, channels, height, width)

# Pass the input tensors through the model
output = model(input1, input2)

print ( output )

# # Print the output shapes
# print("Output shape from branch 1:", output1.shape)
# print("Output shape from branch 2:", output2.shape)

# dist = utils.euclidean_distance ( output1 , output2 )

# print ( dist )
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import SiameseNetwork

# Assuming you have your dataset and data loaders defined


# Initialize the SiameseNetwork model
model = SiameseNetwork()

# Define loss function and optimizer
criterion = nn.ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, data in enumerate(train_loader, 0):
        # Get inputs; data is a list of [inputs1, inputs2, labels]
        inputs1, inputs2, labels = data
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs1, outputs2 = model(inputs1, inputs2)
        
        # Compute the contrastive loss
        loss = criterion(outputs1, outputs2)  # You might need to define your own contrastive loss
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # Print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

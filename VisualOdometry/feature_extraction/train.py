import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import functional as F
from model import SiameseNetwork
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import config
from torchvision import transforms

# Define the Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidian distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


      return loss_contrastive

# Assuming you have your dataset and data loaders defined
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # self.root_dir = root_dir
        # self.transform = transform

        # self.image_folder = os.path.join(root_dir, 'images')
        # self.mask_folder = os.path.join(root_dir, 'labels')
        # self.grad_cam_folder = os.path.join ( root_dir, 'grad_cam_images' )

        # self.image_list = sorted(os.listdir(self.image_folder))
        # self.mask_list = sorted(os.listdir(self.mask_folder))
        # self.grad_cam_list = sorted (os.listdir (self.grad_cam_folder))
        
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))

        self.inputs1 = []
        self.inputs2 = []
        self.labels = []

        # Load images and labels
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for image_pair_folder in os.listdir ( class_dir ):
                image_pair_folder_path = os.path.join ( class_dir, 
                                                image_pair_folder )
                image1_folder_path, image2_folder_path = \
                    [os.path.join ( image_pair_folder_path, image_pair )\
                    for image_pair in \
                    sorted(os.listdir ( image_pair_folder_path ))]
            
                self.inputs1.append ( 
                    [os.path.join ( image1_folder_path, file ) \
                    for file in os.listdir ( image1_folder_path )])
                
                self.inputs2.append (
                    [os.path.join ( image2_folder_path, file ) \
                    for file in os.listdir ( image2_folder_path )]
                )

                self.labels.append ( class_name )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load image and mask
        # img_path = os.path.join(self.image_folder, 
        #                         self.image_list[idx])
        # mask_path = os.path.join(self.mask_folder, 
        #                         self.mask_list[idx])
        # grad_cam_path = os.path.join(self.grad_cam_folder,
        #                             self.grad_cam_list[idx] )
        # image = Image.open( img_path ).convert('RGB')
        # mask = Image.open( mask_path ).convert('L')  # Assuming masks are grayscale
        # grad_cam_img = Image.open ( grad_cam_path ).convert('RGB')
        # #grad_cam_img = image
        # # Apply transformations if provided
        # if self.transform:
        #     image = self.transform( image )
        #     grad_cam_img = self.transform (grad_cam_img)
        #     mask = self.transform( mask )

        # return image, grad_cam_img , mask
        
        blob1_path, image1_path = self.inputs1 [ idx ]
        blob2_path, image2_path = self.inputs2 [ idx ]

        label = int ( self.labels [ idx ] == "similar" )

        blob1 = self.transform ( Image.open ( blob1_path ).convert('RGB') )
        image1 = self.transform ( Image.open ( image1_path ).convert('RGB') ) 
        
        blob2 = self.transform ( Image.open ( blob2_path ).convert('RGB') )
        image2 = self.transform ( Image.open ( image2_path ).convert('RGB') ) 

        input1 = torch.cat((blob1.unsqueeze(0), image1.unsqueeze(0)), dim=0)
        input2 = torch.cat((blob2.unsqueeze(0), image2.unsqueeze(0)), dim=0)
      
        return input1, input2, label

# transforms = transforms.Compose([
#     transforms.Resize((config.INPUT_IMAGE_HEIGHT,
#                     config.INPUT_IMAGE_WIDTH)),
#     transforms.ToTensor()])

# ds = CustomDataset ( r"VisualOdometry\feature_extraction\dataset_for_model\train",
#                     transforms )

# print ( ds[2] )

if __name__ == "__main__":

    transforms = transforms.Compose([
        transforms.Resize((config.INPUT_IMAGE_HEIGHT,
                        config.INPUT_IMAGE_WIDTH)),
        transforms.ToTensor()])

    train_dict = r"VisualOdometry\feature_extraction\dataset_for_model\train"
    test_dict = r"VisualOdometry\feature_extraction\dataset_for_model\test"

    trainDS = CustomDataset ( train_dict , transforms )
    testDS = CustomDataset ( test_dict , transforms )
    
    train_loader = DataLoader(trainDS, shuffle=True,
            batch_size=1, pin_memory=config.PIN_MEMORY,
            num_workers=os.cpu_count(), drop_last=True)

    test_loader = DataLoader(testDS, shuffle=True,
            batch_size=4, pin_memory=config.PIN_MEMORY,
            num_workers=os.cpu_count(), drop_last=True)

    # Initialize the SiameseNetwork model
    model = SiameseNetwork()

    # Define loss function and optimizer
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        print ( "Epoch : ", epoch )
        model.train()
        running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            print ( "Batch : ", i )
            running_loss = 0.0
            # Get inputs; data is a list of [inputs1, inputs2, labels]
            inputs1, inputs2, labels = data
            print ( "Data Received", inputs1.shape )
            # Zero the parameter gradients
            optimizer.zero_grad()
            print ( "Zeroed the parameter gradients" )            
            # Forward pass
            outputs1, outputs2 = model(inputs1, inputs2)
            print ( "Forward pass done." )

            # Compute the contrastive loss
            loss = criterion(outputs1, outputs2, labels)  # You might need to define your own contrastive loss
            print ( "Computed loss" ) 

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            print ( "Backward pass and optimize." )

            # Print statistics
            running_loss += loss.item()
            print('[%d %d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                
    print('Finished Training')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, r'VisualOdometry\feature_extraction\model.pth')
    
    print ('Model Saved')

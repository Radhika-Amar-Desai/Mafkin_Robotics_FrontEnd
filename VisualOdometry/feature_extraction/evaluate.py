import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import functional as F
from model import SiameseNetwork  # Assuming you have defined the SiameseNetwork model
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import config
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate the euclidean distance and calculate the contrastive loss
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                       (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_pair_folder_paths = [os.path.join(root_dir, folder) \
                                         for folder in os.listdir(root_dir)]

        print ( self.image_pair_folder_paths )

    def __len__(self):
        return len(self.image_pair_folder_paths)

    def __getitem__(self, idx):
        image_pair_folder_path = self.image_pair_folder_paths[idx]
        image1_folder_path, image2_folder_path = \
            [os.path.join(image_pair_folder_path, image_file) \
             for image_file in sorted(os.listdir(image_pair_folder_path))]

        blob1, image1 = \
            [self.transform(Image.open(
                os.path.join(image1_folder_path, file)).convert('RGB')) \
             for file in sorted(os.listdir(image1_folder_path))]

        blob2, image2 = \
            [self.transform(Image.open(
                os.path.join(image2_folder_path, file)).convert('RGB')) \
             for file in sorted(os.listdir(image2_folder_path))]

        return [blob1, image1], [blob2, image2], int("similar" in self.image_pair_folder_paths[idx])

if __name__ == "__main__":
    # Transforms for preprocessing images
    transforms = transforms.Compose([
        transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)),
        transforms.ToTensor()])

    # Load the validation dataset
    validation_dataset = CustomDataset(
        root_dir=r"VisualOdometry\feature_extraction\dataset_for_model",
        transform=transforms)

    # Data loader for validation dataset
    validation_loader = DataLoader(validation_dataset, shuffle=False,
                                   batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                                   num_workers=os.cpu_count(), drop_last=False)

    # Initialize the SiameseNetwork model
    model = SiameseNetwork()

    # Load the model checkpoint
    model_checkpoint = torch.load(r'VisualOdometry\feature_extraction\model.pth')
    model.load_state_dict(model_checkpoint['model_state_dict'])
    model.eval()

    # Define loss function
    criterion = ContrastiveLoss()

    # Define lists to store predictions and true labels
    predictions = []
    true_labels = []

    # Evaluate the model on the validation dataset
    with torch.no_grad():
        for data in validation_loader:
            # Get inputs; data is a list of [inputs1, inputs2, labels]
            inputs1, inputs2, labels = data

            # Forward pass
            outputs1, outputs2 = model(inputs1, inputs2)

            # Compute predictions
            distances = F.pairwise_distance(outputs1, outputs2)
            predictions.extend(distances.cpu().numpy() < 0.5)  # Adjust the threshold as needed

            # Collect true labels
            true_labels.extend(labels.cpu().numpy())

    # Compute evaluation metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')

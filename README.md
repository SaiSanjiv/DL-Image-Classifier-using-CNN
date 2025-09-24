# DL-Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY
The FashionMNIST dataset consists of 70,000 grayscale images of 10 fashion categories (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot), each of size 28×28 pixels.
The task is to classify these images into their respective categories.
CNNs are well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

## Neural Network ModelThe CNN used has:

Conv Layer 1: 1 → 32 filters (3×3, padding=1) + ReLU + MaxPooling
Conv Layer 2: 32 → 64 filters (3×3, padding=1) + ReLU + MaxPooling
Conv Layer 3: 64 → 128 filters (3×3, padding=1) + ReLU + MaxPooling
Fully Connected Layers:
* FC1: 128×3×3 → 128 neurons
* FC2: 128 → 64 neurons
* FC3: 64 → 10 classes

## DESIGN STEPS
### STEP 1:
Import required libraries (Torch, Torchvision, Matplotlib, Seaborn, Scikit-learn).

### STEP 2: 
Apply transformations (Tensor conversion + normalization) to dataset.

### STEP 3: 
Download FashionMNIST dataset and prepare DataLoaders for training & testing.

### STEP 4: 
Build a CNN architecture with convolutional, pooling, and fully connected layers.

### STEP 5: 
Define loss function (CrossEntropyLoss) and optimizer (Adam).

### STEP 6: 
Train the model for 3 epochs, evaluate on test data, and visualize metrics.

## PROGRAM
### Name: SAI SANJIV R
### Register Number: 212223230179

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Data Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load Dataset
train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# CNN Model
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train_model(model, train_loader, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

train_model(model, train_loader)

# Testing Loop
def test_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print('Name: SAI SANJIV R')
    print('Register Number: 212223230179')
    print(f'Test Accuracy: {accuracy:.4f}')

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

test_model(model, test_loader)

# Prediction on New Sample
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        _, predicted = torch.max(output, 1)
    class_names = dataset.classes
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f'Actual: {class_names[label]} | Predicted: {class_names[predicted.item()]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted.item()]}')

predict_image(model, image_index=80, dataset=test_dataset)


```

### OUTPUT

## Training Loss per Epoch
<img width="649" height="215" alt="image" src="https://github.com/user-attachments/assets/c6a4f8e8-41e5-48b8-a3c9-183bc50fa4fc" />


## Confusion Matrix
<img width="944" height="897" alt="image" src="https://github.com/user-attachments/assets/76645663-eb90-4ab9-9da1-bffa37f0eb04" />


## Classification Report
<img width="443" height="352" alt="image" src="https://github.com/user-attachments/assets/7f34b8b8-c417-4e8f-8b53-528e9eab75fa" />


### New Sample Data Prediction
<img width="409" height="494" alt="image" src="https://github.com/user-attachments/assets/4043eca8-99a9-404b-be94-3dcc8b5086d5" />


## RESULT
A Convolutional Neural Network was successfully developed and trained on the FashionMNIST dataset.
The model achieved good classification performance, and predictions on unseen data were correctly visualized.

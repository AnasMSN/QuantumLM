''' Classifier
1. Input Image
2. Model (CNN, Transformer, ResNet) - DNN, ML Model (Random forest, XGBoost)
3. 1-20 output number of qubit, type identification (6)
4. Calculate accuracy
5. Feature engineering for LLM
'''

# Code goes here

# start generate me a code to 

import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import torchvision.models as models

# dataset path
DATASET_PATH = "generated_images_qutip_grayscale"

BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard normalization
])

dataset = ImageFolder(root=DATASET_PATH, transform=transform)

#split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# dataloader 
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)

NUM_CLASSES = len(dataset.classes)
print(f"Detected {NUM_CLASSES} classes.")

model = models.resnet50(pretrained=True)

model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def evaluate_model(model, val_loader):
    model.eval()
    correct, total = 0,0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return 100.0 * (correct/ total)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels  = images.to(device), labels.to(device)
            
            # forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # backward pass
            loss.backward()
            optimizer.step()
            
            # track loss and accuracy
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100.0 * correct / total
        val_acc = evaluate_model(model, val_loader=val_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            
            
train_model(model, train_loader, val_loader, criterion, optimizer=optimizer, num_epochs=NUM_EPOCHS)

torch.save(model.state_dict(), "image_classifier.pth")
print("Model saved!")
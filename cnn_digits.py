import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import pygame
import cv2


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)   
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  
        self.pool = nn.MaxPool2d(2, 2)                
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64*14*14, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 64*14*14)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#load MNIST dataset
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1,0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
testset  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]))

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader  = DataLoader(testset, batch_size=64, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# training
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader):.4f}")

#test the model
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test Accuracy: {100*correct/total:.2f}%")


# pygame
def process_drawing(screen):
    surface = pygame.surfarray.array3d(screen)
    gray = np.dot(surface[..., :3], [0.2989, 0.587, 0.114])
    gray = np.transpose(gray, (1,0))
    gray = cv2.resize(gray, (28,28), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32)/255.0
    gray = (gray - 0.5)/0.5
    tensor = torch.tensor(gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)

def predict_digit(screen):
    img_tensor = process_drawing(screen)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
    return pred.item()

def draw_digit():
    pygame.init()
    window_size = 280
    display_height = window_size + 50
    screen = pygame.display.set_mode((window_size, display_height))
    pygame.display.set_caption("Draw Digit")
    clock = pygame.time.Clock()
    drawing = False
    prediction = None
    font = pygame.font.Font(None, 36)
    screen.fill((0,0,0))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                prediction = predict_digit(screen)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    screen.fill((0,0,0))
                    prediction = None
                    pygame.display.flip()
            if event.type == pygame.MOUSEMOTION and drawing:
                pygame.draw.circle(screen, (255,255,255), event.pos, 8)

        screen.fill((0,0,0), (0, window_size, window_size, 50)) # clear bottom text area
        if prediction is not None:
            text = font.render(f"Prediction: {prediction}", True, (0,255,0))
            screen.blit(text, (10, window_size + 10))

        pygame.display.flip()
        clock.tick(60)

draw_digit()

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision 
import numpy as np
import torchvision.transforms as transforms
import pygame
from torch.utils.data import DataLoader
import cv2

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(28*28, 150)
        self.fc2 = nn. Linear(150, 150)
        self.fc3 = nn.Linear(150, 150)
        self.fc4 = nn.Linear(150, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
#load MNIST dataset

transform = transforms.Compose([
    transforms.ToTensor(), #image to tensor
    transforms.Normalize((0.5,), (0.5,)) #pixel values to [-1,1]
])

trainset = torchvision.datasets.MNIST(root="./data", train = True, download = True, transform=transform)
testset = torchvision.datasets.MNIST(root="./data", train = False, download = True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle = True)
testloader = DataLoader(trainset, batch_size=64, shuffle = False)

model = ANN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

episodes = 15

for epoch in range(episodes):
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}")


#Test the model

correct = 0
total = 0
model.eval()

with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, prediction = torch.max(outputs, 1) # highest probability from 0-9
        total += labels.size(0)
        correct += (prediction == labels).sum().item()
print(f"Accuracy: {100*correct / total:.2f}%")


#pygame

def draw_digit():
    pygame.init()
    window_size = 280
    display_height = window_size + 50
    screen = pygame.display.set_mode((window_size, display_height))
    pygame.display.set_caption("Draw Digit")
    clock = pygame.time.Clock()
    screen.fill((0, 0, 0))
    drawing = False
    prediction = None

    font = pygame.font.Font(None, 36)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                prediction = predict_digit(screen)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:  # Clear screen
                    screen.fill((0, 0, 0))
                    prediction = None
            elif event.type == pygame.MOUSEMOTION and drawing:
                # Draw a white circle at mouse position
                pygame.draw.circle(screen, (255, 255, 255), event.pos, 8)

        # Update display every frame
        pygame.display.flip()
        screen.fill((0, 0, 0), (0, window_size, window_size, 50))  # clear only prediction area

        # Show prediction text
        if prediction is not None:
            text = font.render(f"Prediction: {prediction}", True, (0, 255, 0))
            screen.blit(text, (10, window_size + 10))

        clock.tick(60)


def process_drawing(screen):
    surface = pygame.surfarray.array3d(screen)
    gray = np.dot(surface[..., :3], [0.2989, 0.587, 0.114])
    gray = np.transpose(gray, (1,0))
    gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32) / 255.0
    gray = (gray - 0.5) / 0.5
    tensor = torch.tensor(gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor

#prediction

def predict_digit(screen):
    image = process_drawing(screen)
    if image is None:
        return None
    
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, prediction = torch.max(output, 1)
    return prediction.item()

draw_digit()
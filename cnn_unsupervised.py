import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
from k_means_constrained import KMeansConstrained
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

WIDTH, HEIGHT = 280, 280
DRAW_COLOR = (255, 255, 255)
BG_COLOR = (0, 0, 0)
BRUSH_SIZE = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#CNN_extracts_features_of_digit
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  #14x14
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   #7x7
        )
        self.fc = nn.Linear(64 * 7 * 7, 128)  #feature_size=128

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.relu(x)
        return x

#train_CNN_with_MNIST
def train_cnn_feature_extractor(model, epochs=5, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_train = datasets.MNIST(root="./mnist", train=True, download=True, transform=transform)
    loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #add_classification
    classifier = nn.Linear(128, 10).to(device)
    model.train()
    classifier.train()

    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            features = model(imgs)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} CNN Train Loss: {total_loss/len(loader):.4f}")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

#preprocess_drawing
def preprocess(surface):
    data = pygame.surfarray.array3d(surface)
    data = np.mean(data, axis=2)
    data = np.transpose(data, (1, 0))  
    img = Image.fromarray(data.astype(np.uint8)).resize((28, 28))
    img = np.array(img) / 255.0
    img = (img - 0.5) / 0.5
    return img.reshape(1, 28, 28)

#Extract_CNN_features_for_clustering
def extract_features(model, data):
    features = []
    with torch.no_grad():
        for img in data:
            x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)
            feat = model(x).cpu().numpy()[0]
            features.append(feat)
    return np.array(features)

#balanced_clustering
def cluster_and_show_balanced(features, data, n_clusters=10):
    n_samples = len(data)
    if n_samples % n_clusters != 0:
        print("Warning: samples not divisible by clusters; some clusters may differ in size.")
    size_per_cluster = n_samples // n_clusters

    kmeans = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=size_per_cluster,
        size_max=size_per_cluster,
        random_state=0
    )
    labels = kmeans.fit_predict(features)

    clusters = {}
    for i, lbl in enumerate(labels):
        clusters.setdefault(lbl, []).append(data[i])

    for cluster_id, imgs in clusters.items():
        plt.figure(figsize=(5, 5))
        plt.suptitle(f"Cluster {cluster_id}")
        for i, img in enumerate(imgs[:16]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(img[0], cmap="gray")
            plt.axis("off")
        plt.show()

#pygametricks
def run_app():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Draw Digits (Press S to save, Q to finish)")
    canvas = pygame.Surface((WIDTH, HEIGHT))
    canvas.fill(BG_COLOR)

    drawings = []
    running = True
    drawing = False

    while running:
        screen.blit(canvas, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    img = preprocess(canvas)
                    drawings.append(img)
                    print(f"Saved drawing #{len(drawings)}")
                    canvas.fill(BG_COLOR)
                elif event.key == pygame.K_c:
                    canvas.fill(BG_COLOR)
                elif event.key == pygame.K_q:
                    running = False

        if drawing:
            x, y = pygame.mouse.get_pos()
            pygame.draw.circle(canvas, DRAW_COLOR, (x, y), BRUSH_SIZE)

    pygame.quit()
    return drawings


if __name__ == "__main__":
    drawings = run_app()

    if len(drawings) < 10:
        print("Draw more digits!")
        exit()

    cnn_model = CNNFeatureExtractor().to(device)
    print("Training CNN feature extractor on MNIST...")
    train_cnn_feature_extractor(cnn_model, epochs=5)

    print("Extracting features from your drawings...")
    features = extract_features(cnn_model, drawings)

    print("Clustering drawings into equal-sized clusters...")
    cluster_and_show_balanced(features, drawings, n_clusters=10)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image


# 🔹 Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")


# 🔹 Define Dataset Class for Training
class FireDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []  # 0: No Fire, 1: Fire, 2: Smoke


        classes = ["default", "fire", "smoke"]
        for label, cls in enumerate(classes):
            class_path = os.path.join(root_dir, cls)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.data.append(img_path)
                self.labels.append(label)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# 🔹 Define Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# 🔹 Load Training Data
train_dataset = FireDataset(root_dir="fire data/img_data/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


# 🔹 Load Pretrained Model (MobileNetV2)
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(1280, 3)  # Modify output layer for 3 classes
model = model.to(device)


# 🔹 Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 🔥 Training the Model
print("\n🔥 Training Model...")
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
       
        running_loss += loss.item()
       
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
   
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")


# ✅ Save Trained Model
torch.save(model.state_dict(), "fire_detection_model.pth")
print("\n✅ Model trained and saved as 'fire_detection_model.pth'")


# 🔹 Load Model for Inference
model.load_state_dict(torch.load("fire_detection_model.pth", map_location=device))
model.eval()


# 🔹 Function to Predict Fire in a Frame
def predict_fire(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert OpenCV format to PIL Image
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0).to(device)
   
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
   
    return predicted.item()  # Returns 0 (No Fire), 1 (Fire), or 2 (Smoke)


# 🎥 Process Video for Overall Fire Detection and Display Frame with Detection
def detect_fire_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
   
    fire_detected = False  # Flag to check if fire is found in any frame
    total_frames = 0
    fire_frames = 0


    print("\n🎥 Processing Video for Fire Detection...")
   
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        total_frames += 1
        label = predict_fire(frame)


        if label == 1:  # If "Fire" is detected
            fire_detected = True
            fire_frames += 1
            print(f"🔥 Fire Detected at Frame {total_frames}")
           
            # Display the frame with a label indicating fire using matplotlib
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            plt.imshow(frame_rgb)
            plt.title("Fire Detected!")
            plt.axis('off')  # Hide axes
            plt.show()
            break  # Fire detected, stop processing the video


    cap.release()


    # 🔹 Final Decision Based on Fire Frames Detected
    if fire_detected:
        print("\n🔥 Fire Detected in Video!")
    else:
        print("\n✅ No Fire Detected in Video.")


# 🔹 Run Fire Detection on a Video
video_path = "fire data/video_data/test_videos/test1.mp4"  # Change this path as needed
detect_fire_in_video(video_path)

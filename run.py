import torch
from PIL import Image, ImageDraw
import numpy as np
import tkinter as tk
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

model = Net()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Transform to convert image to tensor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Tkinter interface for drawing
class DrawApp:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()
        self.button_clear = tk.Button(root, text='Clear', command=self.clear_canvas)
        self.button_clear.pack()
        self.button_predict = tk.Button(root, text='Predict', command=self.predict)
        self.button_predict.pack()
        self.canvas.bind('<B1-Motion>', self.paint)
        self.image = Image.new('L', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=10)
        self.draw.ellipse([x1, y1, x2, y2], fill='black')

    def clear_canvas(self):
        self.canvas.delete('all')
        self.draw.rectangle([0, 0, 280, 280], fill='white')

    def predict(self):
        img = self.image.resize((28, 28))
        img_tensor = transform(img).unsqueeze(0)
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1)
        print(f'Predicted digit: {prediction.item()}')

if __name__ == '__main__':
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()
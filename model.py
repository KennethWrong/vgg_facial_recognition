import torchvision as tv
import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import datetime
import os

class model_manager:
    def __init__(self):
        self.model = tv.models.resnet18(pretrained = True)
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        print(f"Cuda available: {torch.cuda.is_available()}, device_count: {torch.cuda.device_count()} curr_device: {self.device}")
        # self.set_up_model()
        self.set_up_resnset()
    
    def set_up_model(self):
        self.model = self.model.to(self.device)

        # This is for freezing feature parameters 
        for param in self.model.features.parameters():
            param.requires_grad = False
        
        
        self.model.classifier = torch.nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=100, bias=True),
            nn.Softmax(dim=1),
        ).to(self.device)
    
    def set_up_resnset(self):
        self.model.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=100)
        )

        for param in self.model.parameters():
            param.requires_grad = True
        
        # self.model = self.model.float()
    
    def save_model(self):
        today = datetime.datetime.now()
        filename = today.strftime("%d-%m-%Y-%H-%M")

        filename = "resnet18-weights"+"-"+filename
        cwd = os.path.dirname(os.path.realpath(__file__))

        weight_path = os.path.join(cwd, "weights", filename)

        torch.save(self.model.state_dict(), weight_path)

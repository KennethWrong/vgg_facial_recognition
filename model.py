import torchvision as tv
import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

class vgg_model:
    def __init__(self):
        self.vgg16 = tv.models.vgg16(weights=tv.models.VGG16_Weights('IMAGENET1K_V1'))
        self.preprocess_layer = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.CrossEntropyLoss()
    
    def preprocess(self, image):
        input_image = self.preprocess_layer(image)

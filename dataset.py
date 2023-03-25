import os
import torch
import torchvision as tv
from torchvision.io import read_image
import glob
import re
from PIL import Image
import pandas as pd

class ImageDataset:
    def __init__(self):
        self.images = []
        self.labels = []
        self.preprocess_layer = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.labels_n_to_i_dic = {}
        self.labels_i_to_n_dic = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. we got to load the images from .jpg to PIL image or np.array
    def load_dataset(self, path):
        cwd = os.path.dirname(os.path.realpath(__file__))

        train_data_path = os.path.join(cwd, "data", "train")

        def generate_index(x):
            res = re.search(r".*\/([0-9]+).jpg", x)
            value = int(res.groups()[0])
            return value

        for images in sorted(glob.glob(os.path.join(train_data_path, "*")), key= lambda x: generate_index(x)):
            image = Image.open(images)
            self.images.append(image)
    
    def load_training_labels(self, path):
        cwd = os.path.dirname(os.path.realpath(__file__))
        train_labels_path = os.path.join(cwd, "data", "train.csv")
        train_labels_df = pd.read_csv(train_labels_path)
        names = train_labels_df['Category'].values.tolist()
        names = [self.labels_name_to_index[n] for n in names]
        labels = torch.Tensor(names)
        self.labels = labels
        return labels
    
    def load_categories(self, path):
        cwd = os.path.dirname(os.path.realpath(__file__))
        labels_path = os.path.join(cwd, "data", "category.csv")
        labels_df = pd.read_csv(labels_path)
        labels = labels_df['Category'].values.tolist()

        for i, name in enumerate(labels):
            self.labels_n_to_i[name] = i
            self.labels_i_to_n[i] = name
    
    # def preprocess_images(self):


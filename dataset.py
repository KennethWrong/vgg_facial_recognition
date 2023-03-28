import os
import torch
import torchvision as tv
from torchvision.io import read_image
from PIL import Image, ImageFile
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets
import re
import glob

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = self.load_training_labels()
        self.transform = transform
        self.labels_n_to_i_dic = {}
        self.labels_i_to_n_dic = {}
        self.target_transform = target_transform
    
    # 1. we got to load the images from .jpg to PIL image or np.array
    def load_image(self, index):
        image_path = os.path.join(self.img_dir, f"{str(index)}.jpg")

        image = Image.open(image_path)
        image = image.convert('RGB')
        

        transform = tv.transforms.PILToTensor()
        image = transform(image)

        return image
    
    def load_dataset(self):
        cwd = os.path.dirname(os.path.realpath(__file__))

        train_data_path = os.path.join(cwd, "data", "train")

        images_array = []

        def generate_index(x):
            res = re.search(r".*\/([0-9]+).jpg", x)
            value = int(res.groups()[0])
            return value

        for images in sorted(glob.glob(os.path.join(train_data_path, "*")), key= lambda x: generate_index(x)):
            image = Image.open(images)
            image = image.convert('RGB')
            transform = tv.transforms.PILToTensor()
            image = transform(image)
            images_array.append(image)
        
        return images_array
    
    def load_training_labels(self, path=""):
        cwd = os.path.dirname(os.path.realpath(__file__))
        train_labels_path = os.path.join(cwd, "data", "train.csv")
        train_labels_df = pd.read_csv(train_labels_path)
        names = train_labels_df['Category'].values.tolist()
        
        mapping = self.load_categories()
        names = torch.Tensor([mapping[n] for n in names]).type(torch.int64)
        names = torch.nn.functional.one_hot(names, num_classes=100)
        names = names.type(torch.float32)
        return names
    
    def load_categories(self, path=""):
        cwd = os.path.dirname(os.path.realpath(__file__))
        labels_path = os.path.join(cwd, "data", "category.csv")
        labels_df = pd.read_csv(labels_path)
        labels = labels_df['Category'].values.tolist()

        category_mapping = {}

        for i, name in enumerate(labels):
            category_mapping[name] = i
        
        return category_mapping

    def load_categories_inverse_map(self, path=""):
        cwd = os.path.dirname(os.path.realpath(__file__))
        labels_path = os.path.join(cwd, "data", "category.csv")
        labels_df = pd.read_csv(labels_path)
        labels = labels_df['Category'].values.tolist()

        category_mapping = {}

        for i, name in enumerate(labels):
            category_mapping[i] = name
        
        return category_mapping
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        image = self.load_image(idx)
        target = self.img_labels[idx]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        
        return image, target


class TestImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = self.load_training_labels()
        self.transform = transform
        self.labels_n_to_i_dic = {}
        self.labels_i_to_n_dic = {}
        self.target_transform = target_transform
    
    # 1. we got to load the images from .jpg to PIL image or np.array
    def load_image(self, index):
        image_path = os.path.join(self.img_dir, f"{str(index)}.jpg")

        image = Image.open(image_path)
        image = image.convert('RGB')
        

        transform = tv.transforms.PILToTensor()
        image = transform(image)

        return image
    
    def load_training_labels(self, path=""):
        cwd = os.path.dirname(os.path.realpath(__file__))
        train_labels_path = os.path.join(cwd, "data", "train.csv")
        train_labels_df = pd.read_csv(train_labels_path)
        names = train_labels_df['Category'].values.tolist()
        
        mapping = self.load_categories()
        names = torch.Tensor([mapping[n] for n in names]).type(torch.int64)
        names = torch.nn.functional.one_hot(names, num_classes=100)
        names = names.type(torch.float32)
        return names
    
    def load_categories(self, path=""):
        cwd = os.path.dirname(os.path.realpath(__file__))
        labels_path = os.path.join(cwd, "data", "category.csv")
        labels_df = pd.read_csv(labels_path)
        labels = labels_df['Category'].values.tolist()

        category_mapping = {}

        for i, name in enumerate(labels):
            category_mapping[name] = i
        
        return category_mapping
    
    def __len__(self):
        return len(os.listdir(self.img_dir))
    
    def __getitem__(self, idx):
        image = self.load_image(idx)
        
        if self.transform:
            image = self.transform(image)
        
        return image
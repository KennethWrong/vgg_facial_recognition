import os
import torch
import torchvision as tv
from torchvision.io import read_image
from PIL import Image, ImageFile
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = self.load_training_labels()
        self.transform = transform
        self.labels_n_to_i_dic = {}
        self.labels_i_to_n_dic = {}
        self.target_transform = target_transform
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. we got to load the images from .jpg to PIL image or np.array
    def load_image(self, index):
        image_path = os.path.join(self.img_dir, f"{str(index)}.jpg")

        # print(f"Image index: {index}")
        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(image_path)
        # image.point(lambda p: p*0.0039063096, mode='RGB')
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
        names = [mapping[n] for n in names]
        return names
    
    def load_categories(self, path=""):
        cwd = os.path.dirname(os.path.realpath(__file__))
        labels_path = os.path.join(cwd, "data", "category.csv")
        labels_df = pd.read_csv(labels_path)
        labels = labels_df['Category'].values.tolist()

        category_mapping = {}

        for i, name in enumerate(labels):
            category_mapping[name] = i
            # self.labels_i_to_n_dic[i] = name
        
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
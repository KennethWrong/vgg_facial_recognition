import os
from torchvision.io import read_image
from PIL import Image
import glob
import re
import pandas as pd


cwd = os.path.dirname(os.path.realpath(__file__))
train_labels_path = os.path.join(cwd, "data", "train.csv")
train_labels_df = pd.read_csv(train_labels_path)

cwd = os.path.dirname(os.path.realpath(__file__))
labels_path = os.path.join(cwd, "data", "category.csv")
labels_df = pd.read_csv(labels_path)

labels = labels_df['Category'].values.tolist()

labels_name_to_index = {}
labels_index_to_name = {}

for i, name in enumerate(labels):
    labels_name_to_index[name] = i
    labels_index_to_name[i] = name

names = train_labels_df['Category'].values.tolist()
names = [labels_name_to_index[n] for n in names]

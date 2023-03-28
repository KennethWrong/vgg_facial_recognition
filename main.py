from model import model_manager
import pandas as pd
from torchsummary import summary
from torch.utils.data import DataLoader
import torch
import torchvision as tv
import os
from dataset import ImageDataset, TestImageDataset
import matplotlib.pyplot as plt
import engine
import datetime

cwd = os.path.dirname(os.path.realpath(__file__))

# Setting up the model
model_class = model_manager()
model = model_class.model

#Loading pretrianed weights ONLY UNCOMMENT IF YOU HAVE WEIGHTS
# weight_path = os.path.join(cwd, "weights", "resnet18-weights-28-03-2023-00-07")
# model.load_state_dict(torch.load(weight_path, map_location="cuda"))

model = model_class.model.to(model_class.device)
# summary(model, (3, 224, 224))
# print(model)

train_data_path = os.path.join(cwd, "data", "train")
test_data_path = os.path.join(cwd, "data", "test")

transform = tv.models.ResNet18_Weights.IMAGENET1K_V1.transforms()

dataset = ImageDataset(train_data_path, transform=transform)

generator1 = torch.Generator().manual_seed(42)
training_dataset, testing_dataset = torch.utils.data.random_split(dataset, [0.7, 0.3], generator1)

training_loader = DataLoader(training_dataset, batch_size=16, shuffle=False)
testing_loader = DataLoader(testing_dataset, batch_size=16, shuffle=False)

# Function to help save our model
def save_model():
    today = datetime.datetime.now()
    filename = today.strftime("%d-%m-%Y-%H-%M")

    filename = "resnet18-weights"+"-"+filename

    weight_path = os.path.join(cwd, "weights", filename)

    torch.save(model.state_dict(), weight_path)

# This is for training
results = engine.train(
    model = model,
    train_dataloader= training_loader,
    test_dataloader= testing_loader,
    optimizer= model_class.optimizer,
    loss_fn= model_class.criterion,
    epochs = 10,
    device= model_class.device,
    model_class=model_class
)

# This is for testing on test set with no label for submission
true_test_dataset = TestImageDataset(test_data_path, transform=transform)
true_test_loader = DataLoader(true_test_dataset, shuffle=False)

res = engine.evaluate_step(
    model=model,
    dataloader=true_test_loader,
    device= model_class.device 
)

# Save results as results.csv for submission
i_to_name_map = dataset.load_categories_inverse_map()

res = [i_to_name_map[r] for r in res]

df = pd.DataFrame(res, columns=["category"])
df.to_csv(os.path.join(cwd, "data", "results.csv"))
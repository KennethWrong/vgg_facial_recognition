from model import vgg_model
from torchsummary import summary
from torch.utils.data import DataLoader
import torch
import torchvision as tv
import os
from dataset import ImageDataset
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import engine

cwd = os.path.dirname(os.path.realpath(__file__))

training_model = vgg_model()
vgg = training_model.vgg16.to(training_model.device)
# summary(vgg, (3, 224, 224))

train_data_path = os.path.join(cwd, "data", "train")
test_data_path = os.path.join(cwd, "data", "test")

transform = tv.models.VGG16_Weights.IMAGENET1K_V1.transforms()

dataset = ImageDataset(train_data_path, transform=transform)

generator1 = torch.Generator().manual_seed(42)
training_dataset, testing_dataset = torch.utils.data.random_split(dataset, [0.7, 0.3], generator1)

training_loader = DataLoader(training_dataset, batch_size=1, shuffle=False)
testing_loader = DataLoader(testing_dataset, batch_size=1, shuffle=False)

from timeit import default_timer as timer
start_time = timer()

results = engine.train(
    model = vgg,
    train_dataloader= training_loader,
    test_dataloader= testing_loader,
    optimizer= training_model.optimizer,
    loss_fn= training_model.criterion,
    epochs = 1,
    device= training_model.device
)

end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")




# test_image, test_label = testing_dataset.__getitem__(6)
# test_image = test_image.permute(1, 2, 0)
# train_image, train_label = training_dataset.__getitem__(6)
# train_image = train_image.permute(1, 2, 0)
# plt.figure()
# f, axarr = plt.subplots(2, 1)
# axarr[0].imshow(test_image)
# axarr[0].set_title(label=test_label)
# axarr[1].imshow(train_image)
# axarr[1].set_title(label=train_label)
# plt.show()


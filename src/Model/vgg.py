from torchvision import models
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Seting up the models
vgg = models.vgg16(pretrained = True)
vgg = vgg.to(device)
vgg.eval()
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from src.Model.vgg import vgg
from src.Model.utils import Hook, device
from torchvision import transforms
import argparse
import requests
from io import BytesIO

parser = argparse.ArgumentParser()
parser.add_argument(
    "-iu", "--image_url",
    default=None,
    type=str,
    required=False
)
parser.add_argument(
    "-ip", "--image_path",
    default=None,
    type=str,
    required=False
)
args = parser.parse_args()

# Make gradients calculations from the output channels of the target layer
def get_gradients(net_in, net, layer):
    net_in = net_in.unsqueeze(0).to(device)
    net_in.requires_grad = True
    net.zero_grad()
    hook = Hook(layer)
    net_out = net(net_in)
    loss = hook.output[0].norm()
    loss.backward()
    return net_in.grad.data.squeeze()


# Denormalization image transform
denorm = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                             transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
                             ])


# Run the Google Deep Dream.
def dream(image, net, layer, iterations, lr):
    image_tensor = transforms.ToTensor()(image)
    image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor).to(device)
    for i in range(iterations):
        gradients = get_gradients(image_tensor, net, layer)
        image_tensor.data = image_tensor.data + lr * gradients.data

    img_out = image_tensor.detach().cpu()
    img_out = denorm(img_out)
    img_out_np = img_out.numpy().transpose(1, 2, 0)
    img_out_np = np.clip(img_out_np, 0, 1)
    img_out_pil = Image.fromarray(np.uint8(img_out_np * 255))
    return img_out_pil


# Doing prediction on image and displaying it

# Passing arguments
if args.image_url:
  url = args.image_url
  response = requests.get(url)
  img = Image.open(BytesIO(response.content))
else:
  img = Image.open(
      args.image_path
  )

orig_size = np.array(img.size)
new_size = np.array(img.size) * 0.5

img = img.resize(new_size.astype(int))
layer = list(vgg.features.modules())[27]

# Execute our Deep Dream Function
img = dream(img, vgg, layer, 20, 1)

img = img.resize(orig_size)
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.show()

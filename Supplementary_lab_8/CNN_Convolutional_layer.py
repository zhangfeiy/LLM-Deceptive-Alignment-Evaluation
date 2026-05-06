import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import urllib.request
import matplotlib.pyplot as plt

# 1. Load pre-trained ResNet18 model and set to evaluation mode
print("Loading pre-trained ResNet18...")
model = models.resnet18(pretrained=True)
model.eval()

# We will visualize the feature maps from the first convolutional layer (conv1) of ResNet18, which has 64 filters.
first_conv_layer = model.conv1

# 2. Get a real test image (here we use an open-source dog image from the web for testing)
print("Downloading test image...")
url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
image_path = "test_image.jpg"
urllib.request.urlretrieve(url, image_path)

img = Image.open(image_path)

# 3. Preprocessing the image (resize, crop, convert to tensor, and normalize)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0) # Add a batch dimension

print("Extracting intermediate feature maps...")
# 4. Extract feature maps: pass the image through the first convolutional layer
with torch.no_grad():
    feature_maps = first_conv_layer(input_batch)

# 5. Output the feature maps of the first convolutional layer
print("Generating visualization plot...")
feature_maps = feature_maps.squeeze(0).cpu() # Remove batch dimension and move to CPU for plotting

# Create a 4x4 grid to display the feature maps extracted by the first 16 filters
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle("CNN Feature Map Visualization (First Conv Layer)", fontsize=16)

for i, ax in enumerate(axes.flatten()):
    if i < feature_maps.size(0):
        # Extract individual feature map and convert to numpy array for plotting
        fmap = feature_maps[i].numpy()
        ax.imshow(fmap, cmap='viridis')
        ax.axis('off')
        ax.set_title(f"Filter {i+1}")

plt.tight_layout()
plt.show() 
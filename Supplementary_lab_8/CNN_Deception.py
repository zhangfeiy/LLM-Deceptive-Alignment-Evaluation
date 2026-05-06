import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import urllib.request
import json
import matplotlib.pyplot as plt

# 4. References & Acknowledgments
#The adversarial attack demonstrated in `CNN_Deception.py` implements the Fast Gradient Sign Method (FGSM). This algorithm was originally introduced in the following foundational paper:

# Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples. *In International Conference on Learning Representations (ICLR)*. arXiv preprint arXiv:1412.6572.
print("--- Initializing CNN Deception (Adversarial Attack) Demo ---")

# 1. Load pre-trained ResNet18 model and set to evaluation mode
model = models.resnet18(pretrained=True)
model.eval() 

# the next line is very important, Usually, we use this during inference to turn off Dropout and Batch Normalization. 
# But in an adversarial attack, it is mandatory. We need a deterministic model to calculate precise gradients. 
# Any random noise from Dropout will destroy our goal.


# 2. Get ImageNet class labels (for better visualization of predictions)
# this part is just for utility functions. I download the 1,000 ImageNet class labels 
# so we can see human readable predictions instead of just raw index numbers. 
# Then, I download a standard test image of a dog from the GitHub

url_labels = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels_path = "imagenet_labels.json"
urllib.request.urlretrieve(url_labels, labels_path) # Download the labels file from the web and save it locally
with open(labels_path) as f:
    imagenet_labels = json.load(f) 
# Just use the json dictionary to load the labers so that we can easily map the predicted class index to a readable label.

# 3. Download a test image (a dog image from the web)
url_img = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
image_path = "test_dog.jpg"
urllib.request.urlretrieve(url_img, image_path)

# 4. image preprocessing (resize, crop, convert to tensor)
# I define a standard preprocessing pipeline to resize and center crop the image to 224 by 224, and convert it to a Tensor.
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
# The CNN requires the same size and format of input images as it was trained on. 
# So we need to resize and crop the image to 224x224 pixels, and convert it to a tensor format that the model can process.


# The unsqueeze(0) is to add a batch dimension, since the model expects input in the shape of (batch_size, channels, height, width).
original_tensor = preprocess(Image.open(image_path)).unsqueeze(0)
original_tensor.requires_grad = True 

# In normal deep learning, we calculate gradients for the model's weights. 
# But here, we freeze the weights and ask PyTorch to track the gradients of the input. 
# We are going to train the image, not the model.

# 5. First Phase: Normal Prediction (Baseline)
# For presentation, we will show the original prediction before the attack, so we apply normalization here as well.

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# the mean and std values are the normalization parameters used during the training of the model on ImageNet.
output_normal = model(normalize(original_tensor))
init_pred_index = output_normal.max(1, keepdim=True)[1].item() #input the highest score's index as the predicted class index
init_pred_label = imagenet_labels[init_pred_index]
print(f"Original Prediction: {init_pred_label}")

# Here is where the actual deception happens!

# 6. Second Phase: Creating Deception (FGSM（fast gradient sign method）Attack)

criterion = nn.CrossEntropyLoss() 
target_label = torch.tensor([init_pred_index])
loss = criterion(output_normal, target_label)

# we use CrossEntropyLoss functions because it's the standard loss function for classification tasks
# calculate the error between the model's output and the true label.

model.zero_grad() # zeor output the gradients of the model parameters
loss.backward() #  send them all the way back to the image pixels

data_grad = original_tensor.grad.data # this is the gradient of the loss with respect to the input image.
epsilon = 0.05 

# First, let we see the formula for the FGSM attack: perturbed_image = original_image + epsilon * sign(gradient)
# Normally, in gradient descent, we subtract the gradient to minimize the loss. 
# But we are attacking the model, so we add the gradient to maximize the error
# Using the .sign() because we don't care about the size of the gradient, we just want to make the model fail fastest
# This is the key parameter that controls how much we perturb the image. Even a small epsilon can cause misclassification!

perturbed_tensor = original_tensor + epsilon * data_grad.sign()
perturbed_tensor = torch.clamp(perturbed_tensor, 0, 1) 

# This is a necessary engineering step. 
# Adding noise might push pixel values beyond the valid 0 to 1 range, so clamp acts as a safety limit to prevent errors

# 7. Third Phase: Testing the Deceived CNN
output_deceived = model(normalize(perturbed_tensor))
deceived_pred_index = output_deceived.max(1, keepdim=True)[1].item()
deceived_pred_label = imagenet_labels[deceived_pred_index]
print(f"Deceived Prediction: {deceived_pred_label}")

# I pass this newly poisoned image (the perturbed_tensor) back into the model
# And we take the same process to normalize it and get the new prediction

# 8. Visualization of the original image
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("CNN Vulnerability: Adversarial Deception", fontsize=16)

# we use .squeeze() to remove the batch dimension, .detach() to get a tensor that is not part of the computation graph
# And .permute(1, 2, 0) to change the shape from (channels, height, width) to (height, width, channels) for visualization.
ax1.imshow(original_tensor.squeeze().detach().permute(1, 2, 0).numpy())
ax1.set_title(f"Original\nPredict: {init_pred_label}")
ax1.axis('off') # this is just to turn off the axis for better visualization


# By visualizing this noise, we can see how even small, seemingly imperceptible changes can lead to a completely different prediction by the CNN.
noise = (data_grad.sign().squeeze().detach().permute(1, 2, 0).numpy() + 1) / 2
ax2.imshow(noise)
ax2.set_title("Adversarial Perturbation\n(Gradient Sign)")
ax2.axis('off')

# The same process to visualize the perturbed image
ax3.imshow(perturbed_tensor.squeeze().detach().permute(1, 2, 0).numpy())
ax3.set_title(f"Deceived Image\nPredict: {deceived_pred_label}", color='red')
ax3.axis('off')

plt.tight_layout()
plt.show()
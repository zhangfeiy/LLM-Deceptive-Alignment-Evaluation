# Lab 8 Submission: CNN Interpretability and Architectural Contrast

**Project Type:** Review Paper 
**Review Topic:** Preference Optimization (RLHF/DPO) and Deception in Large Language Models (LLMs)

## 1. Connection to Project: A Contrastive Analysis
My review paper focuses on the alignment of Large Language Models (LLMs) using techniques like RLHF and DPO. In this specific domain, Convolutional Neural Networks (CNNs) are not typically utilized. CNNs rely on convolutional kernels to extract local spatial hierarchies from grid-like data (e.g., pixels). In contrast, LLM alignment requires processing 1D sequential tokens with long-range dependencies, making Transformer architectures (specifically causal self-attention) the standard.

However, a major challenge highlighted in my review is the "black-box" nature of LLMs, making it difficult to detect or interpret deceptive behaviors during optimization. For this lab, I chose to explore CNNs through the lens of **Interpretability**. By understanding how CNNs visually "white-box" their feature extraction process, we can establish a comparative baseline for the challenges of interpreting sequential text models.

## 2. Code Demo Instructions
The included script `CNN_Convolutional_layer.py` is a conceptual proof-of-concept designed for the in-class demo. It passes an image through a pre-trained CNN (ResNet18) and visualizes the intermediate feature maps of the first convolutional layer using `matplotlib`. 

**Requirements:**
```bash
pip install torch torchvision matplotlib pillow

How to run: 
python CNN_Convolutional_layer.py

Simply execute the script. It will automatically download a sample image, process it through the CNN, and display a 4x4 grid of the extracted spatial features.

3. GitHub Code Resources & Relevance
To further explore the contrast between spatial CNNs and sequential alignment, I reviewed the following three repositories:

Resource 1: utkuozbulak/pytorch-cnn-visualizations

GitHub Link: https://github.com/utkuozbulak/pytorch-cnn-visualizations

What it does: This repository implements various techniques to visualize CNNs in PyTorch, such as Gradient-weighted Class Activation Mapping (Grad-CAM) and layer-specific feature visualization. It reveals exactly what visual patterns trigger specific network activations.

Why it is relevant: This repo perfectly illustrates the comparative advantage of CNNs in spatial interpretability. Reviewing how visual features can be mapped back to input pixels highlights the lack of equivalent, straightforward interpretability tools for detecting "reward hacking" or deception in my RLHF review project.

Resource 2: DLR-RM/stable-baselines3

GitHub Link: https://github.com/DLR-RM/stable-baselines3

What it does: This is a set of reliable PyTorch implementations of Reinforcement Learning (RL) algorithms. It extensively uses CNN architectures as policy networks for environments with visual, grid-based state spaces (like Atari games).

Why it is relevant: My review paper studies RL applied to text (RLHF). This repository demonstrates the parallel universe of RL applied to pixels. It provides code-level evidence of how CNNs replace Transformers as the backbone of the policy optimization process when the environment shifts from sequential language to spatial grids.

Resource 3: lucidrains/vit-pytorch

GitHub Link: https://github.com/lucidrains/vit-pytorch

What it does: This provides a simple and clean implementation of the Vision Transformer (ViT), a model that applies the Transformer architecture (originally designed for NLP) directly to image classification, often outperforming traditional CNNs.

Why it is relevant: This architectural implementation proves the versatility of the self-attention mechanisms that my review paper focuses on. It helps explain the ongoing architectural shift and why traditional CNNs are being challenged by the very architectures used in LLM alignment, rounding out the comparative analysis of my review.
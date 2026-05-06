# LLM Alignment Evaluation: A Binary Classification Approach

## Project Overview (Lab 9 Reconstruction)
This repository contains the reconstructed empirical evaluation codebase for Lab 9. To rigorously evaluate the deceptive alignment and reward hacking phenomena in Large Language Models (specifically comparing RLHF vs. DPO behavior), we have reframed the evaluation as a strict **Binary Classification Task** utilizing the TruthfulQA scenario from the HELM benchmark (NeurIPS 2022).

Instead of relying on qualitative, unstructured text generation which is prone to arbitrary length and phrasing variations, this implementation extracts raw next-token logits from the models. This provides exact probabilities for choice selection and enables deterministic machine learning evaluations.

## Methodology: Logit-Based Extraction
Given a prompt $x$ and two candidate choices (A and B), the model's preference is computed by comparing the unnormalized log probabilities (logits) of the specific tokens representing 'A' and 'B'. The hard classification rule is defined as:

$$\hat{y} = \arg\max_{c \in \{A, B\}} \text{Logit}(c | x)$$

To compute probabilistic metrics (like ROC-AUC), we convert the logits into a valid probability distribution using the Softmax function:

$$P(A|x) = \frac{\exp(\text{Logit}_A)}{\exp(\text{Logit}_A) + \exp(\text{Logit}_B)}$$

This method isolates the model's intrinsic alignment representation from generation artifacts.

## Hardware & Optimization
Due to local compute constraints, this pipeline is optimized for **Google Colab (T4 GPU)**. 
- **Quantization:** Implements `bitsandbytes` 4-bit NF4 (NormalFloat4) quantization to fit 7B parameter models within a 16GB VRAM footprint.
- **Compute Dtype:** Utilizes `bfloat16` for memory-efficient forward passes.

## 🚀 How to Run on Google Colab (Reproducibility Guide)

Due to the VRAM requirements for loading 7B parameter LLMs, this project is explicitly optimized to run on a **Google Colab environment with a free T4 GPU**. This avoids local hardware bottlenecks while maintaining rigorous ML evaluation standards.

Please follow these exact steps to reproduce the evaluation metrics:

### Step 1: Set Up the Environment
1. Open a new [Google Colab](https://colab.research.google.com/) notebook.
2. Upload the project files (`main.py`, `evaluate.py`, and `requirements.txt`) directly to the Colab session storage (the folder icon on the left sidebar).
3. **CRITICAL:** Enable the GPU by navigating to `Runtime` -> `Change runtime type` -> Select **T4 GPU** -> click `Save`.

### Step 2: Install Strict Dependencies
In the first notebook cell, run the following command. This installs the specific versions required to support 4-bit NF4 quantization and resolves known `bitsandbytes` environment errors:

```bash
!pip install -r requirements.txt

!python main.py

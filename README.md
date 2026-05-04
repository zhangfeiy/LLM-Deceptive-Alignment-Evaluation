# Empirical Evaluation of LLM Deceptive Alignment via Logit Based Classification

This repository contains the official implementation and evaluation pipeline for our Machine Learning Final Project: **Rethinking Preference Optimization: A Systematic Review and Empirical Classification Analysis of LLM Deceptive Alignment**.

**Team Members:** Zhangfei Yang

## Project Overview
Recent evidence suggests that preference based alignment methods (like RLHF and DPO) may inadvertently induce deceptive alignment—where Large Language Models prioritize sycophancy or surface level helpfulness over factual truth to appease perceived user personas. 

Instead of relying on flawed generative "LLM as a judge" evaluations, this project reframes alignment evaluation as a **strict deterministic binary classification task**. We extract next token logits from a 4-bit quantized 7B model (Zephyr-7b-beta) on the TruthfulQA benchmark to explicitly measure the "alignment tax" via core machine learning metrics, isolating the False Positive Rate (FPR) as a quantitative proxy for reward hacking.

## Environment Setup & Hardware Requirements
This pipeline is optimized for constrained hardware environments, specifically a **Google Colab T4 GPU** (16GB VRAM), utilizing 4-bit NormalFloat (NF4) quantization.

**Dependencies:**
- Python 3.10+
- PyTorch >= 2.0.0
- Transformers >= 4.38.0
- BitsAndBytes >= 0.46.1

**Installation:**
To replicate the environment, run:
```bash
pip install -r requirements.txt
```

**Run the evaluation:**
```bash
python main.py
```

**What the script does:**
1. Loads the HuggingFaceH4/zephyr-7b-beta model using bitsandbytes 4-bit quantization.
2. Fetches the validation split of the TruthfulQA dataset from Hugging Face.
3. Dynamically randomizes the A/B option assignment to strictly eliminate positional token bias.
4. Uses evaluate.py to bypass text generation and extract direct next token logits for choices A and B.
5. Computes classification accuracy, False Positive Rate (sycophancy score), and saves the artifact to a CSV file.

**Repository Structure**
1. main.py: The primary execution script handling model loading, dataset parsing, dynamic A/B randomization, and metrics computation.
2. evaluate.py: Contains the core get_binary_prediction function, which performs the critical mathematical extraction of raw logit values (logit_a and logit_b) to determine the prediction deterministically.
3. requirements.txt: List of all necessary Python packages and specific versions.
4. final_paper.pdf: The complete AAAI formatted paper detailing our theoretical synthesis and empirical findings.
5. Supplymentry files (CNN_Convolutional_layer.py / CNN_Deception.py / imagenet_labels.json / test_dog.jpg) for lab 8: since this project is very hard to connect with CNN, so I set an adversarial attack to CNN convolutional layer for a simulation of deceptive behavior which had presented in lab 8 presentation.


# Empirical Evaluation of LLM Deceptive Alignment via Logit-Based Classification

This repository contains the official implementation and evaluation pipeline for our Machine Learning Final Project: **Rethinking Preference Optimization: A Systematic Review and Empirical Classification Analysis of LLM Deceptive Alignment**.

**Team Members:** Zhangfei Yang

## 📌 Project Overview
Recent evidence suggests that preference-based alignment methods (like RLHF and DPO) may inadvertently induce deceptive alignment—where Large Language Models prioritize sycophancy or surface-level helpfulness over factual truth to appease perceived user personas. 

Instead of relying on flawed generative "LLM-as-a-judge" evaluations, this project reframes alignment evaluation as a **strict deterministic binary classification task**. We extract next-token logits from a 4-bit quantized 7B model (Zephyr-7b-beta) on the TruthfulQA benchmark to explicitly measure the "alignment tax" via core machine learning metrics, isolating the False Positive Rate (FPR) as a quantitative proxy for reward hacking.

## ⚙️ Environment Setup & Hardware Requirements
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

# Measuring Sycophancy in Preference-Aligned Large Language Models

This repository contains the paper, review materials, and supplementary code for a course-level machine learning review project at GWU.

The final paper is a review paper, not a new benchmark paper. It studies how recent work evaluates sycophancy and related failure modes in preference-aligned large language models.

## Paper Title

**Measuring Sycophancy in Preference-Aligned Large Language Models: A Review of Prompt Designs, Metrics, and Evaluation Gaps**

## Project Overview

Preference-aligned LLMs are often evaluated by helpfulness, truthfulness, reward score, or judge preference. However, these metrics do not always measure the same failure.

This project focuses on a specific measurement problem:

> Many papers use the term "sycophancy," but they often measure different behaviors.

For example:

- **Agreement rate** can detect direct agreement with a user's false belief.
- **Truthfulness accuracy** can detect false answers, but may miss whether the model corrected the user's false premise.
- **LLM-as-a-judge scores** may reward fluent or polite responses even when the model avoids clear correction.
- **Reward-model scores** are useful for reward hacking analysis, but they do not directly show conversational sycophancy.

The review therefore organizes prior work by:

1. behavior type,
2. prompt design,
3. evaluation format,
4. metric type.

The main argument is that sycophancy evaluation often suffers from **prompt--metric mismatch**. A metric that works for direct false agreement may fail for hedged correction, false-premise acceptance, or reward-driven accommodation.

## Repository Structure

```text
.
├── Paper/
│   ├── final_paper.pdf
│   ├── main.tex
│   └── references.bib
├── review_materials/
│   ├── search_log.csv
│   ├── coding_scheme.md
│   └── included_studies_coding.csv
├── extract_logit_lab_9/
│   ├── main.py
│   ├── evaluate.py
│   ├── requirements.txt
│   └── README.md
├── Supplementary_lab_8/
│   ├── CNN_Convolutional_layer.py
│   ├── CNN_Deception.py
│   ├── imagenet_labels.json
│   ├── test_dog.jpg
│   └── README.md
├── Response_to_Lab10_Feedback.pdf
└── README.md
This review is PRISMA-guided but not a statistical meta-analysis. The included studies use different models, datasets, prompts, and metrics, so numerical results are not pooled across papers.

The main reproducibility goal of this repository is to make the paper source, review materials, and supplementary diagnostic code available in an organized form.

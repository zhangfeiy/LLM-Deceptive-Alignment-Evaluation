# Measuring Sycophancy in Preference Aligned Large Language Models

This repository contains the paper, review materials, and supplementary code for a course level machine learning review project at GWU.

The final paper is a review paper, not a new benchmark paper. It studies how recent work evaluates sycophancy and related failure modes in preference aligned large language models.

## Paper Title

**Measuring Sycophancy in Preference Aligned Large Language Models: A Review of Prompt Designs, Metrics, and Evaluation Gaps**

## Project Overview

Preference aligned LLMs are often evaluated by helpfulness, truthfulness, reward score, or judge preference. However, these metrics do not always measure the same failure.

This project focuses on a specific measurement problem:

> Many papers use the term "sycophancy," but they often measure different behaviors.

For example:

- **Agreement rate** can detect direct agreement with a user's false belief.
- **Truthfulness accuracy** can detect false answers, but may miss whether the model corrected the user's false premise.
- **LLM as a judge scores** may reward fluent or polite responses even when the model avoids clear correction.
- **Reward model scores** are useful for reward hacking analysis, but they do not directly show conversational sycophancy.

The review therefore organizes prior work by:

1. behavior type,
2. prompt design,
3. evaluation format,
4. metric type.

The main argument is that sycophancy evaluation often suffers from **prompt and metric mismatch**. A metric that works for direct false agreement may fail for hedged correction, false premise acceptance, or reward driven accommodation.

## Repository Structure

```text
.
├── paper/
│   ├── final_paper.pdf
│   ├── main.tex
│   └── references.bib
├── review_materials/
│   ├── included_studies_coding.csv
│   ├── search_log.csv
│   ├── excluded_full_text_records.csv
│   └── coding_scheme.md
├── extract_logit_lab_9/
│   ├── main.py
│   ├── evaluate.py
│   ├── requirements.txt
│   └── README.md
└── Supplementary_lab_8/
    ├── CNN_Convolutional_layer.py
    ├── CNN_Deception.py
    ├── imagenet_labels.json
    └── test_dog.jpg
    └── README.md

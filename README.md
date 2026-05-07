# Measuring Sycophancy in Preference Aligned Large Language Models

This repository contains the paper, review materials, and supplementary code for a course level machine learning Lab experiments at GWU.

The final paper is a review paper, not a new benchmark paper. It studies how recent work evaluates sycophancy and related failure modes in preference aligned large language models.

## Paper Title

**Measuring Sycophancy in Preference Aligned Large Language Models: A Review of Prompt Designs, Metrics, and Evaluation Gaps**

## Project Overview

Preference aligned LLMs are often evaluated by helpfulness, truthfulness, reward score, or judge preference. However, these metrics do not always measure the same failure.

This project focuses on a specific measurement problem:

> Many papers use the term "sycophancy" but they often measure different behaviors.

For example:

- **Agreement rate** can detect direct agreement with a user's false belief.
- **Truthfulness accuracy** can detect false answers, but may miss whether the model corrected the user's false premise.
- **LLM-as-a-judge scores** may reward fluent or polite responses even when the model avoids clear correction.
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
```
# Paper Folder

The Paper/ folder contains the final review paper and related writing materials.
	**final_paper.pdf**: final submitted paper
	**main.tex**: LaTeX source
	**references.bib**: bibliography file

# Review Materials

The review_materials/ folder contains supplementary materials for the PRISMA guided review process described in the paper.
**search_log.csv** search sources, representative queries, and retrieval notes
**coding_scheme.md**: definitions of coding dimensions used in the review
**included_studies_coding.csv**: coding sheet for the studies used in the taxonomy and distribution tables

The included_studies_coding.csv file is intended to correspond to the studies discussed in the paper. Because several papers report multiple evaluation formats or metrics, some coding categories are not mutually exclusive.

# Supplementary Logit Based Diagnostic

The extract_logit_lab_9/ folder contains a small exploratory diagnostic originally developed during Lab 9.

This diagnostic is not the main contribution of the final paper. It is included as supplementary code to illustrate one controlled way to test misleading option selection under user pressure.

The diagnostic uses a multiple choice TruthfulQA style format. Each prompt contains:

**one truthful option**
**one misleading option**
**a short user context prefix that creates pressure toward the misleading answer**

Instead of generating full responses, the diagnostic extracts next token logits for option tokens and treats the model as a binary classifier. The main diagnostic metric is False Positive Rate (FPR), defined as the proportion of cases where the model selects the misleading option.

This pilot should not be interpreted as a full benchmark or as proof that one alignment method is generally safer than another.

# Supplementary Lab 8 Files

The Supplementary_lab_8/ folder contains earlier course demonstration files related to CNN behavior and adversarial style analysis. These files were part of a separate lab presentation and are not part of the final review paper.

# Reproducibility Notes

This review is PRISMA guided but not a statistical meta analysis. The included studies use different models, datasets, prompts, and metrics, so numerical results are not pooled across papers.

The main reproducibility goal of this repository is to make the paper source, review materials, and supplementary diagnostic code available in an organized form.


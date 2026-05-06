```markdown
# Coding Scheme for the Review Paper

This file describes the coding scheme used for the review paper:

**Measuring Sycophancy in Preference Aligned Large Language Models: A Review of Prompt Designs, Metrics, and Evaluation Gaps**

The purpose of the coding was to make the review taxonomy more transparent. The coding categories are not mutually exclusive. A single paper may receive more than one label if it uses multiple evaluation formats, prompt types, or metrics.

## Coding Dimensions

### 1. Primary Category

The main role of the paper in the review.

Examples:

- Preference alignment
- Sycophancy behavior
- Sycophancy mitigation
- Truthfulness benchmark
- Evaluation framework
- Judge-model evaluation
- Reward overoptimization
- Reward gaming
- AI deception
- Red teaming

### 2. Alignment Context

The alignment method or evaluation context discussed in the paper.

Examples:

- RLHF
- DPO
- RLAIF
- Constitutional AI
- Reward modeling
- Benchmark-only
- LLM-as-a-judge
- Red teaming
- Theoretical analysis

### 3. Behavior Measured

The behavior or failure mode targeted by the paper.

Examples:

- Opinion agreement
- False-premise acceptance
- Hedged or weak correction
- Reward-seeking behavior
- Truthfulness failure
- Proxy-objective failure
- Persistent deceptive behavior
- Evaluation bias

### 4. Prompt Format

The type of prompt or task format used in the study.

Examples:

- User-opinion prompt
- False-premise prompt
- Preference-pressure prompt
- Instruction-following prompt
- Multiple-choice conflict prompt
- Free-form conversation
- Benchmark scenario prompt
- Red-teaming prompt
- Not prompt-based

### 5. Evaluation Format

The form of evaluation used in the study.

Examples:

- Free-form generation evaluation
- Multiple-choice or binary-choice evaluation
- Human evaluation / manual annotation
- LLM-as-a-judge evaluation
- Reward-model scoring or reward analysis
- Benchmark-only evaluation
- Red-teaming / adversarial prompting
- Logit or probability-level analysis
- Theoretical analysis

### 6. Metric Type

The main metric or evidence type used in the paper.

Examples:

- Accuracy / truthfulness score
- Agreement rate or agreement classification
- False positive or misleading option selection
- Reward score / reward model analysis
- Win rate / preference score
- LLM as a judge score
- Manual error coding
- Qualitative case analysis
- Theoretical argument

### 7. Role in Review

How the paper is used in the review.

Examples:

- Background on RLHF
- Background on DPO
- Direct sycophancy evidence
- Truthfulness benchmark evidence
- Reward hacking mechanism
- Judge model limitation evidence
- Red team evaluation evidence
- Stronger deception risk evidence

## Coding Principle

The central coding question was:

> Does the metric actually measure the failure behavior claimed by the paper?

This principle supports the paper's main argument about prompt and metric mismatch.

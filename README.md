# Indic Language Counterfactuals

This repository contains code to automatically generate natural language counterfactuals for Indic language sentences. 
## Setup

To set up your conda environment, use `requirements.txt`.

## Usage

### Translating Datasets

To translate the original English datasets, use `translate_notebook.ipynb`. There are two methods available: using IndicTrans or Helsinki NLP's translation model for Indic languages. IndicTrans provides better results but is slower.

### Generating Counterfactuals

To generate counterfactuals, use `sentence_score.ipynb`. This notebook walks you through the process of fluency filtering, masking, finetuning, and some downstream tasks for evaluation.

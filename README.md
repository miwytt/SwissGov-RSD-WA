# SwissGov-RSD: A Human-annotated, Cross-lingual Benchmark for the Recognition of Token-level Semantic Differences Between Related Documents

[![Paper](https://img.shields.io/badge/📄%20Paper-arXiv%3A2512.07538-B31B1B.svg)](https://arxiv.org/pdf/2512.07538)
[![Dataset](https://img.shields.io/badge/🤗-Huggingface%20Dataset-yellow.svg)](https://huggingface.co/datasets/ZurichNLP/SwissGov-RSD)

A comprehensive benchmark for evaluating token-level semantic difference recognition across multiple languages including iSTS-RSD ([Vamvas and Sennrich, 2023](https://aclanthology.org/2023.emnlp-main.835/)) and SwissGov-RSD ([Wastl et al., 2025]([https://aclanthology.org/2023.emnlp-main.835/](https://www.arxiv.org/abs/2512.07538)).

---

## Installation

We recommend using Python 3.11 for this project.

Install the required dependencies in two steps:

**Step 1: Install main requirements**
```bash
pip install -r requirements.txt
```

**Step 2: Install RSD-specific requirements**
```bash
pip install -r rsd/requirements.txt
```

---

## Model Training

### Encoder Models

Before training encoder models:

1. Adjust the data path in `encoders/finetuning/data`
2. Update the `output_dir` name in `encoders/finetuning/run_train_modernbert-large.sh`
3. Update the model path in `scripts/run_predict_encoders.sh`

Then run the training script:

```bash
bash encoders/finetuning/run_train_modernbert-large.sh
```

### Large Language Models (LLMs)

**Llama Models:**

```bash
bash llama/run_train_llama_8b.sh
```

**GPT-4o-mini:**

Training for GPT-4o-mini was performed using the browser application.

---

## Running Predictions

### Encoder-based Models

```bash
bash scripts/run_predict_encoder.sh
```

### Llama Models

```bash
bash scripts/run_predict_llama.sh
```

### OpenAI Models

```bash
python scripts/predict_openai.py --model [openai_modelname]
```

### Llama 405B (Fireworks)

```bash
python scripts/predict_llama_405b_fireworks
```

### DeepSeek R1

```bash
python scripts/predict_deepseekr1.py
```

---

## Evaluation

### SwissGov-RSD Evaluation

Evaluate predictions on the SwissGov-RSD benchmark:

```bash
python -m scripts.evaluate_predictions_admin [prediction_path_prefix]
```

**Example prediction path prefix:**
```
data/evaluation/encoder_predictions/dev/DiffAlign(model=Alibaba-NLP_gte-multilingual-base, layer=-1_admin_
```

**Special Note for DiffAlign with XLM-R + SimCSE:**

When evaluating DiffAlign with a model that can take only relatively short input sequences, e.g. XLM-R + SimCSE for SwissGov, add the `--short` flag:

```bash
python -m scripts.evaluate_predictions_admin [prediction_path_prefix] --short
```

### iSTS-RSD Evaluation

Evaluate predictions on the iSTS-RSD benchmark:

```bash
python -m scripts.evaluate_predictions_ists [path_to_preds]
```

---

## Utilities and Tools

### Dataset Statistics

Get statistics about the test dataset:

```bash
python -m scripts.get_test_data_stats
```

### Label Visualization

**Visualize Gold Labels:**

Display annotations for a specific sample from the gold standard:

```bash
python -m scripts.visualize_labeled_text [path_to_gold] [sample_id] --gold
```

**Example:**
- `sample_id`: `admin_de_1`
- `path_to_gold`: Full path to the file containing gold annotations

**Visualize Predicted Labels:**

Display annotations from model predictions:

```bash
python -m scripts.visualize_labeled_text [path_to_preds] [sample_id]
```

**Visualize iSTS-RSD Sample:**

Create a visualization example from the iSTS-RSD dataset:

```bash
python -m scripts.create_rsd_example
```

### Length Analysis Statistics

Get correlation statistics per bucket for length analysis:

```bash
python -m scripts.get_corr_per_bucket [path_to_preds] --get-corr-per-bucket
```

**Example path:**
```
data/evaluation/encoder_predictions/dev/DiffAlign(model=Alibaba-NLP_gte-multilingual-base, layer=-1_admin_
```

### Label Projection

The following scripts are available for label projection and evaluation:

- `scripts/project_labels.py` - Project labels across languages
- `scripts/merge_projected_data.py` - Merge projected label data
- `scripts/sample_projected_data.py` - Sample from projected data

### Latency Testing

Test model inference latency:

```bash
python -m scripts.latency_test
```

---

## Citation

If you use this library in your research, please cite the accompanying paper:

```bibtex
@misc{wastl2025swissgovrsdhumanannotatedcrosslingualbenchmark,
      title={SwissGov-RSD: A Human-annotated, Cross-lingual Benchmark for Token-level Recognition of Semantic Differences Between Related Documents}, 
      author={Michelle Wastl and Jannis Vamvas and Rico Sennrich},
      year={2025},
      eprint={2512.07538},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.07538}, 
}
```




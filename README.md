# **MMTRC**

**Multi-Modal Temporal Relation Classification**

---

## 📊 1. Dataset Analysis

Run the notebook `dataset_analysis.ipynb` to analyze the structure and contents of the dataset.

This includes:

* Number of samples and steps per instance.
* Distribution of temporal relation labels (e.g., "yes"/"no").
* Optional visual checks and sanity validations.

---

## 🧪 2. Evaluation

To run an evaluation, **modify the `config.py` file to select the experiment you want to perform**.

Example configuration:

```python
MODEL = "llava"  # Options: 'qwen', 'llava'
PATH_DATASET = "dataset/english_N1000_2025-02-11.json"
TYPE_PROMPT = "after"  # Options: 'before', 'after'
RANDOM_ORDER = True
```

Then execute:

```bash
python evaluate.py
```

This will generate a CSV file containing predictions and ground truth labels.

---

### 📈 Analyze Results

To obtain the performance of one or more experiments:

```bash
python analyze_prediction.py
```

Inside the script, specify the CSV files of the experiments you want to analyze.
The script will compute performance metrics such as accuracy and F1 score.

---

## 🤖 Models Used

This project supports the following models:

* **Qwen2.5-VL-7B-Instruct**
  🔗 [See documentation](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

* **LLaVA-OneVision-Qwen2-7B-OV**
  🔗 [See documentation](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov)

Please refer to the official model pages for installation and setup instructions.


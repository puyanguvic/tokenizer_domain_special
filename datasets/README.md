# Dataset

This project uses datasets hosted on Hugging Face.
- HDFS security log datatset: logfit-project/HDFS_v1
- Phish HTML datast: puyang2025/phish_html
- Phishing Email Dataset: puyang2025/seven-phishing-email-datasets
- WAF dataset: puyang2025/waf_data_v2

To download:
```python
from datasets import load_dataset
dataset = load_dataset("your-org/your-dataset")
```

To run CTok experiments on a Hugging Face dataset:
```bash
python run_ctok_experiment.py \
  --dataset logfit-project/HDFS_v1 \
  --split train \
  --text_key text \
  --label_key label \
  --outdir ./ctok_hdfs_demo \
  --vocab_size 8192 \
  --max_len 12 \
  --min_freq 50 \
  --semantic_mode mi \
  --lambda_sem 50.0 \
  --use_ascii_base \
  --emit_code
```

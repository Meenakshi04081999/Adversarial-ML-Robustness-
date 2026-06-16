# Adversarial-ML-Robustness - Task 3

## How to Recreate the Best Leaderboard Result

**Best leaderboard score: 0.555898**
**Method: PGD Adversarial Training | Model: ResNet18 | Epochs: 100**

### 1. Requirements
```bash
python3 -m venv ~/venvs/robust_env
source ~/venvs/robust_env/bin/activate
pip install torch torchvision numpy
```

### 2. Download the dataset
```bash
wget "https://huggingface.co/datasets/SprintML/tml26_task3/resolve/main/train.npz" -O train.npz
```

### 3. Train the model
```bash
python3 experiments/train_v1_pgd_baseline.py
```

### 4. Submit
Edit submission.py with your API key and run:
```bash
python3 submission.py
```

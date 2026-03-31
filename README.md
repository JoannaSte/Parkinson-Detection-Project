# Parkinson Detection from Voice
This project focuses on detecting Parkinson’s disease using voice recordings and deep learning. The model is based on a pretrained WavLM, fine-tuned for classification of pathological vs healthy speech.

Training and experimentation are implemented using PyTorch Lightning, which provides a structured and scalable training pipeline.

## Project Overview
1. Input: Raw audio recordings of speech
2. Model: Fine-tuned WavLM-based architecture
3. Framework: PyTorch Lightning
4. Task: Binary / multi-class classification (Parkinson detection)
5. Data augmentation: Optional (e.g., noise, filtering, time stretching)

## Project Structure
```
.
├── create_dataset.py        # Custom dataset class
├── create_dataloader.py     # Data loading and preprocessing
├── model.py                # Model architectures (WavLM-based)
├── Trainer.py              # LightningModule (Learner)
├── train.py                # Training script
├── config.yaml             # Configuration file
```

## Installation
```bash
git clone https://github.com/your-username/parkinson-detection.git
cd parkinson-detection
pip install -r requirements.txt
```


## Dataset 

The dataset is not included due to privacy and size constraints. The original dataset consists of approximately 100 patients and is balanced in terms of age, sex, and the ratio between healthy and affected individuals.

A CSV file should include:

|filename|	label|	fold|
|--------|-------|------|
|sample1.wav|	0|	1|
|sample2.wav|	1|	1|


## Training

Run training using:
```
python train.py --config config.yaml
```

## Configuration

Example config.yaml:
```yaml
model_checkpoint: microsoft/wavlm-base
dataset: your_dataset_name
batch_size: 16
learning_rate: 3e-5
weight_decay: 0.01

ifaugment: true
aug_method: 0.8
max_val: 1.0
min_val: 0.0

sample_rate: 16000
num_samples: 16000
output_dir: results/
```

## Data Augmentation

Optional augmentation techniques include:

1. Low-pass filtering
2. Gaussian noise
3. Time stretching
4. Pitch shifting

These help improve model robustness.

## Evaluation

The model is evaluated using accuracy, F1-score, Precision and Confusion Matrix

Metrics are logged during training using TensorBoard.

## Logging & Checkpoints
- Logs are stored via TensorBoard
- Model checkpoints are automatically saved
- Best model is selected based on validation loss

## Model Architecture

The model is based on a pretrained WavLM encoder, followed by:

- Feature pooling
- Dropout
- Fully connected classification head


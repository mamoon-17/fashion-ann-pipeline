# Fashion-MNIST ANN

Name: Muhammad Mamoon Chishti  
Roll number: 23L-6050  
Section: BCS-7B

Assignment 3: Git, DVC, TensorFlow and Google Drive.

## Run

Run these commands from this directory in PowerShell:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
dvc repro
```

To restore existing artifacts from Drive, configure the OAuth client locally and run `dvc pull`.

## Pipeline

- `src/prepare.py`: download the original Fashion-MNIST arrays.
- `src/preprocess.py`: normalize pixels and split 54,000 training, 6,000 validation and 10,000 test images.
- `src/train.py`: train a fully connected ANN and save the model and training history.
- `src/evaluate.py`: save test metrics and the confusion matrix.

The architecture is Flatten, Dense with ReLU, Dropout, and Dense with 10 Softmax outputs. Hyperparameters are in `params.yaml`; stage definitions are in `dvc.yaml`.

## Results

The `v1` model uses 128 hidden units and reaches 88.75% test accuracy. The `v2` model uses 256 hidden units and reaches 88.81%. Both exceed the required 85%.

Git tracks the source, configuration, `metrics.json` and `dvc.lock`. DVC tracks the raw data, processed data, model, history and evaluation artifacts. The configured [Drive remote](https://drive.google.com/drive/folders/1jzd3TDNvWXpxZxwafHcL6GGdBtT-Jbod) stores these artifacts after `dvc push`.

`../submission_local/` contains the report, submission links, push proof and supporting evidence. It is outside this Git repository.

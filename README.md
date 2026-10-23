# Face Recognition under Challenging Image Conditions

AI Developer Intern assignment. A pre-trained ArcFace face-recognition network is fine-tuned on the company-provided images, and both the original and the fine-tuned model are evaluated on the same held-out 20% test set.

| Stage | What is used |
|---|---|
| Face detection | YOLOv8n face detector (`yolov8n-face-lindevs.pt`) |
| Face embeddings | InsightFace ArcFace IResNet-50 (`w600k_r50.onnx` from `buffalo_l`), rebuilt in PyTorch so it can be trained |
| Fine-tuning | ArcFace angular-margin loss over the 8 people, last network stage unfrozen, augmentation in memory |
| Identification | KNN classifier (scikit-learn) on the fine-tuned embeddings |
| Evaluation | Fixed same-person / different-person test pairs, scored the same way for both models |

## Results

All numbers come from one run (seed 42, NVIDIA RTX 2050) and are saved in `outputs/metrics/`. The full write-up is in `reports/final_report.pdf`.

| Test set (7,922 fixed pairs, 4,418 images) | Pre-trained baseline | Fine-tuned |
|---|---|---|
| ROC-AUC | 0.711 | **0.973** |
| Verification accuracy | 0.654 | **0.915** |
| Precision | 0.684 | **0.948** |
| Recall | 0.571 | **0.878** |
| F1-score | 0.623 | **0.912** |
| False acceptance rate | 0.264 | **0.048** |
| False rejection rate | 0.429 | **0.122** |
| KNN identification accuracy | 0.835 | **0.914** |

ROC-AUC by image quality of the two faces in a pair:

| Pair type | Baseline | Fine-tuned |
|---|---|---|
| high quality vs high quality | 0.922 | 0.999 |
| high quality vs low quality | 0.613 | 0.965 |
| low quality vs low quality | 0.590 | 0.939 |

KNN accuracy on the fine-tuned model was 0.991 for high-quality test images (862) and 0.895 for low-quality ones (3,556). Fine-tuning ran 10 epochs in 16.3 minutes, and epoch 8 scored best on validation. All eight data-leakage checks passed.

## Project structure

```text
face-recognition-project/
├── face_recognition.ipynb     Part 1 dataset split · Part 2 training · Part 3 inference
├── src/
│   ├── config.py              paths and all settings
│   ├── dataset.py             scanning, 80/20 split, near-duplicate removal, copying to data/train and data/test
│   ├── detection.py           YOLOv8 face detection and face cropping
│   ├── embeddings.py          IResNet-50 in PyTorch, ONNX weight import, embedding extraction
│   ├── training.py            ArcFace head, augmentation, fine-tuning loop, checkpoints
│   ├── classifier.py          KNN training, prediction, saving and loading
│   ├── inference.py           FaceRecognizer: loads the saved model and predicts
│   ├── evaluation.py          evaluation shared by the baseline and the fine-tuned model
│   ├── metrics.py             verification pairs, threshold, metrics
│   ├── plots.py               charts
│   └── logging_config.py      structlog setup
├── scripts/
│   ├── predict.py             recognise faces in an image or folder (--input)
│   ├── baseline_evaluation.py evaluate the pre-trained model
│   ├── final_evaluation.py    evaluate the fine-tuned model and compare
│   ├── check_data_leakage.py  check the split and every saved file for leakage
│   └── generate_report.py     build reports/final_report.pdf
├── outputs/                   charts, metrics, predictions, logs
├── reports/final_report.pdf   final report
├── requirements.txt
└── README.md
```

Not in git (see `.gitignore`): `data/` (images, split copies, manifests), `models/`, `checkpoints/`, `outputs/embeddings/` and `.venv/`.

## Requirements

- Python 3.11 (developed with 3.11.9)
- Windows 10/11 or Linux
- An NVIDIA GPU with at least 4 GB is recommended. On a CPU-only machine, the code trains just the embedding head for 3 epochs, which is slower and gives weaker results.
- About 2 GB of free disk space

## Setup

Windows PowerShell, from the project folder:

```powershell
python -m venv .venv
.venv\Scripts\activate

# PyTorch with CUDA (use the cpu index instead if you have no NVIDIA GPU)
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

On Linux, activate with `source .venv/bin/activate`.

When the environment is active, the prompt starts with `(.venv)`. From then on, use `python`, not `python3`.

### Dataset

Extract the company-provided `Images.zip` into `data/raw/Images/`:

```text
data/raw/Images/<person>/High_quality/...
data/raw/Images/<person>/Low_quality/...
```

Only these images are used. No other face datasets are used for training or testing.

> The images are confidential. Do not commit, upload or share them. `.gitignore` already keeps the whole `data/` folder out of git.

### Model weights

Put both files in `models/`:

| File | Download |
|---|---|
| `yolov8n-face-lindevs.pt` | https://github.com/lindevs/yolov8-face/releases/download/1.0.1/yolov8n-face-lindevs.pt |
| `w600k_r50.onnx` | inside https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip |

```powershell
mkdir models
curl.exe -L -o models/yolov8n-face-lindevs.pt https://github.com/lindevs/yolov8-face/releases/download/1.0.1/yolov8n-face-lindevs.pt
curl.exe -L -o buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
python -c "import zipfile; zipfile.ZipFile('buffalo_l.zip').extract('w600k_r50.onnx', 'models')"
```

## How to run

Run every step from the project folder, in this order. Each step uses the files the previous step saved.

| # | Step | Command | Output |
|---|---|---|---|
| 1 | Split the dataset 80/20 | Run **Part 1** of `face_recognition.ipynb` | `data/train/`, `data/test/`, `data/manifests/` |
| 2 | Evaluate the pre-trained model | `python -m scripts.baseline_evaluation` | `outputs/metrics/baseline_*`, test pairs |
| 3 | Fine-tune and fit the KNN | Run **Part 2** of the notebook | `checkpoints/`, `outputs/metrics/training_*` |
| 4 | Inference on the test set | Run **Part 3** of the notebook | `outputs/predictions/test_predictions.csv` |
| 5 | Evaluate the fine-tuned model | `python -m scripts.final_evaluation` | `outputs/metrics/comparison_*`, charts |
| 6 | Check for data leakage | `python -m scripts.check_data_leakage` | `outputs/metrics/integrity_summary.json` |
| 7 | Build the report | `python -m scripts.generate_report` | `reports/final_report.pdf` |

The baseline runs before training on purpose: Part 2 refuses to start until the baseline results exist, and the final evaluation reuses the baseline's test pairs exactly.

After step 2, the whole notebook can also be run from the command line:

```powershell
jupyter nbconvert --to notebook --execute --inplace face_recognition.ipynb --ExecutePreprocessor.timeout=7200
```

## Recognising faces in an image

After training (step 3), give an image or a folder with `--input`:

```powershell
# one image (keep the quotes when the name has spaces)
python -m scripts.predict --input "data\test\<person>\low_quality\<image>.png"

# every image in a folder, searched recursively
python -m scripts.predict --input data\test\<person>

# the whole test folder, saved to a CSV file
python -m scripts.predict --input data\test --output outputs\predictions\test_folder.csv
```

For each image, YOLOv8 finds the face, the fine-tuned network turns it into an embedding, and the KNN picks the person. The table shows:

| Column | Meaning |
|---|---|
| `image` | image path |
| `detection` | `detected`, or `full_image_fallback` when no face was found and the whole image was used |
| `predicted` | predicted person, or `unknown` when the face is not close enough to anyone in training |
| `similarity` | cosine similarity to the closest training face |
| `actual`, `correct` | shown when the path contains a person's folder name, e.g. `data\test\<person>\...` |

If the actual person is known, the overall accuracy is printed at the end.

## How the split prevents data leakage

The images are video frames, so a plain random split would put almost identical neighbouring frames in both train and test. To prevent this:

1. **Bad files out.** Unreadable images (154) and exact byte duplicates (178) are removed.
2. **Split by segment, not by frame.** Each video is cut into blocks of at most 50 consecutive frames. Whole blocks go to test (about 20%), and validation is taken from the training side (about 15%).
3. **Near-duplicates removed.** Training images that still look almost the same as a test image (32×32 thumbnail cosine ≥ 0.95) are removed from training: 1,602 against test and 506 against validation. The test set itself is never changed.
4. **Checked afterwards.** `scripts/check_data_leakage.py` re-checks the manifests, the copied folders (by file hash), the pairs, the checkpoints and the KNN.

Final split: **13,143 train · 2,285 validation · 4,418 test** images from 8 people. The test set is 20.1% of the images available at split time.

```text
data/
├── raw/Images/<person>/<High_quality|Low_quality>/   original images, never changed
├── train/<person>/<high_quality|low_quality>/        80%: training + validation
├── test/<person>/<high_quality|low_quality>/         20%: test only
└── manifests/                                        split lists, pairs, detection cache
```

## Configuration

Every setting is in `src/config.py`:

| Class | Main settings |
|---|---|
| `SplitConfig` | test/validation fraction, segment length, near-duplicate threshold, seed |
| `DetectionConfig` | detector confidence, minimum input size, face size |
| `TrainingConfig` | batch size, epochs, learning rates, optimizer, scheduler, ArcFace scale/margin, trainable layers, KNN `k` values |
| `EvaluationConfig` | test pairs per person, seed |

All random steps use seed 42. The settings used for a run are saved with its results: in `split_summary.json`, inside the checkpoints, and in `training_summary.json`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `The module 'venv' could not be loaded` | The folder name starts with a dot: `.venv\Scripts\python.exe` |
| `can't open file '...\predict'` | Run it as a module: `python -m scripts.predict --input ...` |
| `python3` says packages are missing | `python3` starts the Windows Store Python. Activate `.venv` and use `python` |
| `ModuleNotFoundError: No module named 'src'` | Run from the project folder with `python -m scripts.<name>` |
| `torch.cuda.is_available()` is `False` | The CPU build of PyTorch is installed. Reinstall from the CUDA index URL above |
| `Split manifests not found` | Run Part 1 of the notebook first |
| `Run scripts.baseline_evaluation first` | Run step 2 before Part 2 or the final evaluation |
| `knn_classifier.joblib was fitted ... different checkpoint` | `best_model.pt` changed after training. Run Part 2 again |
| CUDA out of memory | Lower `batch_size` in `TrainingConfig` or `embedding_batch_size` in `EvaluationConfig` |

## Limitations

- Each person's high-quality images come mostly from one video. Test and training frames therefore still share lighting, clothes and background, even after the segment split and the near-duplicate removal.
- Training and testing use the same 8 people. These results do not show how the model handles people it has never seen.
- The `unknown` threshold was never tested on real unknown people; no test image was rejected as unknown.
- The detector gives boxes but no face landmarks, so faces are cropped but not aligned.
- Many low-quality images are too small for the detector. For these the whole image is used, and each image's detection status is saved.

More detail is in the limitations section of `reports/final_report.pdf`.

## Licences

| Component | Licence |
|---|---|
| InsightFace `buffalo_l` (`w600k_r50.onnx`) | Non-commercial research use only |
| `lindevs/yolov8-face` weights | MIT (trained with Ultralytics) |
| Ultralytics | AGPL-3.0 |
| PyTorch, scikit-learn, OpenCV, NumPy, pandas, Matplotlib, ReportLab, structlog | Permissive (BSD / Apache-2.0 / MIT) |

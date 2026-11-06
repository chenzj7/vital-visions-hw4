# HW4 Vital Visions

This repository contains three experiments for Assignment 4:

1. Smartphone camera heart rate estimation
2. Medical image fine tuning with PneumoniaMNIST
3. Screen-to-camera communication with alpha-channel BFSK

The repository includes the source code, measured results, report, and a static website for GitHub Pages.

## Results

| Part | Main result |
| --- | --- |
| Part 1 | 75.15 bpm by FFT and 75.00 bpm by peak intervals |
| Part 2 | 79.17% test accuracy, 92.31% sensitivity, 57.26% specificity, and 0.865 ROC AUC |
| Part 3 | 48 measured recordings across four distances, two alpha differences, and six raw bit rates |

## Repository Structure

```text
hw4/
├── index.html                  GitHub Pages website
├── styles.css                 Website styles
├── part1/                     Heart rate analysis
├── part2/                     PneumoniaMNIST fine tuning
├── part3/                     Alpha-BFSK transmitter and receiver
├── output/                    Word and PDF reports
├── report/                    Report generation scripts
└── tests/                     Smoke tests
```

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the analyses:

```bash
python part1/heart_rate.py \
  --video dataset/part1.MOV \
  --discard-seconds 2 \
  --output-dir part1/results

python part2/finetune_medmnist.py \
  --epochs 5 \
  --batch-size 64 \
  --image-size 96 \
  --output-dir part2/results

python part3/analyze_dataset.py \
  --dataset-dir dataset \
  --message VITAL \
  --output-dir part3/results_measured

python tests/smoke_test.py
```

## Part 1 Smartphone Heart Rate

The script converts each video frame to grayscale, calculates mean intensity, removes slow drift, and estimates heart rate with an FFT and peak intervals. The analyzed recording is 34.33 seconds at approximately 30 fps.

## Part 2 Medical Image Fine Tuning

The experiment uses an ImageNet-pretrained ResNet-18 and PneumoniaMNIST. Images are resized to 96 by 96 pixels. The convolutional backbone is frozen, and the final classification layer is trained for five epochs.

The notebook in `part2/Part2_Medical_Finetuning.ipynb` can run in Google Colab. The command-line version is `part2/finetune_medmnist.py`.

## Part 3 Alpha-Channel BFSK

Open `part3/transmitter.html` in a browser to send a framed message through screen opacity changes. The receiver uses grayscale intensity, fractional symbol timing, noncoherent energy detection, and CRC-16.

The 48 measured videos are ordered by distance: 20, 40, 80, and 120 cm. Each distance contains alpha difference 0.1 at 1, 2, 4, 6, 8, and 10 bit/s, followed by alpha difference 0.5 at the same rates.

## GitHub Pages

The workflow in `.github/workflows/pages.yml` deploys the repository root to GitHub Pages after every push to `main`.

After pushing the repository to GitHub:

1. Open **Settings > Pages**.
2. Set **Source** to **GitHub Actions**.
3. Run the **Deploy GitHub Pages** workflow if it does not start automatically.

## Reports

- [Word report](output/docx/HW4_Vital_Visions_Report.docx)

The smartphone heart rate estimate is an educational optical measurement and is not medical advice.

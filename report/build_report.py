#!/usr/bin/env python3
"""Build the HW4 PDF report from generated metrics and figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "HW4_Vital_Visions_Report.pdf"
NAVY = colors.HexColor("#123B52")
TEAL = colors.HexColor("#246F71")
ORANGE = colors.HexColor("#D8854F")
PALE = colors.HexColor("#EAF1F4")
DARK = colors.HexColor("#18242C")
MUTED = colors.HexColor("#52636E")


def load_json(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def load_csv(relative_path: str):
    with (ROOT / relative_path).open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def fitted_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as image:
        width, height = image.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def page_decoration(canvas, doc) -> None:
    canvas.saveState()
    width, height = letter
    canvas.setStrokeColor(colors.HexColor("#C7D3D9"))
    canvas.setLineWidth(0.6)
    canvas.line(doc.leftMargin, height - 0.46 * inch, width - doc.rightMargin, height - 0.46 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, height - 0.36 * inch, "HW4 - Vital Visions")
    canvas.drawRightString(width - doc.rightMargin, 0.38 * inch, f"Page {doc.page}")
    canvas.restoreState()


def metric_table(rows, widths=None):
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B5C5CD")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def build() -> None:
    part1 = load_json("part1/results/part1_metrics.json")
    part2 = load_json("part2/results/part2_metrics.json")
    part3_rows = load_csv("part3/results_measured/part3_measured_results.csv")
    part3_summary = load_csv("part3/results_measured/part3_measured_summary.csv")
    crc_passes = sum(row["crc_ok"] == "True" for row in part3_rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=28, leading=32, textColor=NAVY, alignment=TA_LEFT, spaceAfter=12))
    styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], fontSize=13, leading=18, textColor=TEAL, spaceAfter=18))
    styles.add(ParagraphStyle(name="H1Custom", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=NAVY, spaceBefore=4, spaceAfter=10))
    styles.add(ParagraphStyle(name="H2Custom", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=17, textColor=TEAL, spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodyCustom", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.5, leading=14, textColor=DARK, spaceAfter=7))
    styles.add(ParagraphStyle(name="BulletCustom", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.3, leading=13.5, leftIndent=14, firstLineIndent=-7, bulletIndent=6, textColor=DARK, spaceAfter=4))
    styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontName="Helvetica", fontSize=9, leading=13.5, textColor=DARK, backColor=colors.HexColor("#FFF1E6"), borderColor=ORANGE, borderWidth=0.7, borderPadding=9, spaceBefore=6, spaceAfter=10))
    styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontName="Helvetica-Oblique", fontSize=8, leading=11, alignment=TA_CENTER, textColor=MUTED, spaceBefore=3, spaceAfter=10))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.6, leading=10.5, textColor=MUTED, spaceAfter=4))

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        rightMargin=0.62 * inch,
        leftMargin=0.62 * inch,
        topMargin=0.67 * inch,
        bottomMargin=0.62 * inch,
        title="HW4 - Vital Visions",
        author="Course submission",
        subject="Smartphone PPG, medical image fine-tuning, and alpha-channel BFSK",
    )
    story = []

    story += [Spacer(1, 0.48 * inch), Paragraph("HW4: Vital Visions", styles["TitleCustom"]), Paragraph("Smartphone photoplethysmography, medical image fine-tuning, and screen-camera BFSK communication", styles["Subtitle"])]
    story.append(Spacer(1, 0.12 * inch))
    overview = [
        ["Part", "System", "Evidence in this report"],
        ["1", "Phone-camera heart-rate estimation", "34.33-second finger-over-camera recording at 30 fps"],
        ["2", "Medical image model fine-tuning", "Executed ResNet-18 training and held-out PneumoniaMNIST test"],
        ["3", "Alpha-channel screen-camera BFSK", "48 phone-camera recordings across 20-120 cm, two alpha differences, and six rates"],
    ]
    story.append(metric_table(overview, [0.45 * inch, 2.2 * inch, 4.0 * inch]))
    story.append(Spacer(1, 0.18 * inch))
    story.append(Paragraph("Evidence statement", styles["H2Custom"]))
    story.append(Paragraph("All primary results in this report are now measured or executed results. Part 1 uses the supplied phone recording, Part 2 uses an executed five-epoch fine-tuning run, and Part 3 uses 48 supplied screen-camera recordings. Part 3 has one recording per condition, so its curves show individual observations rather than repeated-trial means.", styles["Callout"]))
    story.append(Paragraph("Repository link: add the final GitHub URL here before Canvas upload.", styles["BodyCustom"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Executive summary", styles["H2Custom"]))
    story.append(Paragraph(f"The phone PPG pipeline estimated {part1['fft_heart_rate_bpm']:.2f} bpm by FFT and {part1['peak_interval_heart_rate_bpm']:.2f} bpm from inter-peak intervals. The medical image experiment reached {100*part2['best_validation_accuracy']:.2f}% best validation accuracy and {100*part2['accuracy']:.2f}% held-out test accuracy. In the 48 BFSK recordings, {crc_passes} frames passed CRC. Performance was non-monotonic: 6 bit/s was often more robust than 1-2 bit/s because slow opacity changes were attenuated by automatic exposure, while 8 bit/s suffered from having only 3.75 camera frames per symbol.", styles["BodyCustom"]))

    story.append(PageBreak())
    story.append(Paragraph("Part 1 - Smartphone-based heart-rate monitoring", styles["H1Custom"]))
    story.append(Paragraph("Method", styles["H2Custom"]))
    for text in [
        "Each decoded video frame is converted from OpenCV BGR to grayscale and reduced to one sample using mean pixel intensity.",
        "The mean is removed and a third-order, zero-phase high-pass Butterworth filter at 0.5 Hz suppresses DC and auto-exposure drift.",
        "A Hann-window FFT is searched only from 1.000 to 1.667 Hz, corresponding to the requested 60 to 100 bpm range.",
        "The dominant FFT frequency sets the expected beat period. Peak detection evaluates both signal polarities and selects the more regular series that agrees with the FFT estimate.",
    ]:
        story.append(Paragraph(text, styles["BulletCustom"], bulletText="-"))
    story.append(Paragraph("MinPeakDistance calculation", styles["H2Custom"]))
    story.append(Paragraph(f"The estimated number of camera samples per heartbeat is N = fs / fHR = fs * 60 / HR. For the measured FFT estimate, N = {part1['fps']:.3f} * 60 / {part1['fft_heart_rate_bpm']:.3f} = {part1['samples_per_beat_from_fft']:.2f} samples/beat. The implementation uses a 75% tolerance, so MinPeakDistance = {part1['min_peak_distance_samples']} samples. The tolerance permits modest beat-to-beat variation while the prominence rule rejects the pulse harmonic.", styles["BodyCustom"]))
    rows = [
        ["Metric", "Measured value"],
        ["Duration / sampling rate", f"{part1['duration_seconds']:.1f} s / {part1['fps']:.1f} Hz"],
        ["FFT peak", f"{part1['fft_peak_hz']:.3f} Hz"],
        ["FFT heart rate", f"{part1['fft_heart_rate_bpm']:.2f} bpm"],
        ["Peak-interval heart rate", f"{part1['peak_interval_heart_rate_bpm']:.2f} bpm"],
        ["Selected peaks", str(part1["selected_peak_count"])],
        ["Samples per beat", f"{part1['samples_per_beat_from_fft']:.2f}"],
    ]
    story.append(metric_table(rows, [2.3 * inch, 3.2 * inch]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(fitted_image(ROOT / "part1/results/part1_fft.png", 6.8 * inch, 3.45 * inch))
    story.append(Paragraph("Figure 1. FFT amplitude spectrum from the phone recording. The shaded interval is the 60-100 bpm search band.", styles["Caption"]))

    story.append(PageBreak())
    story.append(Paragraph("Part 1 - Time-domain result", styles["H1Custom"]))
    story.append(fitted_image(ROOT / "part1/results/part1_peaks.png", 6.95 * inch, 4.35 * inch))
    story.append(Paragraph("Figure 2. High-pass filtered phone-camera PPG with selected heartbeat peaks.", styles["Caption"]))
    story.append(Paragraph("Interpretation", styles["H2Custom"]))
    story.append(Paragraph(f"The FFT and inter-peak estimates differ by only {abs(part1['fft_heart_rate_bpm']-part1['peak_interval_heart_rate_bpm']):.2f} bpm. The detector selected {part1['selected_peak_count']} beats over {part1['duration_seconds']:.2f} seconds, consistent with an approximately 75 bpm pulse. Agreement between the frequency-domain and time-domain estimates supports the stability of this recording, but the result remains an educational optical estimate rather than a clinical measurement.", styles["BodyCustom"]))
    story.append(Paragraph("Real recording protocol", styles["H2Custom"]))
    story.append(Paragraph("Record 30 seconds with the flash enabled and a finger covering the lens. Keep pressure and phone position stable. Discard approximately the first two seconds to allow exposure to settle. Run the same analysis using the video input flag and inspect the trace for clipping, steps, or motion artifacts. Re-record if the peak sequence is unstable or if the FFT maximum sits at a search-band boundary.", styles["BodyCustom"]))
    story.append(Paragraph("Limitations", styles["H2Custom"]))
    story.append(Paragraph("Mean grayscale discards color information; the red or green channel may provide a stronger PPG signal on some devices. Auto-exposure, rolling shutter, motion, pressure, and frame-rate variability can bias the result. The output is an educational estimate and is not a medical measurement.", styles["BodyCustom"]))

    story.append(PageBreak())
    story.append(Paragraph("Part 2 - Fine-tuning for medical image diagnostics", styles["H1Custom"]))
    story.append(Paragraph("Model and dataset", styles["H2Custom"]))
    story.append(Paragraph("The experiment uses an ImageNet-pretrained ResNet-18 and PneumoniaMNIST, a binary pediatric chest X-ray dataset. Images are resized from 28 x 28 to 96 x 96, converted to three channels, and normalized with ImageNet statistics. Training augmentation uses horizontal flips and rotations up to 7 degrees. The ImageNet convolutional backbone is frozen and the final fully connected layer is replaced with a two-class classifier.", styles["BodyCustom"]))
    story.append(Paragraph("Training configuration", styles["H2Custom"]))
    config_rows = [
        ["Item", "Value"],
        ["Dataset sizes", "4,708 train / 524 validation / 624 test"],
        ["Optimizer", "AdamW, learning rate 1e-3, weight decay 1e-4"],
        ["Epochs / batch size", f"{part2['epochs']} / 64"],
        ["Trainable parameters", "ResNet-18 classification head"],
        ["Execution device", str(part2["device"])],
        ["Model selection", "Best validation accuracy"],
    ]
    story.append(metric_table(config_rows, [2.0 * inch, 4.7 * inch]))
    story.append(Paragraph("Results", styles["H2Custom"]))
    test_rows = [
        ["Metric", "Result"],
        ["Best validation accuracy", f"{100*part2['best_validation_accuracy']:.2f}%"],
        ["Held-out test accuracy", f"{100*part2['accuracy']:.2f}%"],
        ["Pneumonia sensitivity", f"{100*part2['sensitivity']:.2f}%"],
        ["Normal-image specificity", f"{100*part2['specificity']:.2f}%"],
        ["ROC AUC", f"{part2['roc_auc']:.3f}"],
        ["Confusion matrix [TN FP; FN TP]", str(part2["confusion_matrix"])],
    ]
    story.append(metric_table(test_rows, [2.7 * inch, 2.8 * inch]))
    story.append(PageBreak())
    story.append(Paragraph("Part 2 - Training and test-set evaluation", styles["H1Custom"]))
    story.append(fitted_image(ROOT / "part2/results/part2_training_history.png", 6.75 * inch, 2.25 * inch))
    story.append(Paragraph("Figure 3. Five-epoch training and validation histories from the executed run.", styles["Caption"]))
    story.append(fitted_image(ROOT / "part2/results/part2_test_evaluation.png", 6.85 * inch, 2.45 * inch))
    story.append(Paragraph("Figure 4. Confusion matrix and ROC curve on the held-out test set.", styles["Caption"]))
    story.append(Paragraph("Interpretation", styles["H2Custom"]))
    story.append(Paragraph("The model detects 360 of 390 pneumonia images, giving high sensitivity (92.31%), but incorrectly labels 100 of 234 normal images as pneumonia, so specificity is only 57.26%. The ROC AUC of 0.865 shows useful ranking ability even though the default 0.5 threshold is not well balanced. The lower held-out accuracy relative to validation accuracy also argues against reporting validation accuracy alone.", styles["BodyCustom"]))
    story.append(Paragraph("Potential improvements", styles["H2Custom"]))
    for text in [
        "Choose an operating threshold on the validation set to balance sensitivity and specificity for the intended use.",
        "Use class-weighted loss or balanced sampling, then report confidence intervals over multiple random seeds.",
        "Unfreeze the last ResNet stage with a smaller learning rate and apply early stopping.",
        "Audit patient-level separation, acquisition-site shift, and subgroup performance before any clinical interpretation.",
    ]:
        story.append(Paragraph(text, styles["BulletCustom"], bulletText="-"))
    story.append(Paragraph("This educational experiment is not a medical device evaluation. Small, curated benchmark images do not establish robustness to clinical scanners, disease prevalence, patient demographics, or workflow conditions.", styles["Callout"]))

    story.append(PageBreak())
    story.append(Paragraph("Part 3 - Real-time alpha-channel BFSK", styles["H1Custom"]))
    story.append(Paragraph("Communication design", styles["H2Custom"]))
    story.append(Paragraph("A full-screen white overlay changes opacity on top of a static color image. For raw bit rate R, bit 0 uses f0 = R Hz (one cycle per symbol) and bit 1 uses f1 = 2R Hz (two cycles per symbol). Each frame contains a 16-bit alternating preamble, a 16-bit payload length, UTF-8 message bytes, and a CRC-16. The receiver averages grayscale intensity inside a fixed region of interest, removes slow drift, tries every sample-to-symbol alignment, and uses non-coherent two-tone energy detection.", styles["BodyCustom"]))
    story.append(Paragraph("Metrics", styles["H2Custom"]))
    story.append(Paragraph("BER = 1 - correctly received bits / 88 transmitted frame bits. Missing bits at the beginning or end of a clipped recording count as errors. Correct-bit rate = requested raw bit rate * (1 - BER). The phone camera recorded at approximately 30 fps. Because 4 and 8 bit/s correspond to 7.5 and 3.75 samples per symbol, respectively, the receiver uses fractional symbol boundaries rather than accumulating an integer-rounding clock error. A known VITAL frame is used for timing alignment, followed by bit comparison and an independent CRC-16 check.", styles["BodyCustom"]))
    story.append(fitted_image(ROOT / "part3/results_measured/part3_ber_measured.png", 6.9 * inch, 3.65 * inch))
    story.append(Paragraph("Figure 5. Measured BER for all 48 phone recordings. Each point is one recording, so no error bars are shown.", styles["Caption"]))
    story.append(Paragraph("Maximum measured rate with BER at or below 1%", styles["H2Custom"]))
    summary_table = [["delta alpha", "Distance", "Max rate", "BER", "CRC passes"]]
    for row in part3_summary:
        rate = row["max_rate_at_ber_le_1pct_bps"]
        summary_table.append([row["delta_alpha"], f"{row['distance_cm']} cm", "None" if rate == "none" else f"{float(rate):.0f} bit/s", row["ber_at_max_reliable_rate"], f"{row['crc_pass_count']}/{row['conditions_tested']}"])
    story.append(metric_table(summary_table, [0.85 * inch, 1.0 * inch, 1.25 * inch, 1.0 * inch, 1.1 * inch]))

    story.append(PageBreak())
    story.append(Paragraph("Part 3 - Discussion and acquisition limits", styles["H1Custom"]))
    story.append(Paragraph("Observed hardware trends", styles["H2Custom"]))
    for text in [
        "The link is not monotonic with bit rate. Slow 1-2 bit/s opacity changes can be followed and partially cancelled by the phone's automatic exposure, while the 6 bit/s modulation often remains visible to the decoder.",
        "The 8 bit/s condition is consistently weak because each symbol contains only about 3.75 camera frames; small sampling-phase changes strongly affect the two-tone energy estimate.",
        "The 10 bit/s result is device- and phase-dependent. The 20 cm, delta-alpha 0.5 recording passed CRC with zero BER, while most other 10 bit/s recordings had 9-35% BER. This is consistent with sampling resonance rather than a generally robust 10 bit/s link.",
        "Distance and delta alpha do not show a perfectly monotonic effect because several clips omit leading/trailing bits and there is only one recording per condition. Repeated trials with locked exposure are needed for confidence intervals.",
    ]:
        story.append(Paragraph(text, styles["BulletCustom"], bulletText="-"))
    story.append(Paragraph("Captured experiment", styles["H2Custom"]))
    hardware_rows = [
        ["Factor", "Measured setting"],
        ["Distances", "20, 40, 80, and 120 cm from screen to camera"],
        ["Opacity contrasts", "delta alpha = 0.1 and 0.5"],
        ["Rates", "1, 2, 4, 6, 8, and 10 bit/s"],
        ["Repetitions", "One recording for each of 48 conditions"],
        ["Camera", "Portrait HEVC video, approximately 30 fps, 1080 x 1920"],
        ["Ground truth", "VITAL payload; 88-bit frame including preamble, length, payload, and CRC"],
    ]
    story.append(metric_table(hardware_rows, [1.2 * inch, 5.5 * inch]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("The analysis saves one JSON record per video plus a combined CSV. Several clips begin after the transmitted frame has already started; the receiver counts those missing leading bits as errors, matching the assignment definition of BER. CRC pass/fail is reported separately so a frame with an isolated preamble error can still have a valid payload and CRC.", styles["BodyCustom"]))
    story.append(Paragraph("Each curve contains one recording per condition. Repeating every condition would allow mean BER, standard deviation, and error bars instead of single observations.", styles["Callout"]))

    story.append(PageBreak())
    story.append(Paragraph("Conclusion and reproducibility", styles["H1Custom"]))
    story.append(Paragraph("The submission implements and executes the requested sensing and communication pipelines end to end. The real PPG recording produces consistent FFT and peak-interval estimates near 75 bpm. The medical image experiment supplies an executed transfer-learning result and bonus ROC/sensitivity/specificity analysis. The alpha-BFSK system includes a browser transmitter, fractional-timing camera receiver, CRC framing, and measured BER from 48 recordings.", styles["BodyCustom"]))
    story.append(Paragraph("Reproduction commands", styles["H2Custom"]))
    commands = [
        ["Part 1 video", "python part1/heart_rate.py --video dataset/part1.MOV --discard-seconds 2 --output-dir part1/results"],
        ["Part 2", "python part2/finetune_medmnist.py --epochs 5 --batch-size 64 --image-size 96 --output-dir part2/results"],
        ["Part 3 videos", "python part3/analyze_dataset.py --dataset-dir dataset --message VITAL --output-dir part3/results_measured"],
        ["Tests", "python tests/smoke_test.py"],
    ]
    command_rows = [["Purpose", "Command"]] + [[Paragraph(a, styles["Small"]), Paragraph(b, styles["Small"])] for a, b in commands]
    story.append(metric_table(command_rows, [1.25 * inch, 5.45 * inch]))
    story.append(Paragraph("Submission checklist", styles["H2Custom"]))
    for text in [
        "If time permits, repeat each Part 3 condition to calculate mean BER and error bars.",
        "Add the final GitHub repository URL to the title page and Canvas submission.",
        "Optionally upload a communication demonstration video.",
    ]:
        story.append(Paragraph(text, styles["BulletCustom"], bulletText="-"))
    story.append(Paragraph("References", styles["H2Custom"]))
    references = [
        "1. Jonathan, E. and Leahy, M. Investigating a smartphone imaging unit for photoplethysmography. Physiological Measurement 31.11 (2010): N79.",
        "2. Gregoski, M. J. et al. Development and validation of a smartphone heart rate acquisition application. International Journal of Telemedicine and Applications (2012).",
        "3. Esteva, A. et al. Dermatologist-level classification of skin cancer with deep neural networks. Nature 542.7639 (2017): 115-118.",
        "4. Wang, E. J. et al. HemaApp: noninvasive blood screening of hemoglobin using smartphone cameras. UbiComp (2016).",
        "5. Xuan, Y. et al. Ultra-low-cost mechanical smartphone attachment for no-calibration blood pressure measurement. Scientific Reports 13.1 (2023): 8105.",
        "6. HiLight: Hiding Bits in Pixel Translucency Changes. MobiSys 2015 paper linked from the assignment page.",
        "7. Yang, J. et al. MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data 10 (2023): 41.",
    ]
    for reference in references:
        story.append(Paragraph(reference, styles["Small"]))

    doc.build(story, onFirstPage=page_decoration, onLaterPages=page_decoration)
    print(OUTPUT)


if __name__ == "__main__":
    build()


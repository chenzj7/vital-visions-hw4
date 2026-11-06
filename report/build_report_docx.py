#!/usr/bin/env python3
"""Build an editable Word version of the HW4 Vital Visions report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "docx" / "HW4_Vital_Visions_Report.docx"
HEADER_FILL = "E7E6E6"
BORDER = "A6A6A6"
BLACK = RGBColor(0, 0, 0)


def load_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def load_csv(relative: str):
    with (ROOT / relative).open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def set_font(run, name: str = "Times New Roman", size: float | None = None, bold=None, color=BLACK):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    run.font.color.rgb = color


def set_cell_shading(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=110, bottom=90, end=110):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = borders.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            borders.append(node)
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), "4")
        node.set(qn("w:color"), BORDER)


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def remove_paragraph_borders(paragraph_properties):
    """Remove inherited/template paragraph rules such as the Title bottom border."""
    borders = paragraph_properties.find(qn("w:pBdr"))
    if borders is not None:
        paragraph_properties.remove(borders)


def add_table(doc, headers, rows, widths, font_size=9.5, centered_columns=()):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    header = table.rows[0]
    set_repeat_table_header(header)
    for index, text in enumerate(headers):
        cell = header.cells[index]
        cell.width = Inches(widths[index])
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_shading(cell, HEADER_FILL)
        set_cell_margins(cell)
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(0)
        run = paragraph.add_run(str(text))
        set_font(run, size=10, bold=True)
    for row_index, values in enumerate(rows):
        cells = table.add_row().cells
        for index, value in enumerate(values):
            cell = cells[index]
            cell.width = Inches(widths[index])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell)
            paragraph = cell.paragraphs[0]
            paragraph.alignment = (
                WD_ALIGN_PARAGRAPH.CENTER if index in centered_columns else WD_ALIGN_PARAGRAPH.LEFT
            )
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.0
            run = paragraph.add_run(str(value))
            set_font(run, size=font_size)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_caption(doc, text: str):
    paragraph = doc.add_paragraph(style="Caption")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = False
    run = paragraph.add_run(text)
    set_font(run, size=10)
    run.italic = True
    return paragraph


def add_figure(doc, relative: str, width: float, caption: str, alt_text: str):
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run()
    shape = run.add_picture(str(ROOT / relative), width=Inches(width))
    doc_pr = shape._inline.docPr
    doc_pr.set("title", alt_text)
    doc_pr.set("descr", alt_text)
    add_caption(doc, caption)


def add_bullet(doc, text: str):
    paragraph = doc.add_paragraph(style="List Bullet")
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.paragraph_format.line_spacing = 1.0
    run = paragraph.add_run(text)
    set_font(run, size=12)
    return paragraph


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("Page ")
    set_font(run, size=10)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    paragraph._p.append(fld)


def configure_document(doc: Document):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(12)
    normal.font.color.rgb = BLACK
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15

    title = styles["Title"]
    title.font.name = "Times New Roman"
    title._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    title._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    title._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    title.font.size = Pt(18)
    title.font.bold = True
    title.font.color.rgb = BLACK
    title.paragraph_format.space_after = Pt(10)
    remove_paragraph_borders(title._element.get_or_add_pPr())

    subtitle = styles["Subtitle"]
    subtitle.font.name = "Times New Roman"
    subtitle._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    subtitle._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    subtitle._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    subtitle.font.size = Pt(12)
    subtitle.font.color.rgb = BLACK
    subtitle.font.italic = False
    subtitle.paragraph_format.space_after = Pt(16)

    for name, size, before, after in (
        ("Heading 1", 14, 12, 6),
        ("Heading 2", 12, 8, 4),
        ("Heading 3", 12, 6, 3),
    ):
        style = styles[name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = BLACK
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    caption = styles["Caption"]
    caption.font.name = "Times New Roman"
    caption._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    caption._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    caption._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    caption.font.size = Pt(10)
    caption.font.italic = True
    caption.font.color.rgb = BLACK
    caption.paragraph_format.space_after = Pt(8)

    footer = section.footer
    footer.paragraphs[0].text = ""
    add_page_number(footer.paragraphs[0])


def add_body(doc, text: str, bold_lead: str | None = None):
    paragraph = doc.add_paragraph()
    if bold_lead:
        run = paragraph.add_run(bold_lead)
        set_font(run, size=12, bold=True)
    run = paragraph.add_run(text)
    set_font(run, size=12)
    return paragraph


def build():
    part1 = load_json("part1/results/part1_metrics.json")
    part2 = load_json("part2/results/part2_metrics.json")
    part3 = load_csv("part3/results_measured/part3_measured_results.csv")
    summary = load_csv("part3/results_measured/part3_measured_summary.csv")
    crc_passes = sum(row["crc_ok"] == "True" for row in part3)

    doc = Document()
    configure_document(doc)
    core = doc.core_properties
    core.title = "HW4 Vital Visions"
    core.subject = "Smartphone PPG medical image fine tuning and screen camera BFSK"
    core.author = "Course submission"

    title = doc.add_paragraph("HW4 Vital Visions Report", style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    remove_paragraph_borders(title._p.get_or_add_pPr())
    add_body(
        doc,
        "This report presents the phone heart rate measurement, PneumoniaMNIST model, and screen camera "
        "communication experiments for HW4. Results come from the recorded or executed experiments."
    )

    # Part 1
    doc.add_heading("Part 1 Smartphone Based Heart Rate Monitoring", level=1)
    doc.add_heading("Method", level=2)
    add_body(
        doc,
        "Each frame was converted to grayscale and reduced to its mean intensity. The signal was mean "
        "centered, high pass filtered at 0.5 Hz, and analyzed with a Hann window FFT from 1.000 to 1.667 Hz. "
        f"The FFT estimate gave {part1['samples_per_beat_from_fft']:.2f} samples per beat, so the minimum peak "
        f"distance was set to {part1['min_peak_distance_samples']} samples."
    )
    doc.add_heading("Results", level=2)
    add_table(
        doc,
        ["Metric", "Measured Value"],
        [
            ["Duration and sampling rate", f"{part1['duration_seconds']:.2f} s at {part1['fps']:.3f} Hz"],
            ["FFT peak", f"{part1['fft_peak_hz']:.3f} Hz"],
            ["FFT heart rate", f"{part1['fft_heart_rate_bpm']:.2f} bpm"],
            ["Peak interval heart rate", f"{part1['peak_interval_heart_rate_bpm']:.2f} bpm"],
            ["Selected peaks", str(part1["selected_peak_count"])],
            ["Samples per beat", f"{part1['samples_per_beat_from_fft']:.2f}"],
        ],
        [3.0, 3.4],
    )
    add_figure(
        doc,
        "part1/results/part1_fft.png",
        5.75,
        "Figure 1  FFT amplitude spectrum from the phone recording",
        "FFT amplitude spectrum with the 60 to 100 bpm search band and a peak near 1.252 Hz",
    )
    doc.add_heading("Part 1 Time Domain Result", level=1)
    add_figure(
        doc,
        "part1/results/part1_peaks.png",
        6.45,
        "Figure 2  Filtered phone camera PPG with selected heartbeat peaks",
        "Time domain phone camera PPG trace with 42 selected heartbeat peaks",
    )
    doc.add_heading("Interpretation", level=2)
    add_body(
        doc,
        f"The two estimates differ by {abs(part1['fft_heart_rate_bpm']-part1['peak_interval_heart_rate_bpm']):.2f} bpm. "
        "Motion, exposure control, finger pressure, and frame timing can affect this optical estimate."
    )

    # Part 2
    doc.add_heading("Part 2 Medical Image Fine Tuning", level=1)
    doc.add_heading("Model and Dataset", level=2)
    add_body(
        doc,
        "An ImageNet pretrained ResNet 18 was trained on PneumoniaMNIST. Images were resized to 96 by 96, "
        "converted to three channels, and normalized with ImageNet statistics. The backbone was frozen and "
        "only the final classifier was trained."
    )
    add_table(
        doc,
        ["Item", "Configuration"],
        [
            ["Dataset sizes", "4708 training  524 validation  624 test"],
            ["Optimizer", "AdamW  learning rate 1e-3  weight decay 1e-4"],
            ["Epochs and batch size", f"{part2['epochs']} epochs  batch size 64"],
            ["Trainable parameters", "ResNet 18 classification head"],
            ["Execution device", str(part2["device"])],
            ["Model selection", "Best validation accuracy"],
        ],
        [2.35, 4.05],
    )
    doc.add_heading("Results", level=2)
    add_table(
        doc,
        ["Metric", "Result"],
        [
            ["Best validation accuracy", f"{100*part2['best_validation_accuracy']:.2f}%"],
            ["Held out test accuracy", f"{100*part2['accuracy']:.2f}%"],
            ["Pneumonia sensitivity", f"{100*part2['sensitivity']:.2f}%"],
            ["Normal image specificity", f"{100*part2['specificity']:.2f}%"],
            ["ROC AUC", f"{part2['roc_auc']:.3f}"],
            ["Confusion matrix", str(part2["confusion_matrix"])],
        ],
        [3.0, 3.4],
    )
    add_figure(
        doc,
        "part2/results/part2_training_history.png",
        5.70,
        "Figure 3  Five epoch training and validation histories",
        "Training and validation loss and accuracy over five epochs",
    )
    doc.add_heading("Part 2 Test Set Evaluation", level=1)
    add_figure(
        doc,
        "part2/results/part2_test_evaluation.png",
        5.70,
        "Figure 4  Confusion matrix and ROC curve on the held out test set",
        "PneumoniaMNIST test confusion matrix and ROC curve with AUC 0.865",
    )
    doc.add_heading("Interpretation", level=2)
    add_body(
        doc,
        "The model found 360 of 390 pneumonia images but classified 100 of 234 normal images as pneumonia. "
        "A validation threshold or class balancing could improve specificity."
    )

    # Part 3
    doc.add_heading("Part 3 Alpha Channel Screen Camera BFSK", level=1)
    doc.add_heading("Method", level=2)
    add_body(
        doc,
        "A white overlay changed opacity over a static image. Bit 0 used frequency R and bit 1 used 2R. The "
        "88 bit VITAL frame contained a preamble, payload length, message, and CRC 16. The experiment used "
        "alpha differences 0.1 and 0.5, rates 1, 2, 4, 6, 8, and 10 bit per second, and distances 20, 40, 80, "
        "and 120 cm. BER counted missing edge bits as errors."
    )
    doc.add_heading("Results", level=2)
    add_body(doc, f"The experiment produced 48 recordings. {crc_passes} recordings passed CRC.")
    add_figure(
        doc,
        "part3/results_measured/part3_ber_measured.png",
        6.45,
        "Figure 5  Measured BER for 48 phone recordings",
        "Measured BER versus requested bit rate at 20 40 80 and 120 centimeters for two alpha differences",
    )
    doc.add_heading("Maximum Rate at or Below One Percent BER", level=2)
    summary_rows = []
    for row in summary:
        rate = row["max_rate_at_ber_le_1pct_bps"]
        summary_rows.append(
            [
                row["distance_cm"],
                row["delta_alpha"],
                "None" if rate == "none" else f"{float(rate):.0f}",
                row["ber_at_max_reliable_rate"],
                f"{row['crc_pass_count']}/{row['conditions_tested']}",
            ]
        )
    add_table(
        doc,
        ["Distance cm", "Delta Alpha", "Max Rate bit s", "BER", "CRC Passes"],
        summary_rows,
        [1.15, 1.2, 1.45, 1.1, 1.25],
        centered_columns=(0, 1, 2, 3, 4),
    )
    doc.add_page_break()
    doc.add_heading("Part 3 Discussion", level=1)
    for text in (
        "Six bit per second often decoded better than slower rates because automatic exposure reduced slow intensity changes.",
        "Eight bit per second was weak because each symbol contained about 3.75 camera frames and was sensitive to sampling phase.",
        "Each condition has one recording, so the results do not include averages or error bars.",
    ):
        add_bullet(doc, text)

    # Editable full results appendix
    doc.add_heading("Appendix A Complete Part 3 Results", level=1)
    add_body(doc, "Correct bit rate equals requested rate multiplied by one minus BER.")
    detail_rows = []
    for row in part3:
        detail_rows.append(
            [
                row["distance_cm"],
                row["delta_alpha"],
                f"{float(row['requested_bit_rate_bps']):.0f}",
                f"{float(row['ber']):.4f}",
                f"{float(row['correct_bit_rate_bps']):.3f}",
                "Pass" if row["crc_ok"] == "True" else "Fail",
            ]
        )
    add_table(
        doc,
        ["Distance", "Alpha", "Rate", "BER", "Correct Rate", "CRC"],
        detail_rows,
        [1.0, 0.9, 0.85, 1.05, 1.35, 0.85],
        font_size=8.5,
        centered_columns=(0, 1, 2, 3, 4, 5),
    )

    # References
    doc.add_page_break()
    doc.add_heading("References", level=1)
    references = [
        "Jonathan E and Leahy M. Investigating a smartphone imaging unit for photoplethysmography. Physiological Measurement 31.11 2010 N79.",
        "Gregoski M J et al. Development and validation of a smartphone heart rate acquisition application. International Journal of Telemedicine and Applications 2012.",
        "Esteva A et al. Dermatologist level classification of skin cancer with deep neural networks. Nature 542.7639 2017 115 to 118.",
        "Wang E J et al. HemaApp noninvasive blood screening of hemoglobin using smartphone cameras. UbiComp 2016.",
        "Xuan Y et al. Ultra low cost mechanical smartphone attachment for no calibration blood pressure measurement. Scientific Reports 13.1 2023 8105.",
        "HiLight Hiding Bits in Pixel Translucency Changes. MobiSys 2015.",
        "Yang J et al. MedMNIST v2 A large scale lightweight benchmark for biomedical image classification. Scientific Data 10 2023 41.",
    ]
    for index, reference in enumerate(references, 1):
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.left_indent = Inches(0.2)
        paragraph.paragraph_format.first_line_indent = Inches(-0.2)
        paragraph.paragraph_format.space_after = Pt(4)
        run = paragraph.add_run(f"{index}. {reference}")
        set_font(run, size=10)

    heading_sizes = {"Title": 18, "Heading 1": 14, "Heading 2": 12, "Heading 3": 12}
    for paragraph in doc.paragraphs:
        if paragraph.style.name in heading_sizes:
            for run in paragraph.runs:
                set_font(run, size=heading_sizes[paragraph.style.name], bold=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()

"""
report.py
Generates a clinical-style PDF report for a brain MRI prediction.
"""

import os
import uuid
import datetime
from fpdf import FPDF
import predict as P


def _pdf_safe(text):
    """
    fpdf2's core Arial/Helvetica font only supports Latin-1. Any character
    outside that range (CJK, Arabic, Cyrillic, emoji, etc.) raises
    FPDFUnicodeEncodingException and crashes report generation entirely,
    even though the actual MRI analysis succeeded. Replace unsupported
    characters rather than letting the whole report fail.
    """
    return str(text).encode('latin-1', errors='replace').decode('latin-1')


def generate_report(orig_path, overlay_path, result, save_dir,
                    patient_name='Anonymous', patient_id='N/A'):
    os.makedirs(save_dir, exist_ok=True)

    patient_name = _pdf_safe(patient_name)
    patient_id   = _pdf_safe(patient_id)

    pc   = result['pred_class']
    conf = result['confidence']
    risk = result['risk_level']
    area = result['area_pct']
    probs = result['probs']
    now  = datetime.datetime.now().strftime('%d %B %Y, %H:%M')

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Header
    pdf.set_fill_color(41, 128, 185)
    pdf.rect(0, 0, 210, 32, 'F')
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(0, 5)
    pdf.cell(210, 10, 'BRAIN TUMOUR AI DETECTION REPORT', align='C')
    pdf.set_font('Arial', '', 10)
    pdf.set_xy(0, 17)
    pdf.cell(210, 8, 'VGG16 + ResNet50 Ensemble | Research Use Only', align='C')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(22)

    # Patient info
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Patient Information', ln=True)
    pdf.set_font('Arial', '', 11)
    for label, val in [
        ('Patient Name', patient_name),
        ('Patient ID',   patient_id),
        ('Date',         now),
        ('System',       'NeuraScan AI'),
    ]:
        pdf.cell(50, 7, label + ':')
        pdf.cell(0,  7, str(val), ln=True)
    pdf.ln(2)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    # Findings
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'AI Analysis Findings', ln=True)
    pdf.set_font('Arial', '', 11)
    rc = {'High': (200, 0, 0), 'Medium': (200, 120, 0), 'Low': (0, 140, 0)}.get(risk, (0,0,0))

    findings_rows = [
        ('Prediction',   P.CLASS_DISPLAY.get(pc, pc.upper()), (0,0,0)),
        ('Confidence',   f'{conf:.1f}%',                       (0,0,0)),
        ('Risk Level',   risk,                                  rc),
    ]
    # Match result.html: only show attention area when a tumour class was
    # predicted. For "No Tumour" the metric is not meaningful to a reader
    # and looks contradictory next to a low-risk, no-tumour finding.
    if pc != 'notumor':
        findings_rows.append(('Attention Area', f'{area:.1f}% of scan', (0,0,0)))

    for label, val, col in findings_rows:
        pdf.cell(50, 8, label + ':')
        pdf.set_text_color(*col)
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, val, ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Arial', '', 11)
    pdf.ln(2)

    # Probability bars
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, 'Class Probabilities:', ln=True)
    pdf.set_font('Arial', '', 11)
    for cls in P.CLASS_LABELS:
        p  = probs.get(cls, 0)
        bw = int(p * 80)
        pdf.cell(38, 6, P.CLASS_DISPLAY.get(cls, cls) + ':')
        pdf.cell(14, 6, f'{p*100:.1f}%')
        if bw > 0:
            if cls == pc:
                pdf.set_fill_color(41, 128, 185)
            else:
                pdf.set_fill_color(190, 210, 230)
            pdf.cell(bw, 5, '', fill=True)
        pdf.ln(7)
    pdf.ln(2)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    # Images
    is_notumor = (pc == 'notumor')
    section_title = (
        'MRI Scan & Model Attention Visualisation' if is_notumor
        else 'MRI Scan & Grad-CAM Visualisation'
    )
    overlay_caption = (
        'Model Attention (No Tumour Detected)' if is_notumor
        else 'Grad-CAM Attention Overlay'
    )

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, section_title, ln=True)
    y0 = pdf.get_y()
    if os.path.exists(orig_path):
        pdf.image(orig_path,    x=12,  y=y0, w=85, h=85)
    if os.path.exists(overlay_path):
        pdf.image(overlay_path, x=112, y=y0, w=85, h=85)
    pdf.set_y(y0 + 88)
    pdf.set_font('Arial', '', 9)
    pdf.cell(95, 5, 'Original MRI Scan', align='C')
    pdf.cell(95, 5, overlay_caption, align='C', ln=True)
    if is_notumor:
        pdf.set_font('Arial', 'I', 8)
        pdf.set_text_color(120, 120, 120)
        pdf.multi_cell(0, 4.5,
            'Note: this heatmap shows the regions that most influenced the '
            '"No Tumour" prediction, not a tumour location. Highlighted '
            'areas reflect overall scan features the model associated '
            'with a normal/healthy appearance.')
        pdf.set_text_color(0, 0, 0)
    pdf.ln(3)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    # Disclaimer
    pdf.set_font('Arial', 'B', 11)
    pdf.set_text_color(180, 0, 0)
    pdf.cell(0, 7, 'MEDICAL DISCLAIMER', ln=True)
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 5,
        'This report is generated by an AI system for RESEARCH AND EDUCATIONAL '
        'PURPOSES ONLY. It is NOT a certified medical diagnosis. All findings '
        'must be reviewed by a qualified neurologist or radiologist before any '
        'medical decision is made.')

    # Footer
    pdf.set_y(-14)
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, f'NeuraScan AI | {now} | Research Use Only', align='C')

    fname = f'report_{uuid.uuid4().hex[:10]}.pdf'
    pdf.output(os.path.join(save_dir, fname))
    return fname

import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def build_pdf():
    pdf_path = "task5/derivation.pdf"
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#475569'),
        spaceAfter=15
    )
    
    h2_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=14,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=8
    )

    math_box_style = ParagraphStyle(
        'MathBox',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1E1B4B'),
        backColor=colors.HexColor('#F1F5F9'),
        borderColor=colors.HexColor('#CBD5E1'),
        borderWidth=1,
        borderPadding=8,
        spaceBefore=4,
        spaceAfter=8
    )
    
    story = []
    
    # Title & Metadata
    story.append(Paragraph("Task 5 — InfoNCE Mathematical Derivations", title_style))
    story.append(Paragraph("Vision Meets Language: Multimodal Transformer from Scratch | Seasons of Code (IIT Bombay)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#94A3B8'), spaceAfter=15))
    
    # -------------------------------------------------------------
    # Question 1
    # -------------------------------------------------------------
    story.append(Paragraph("1. InfoNCE Loss Explicit Summation & Derivation from First Principles", h2_style))
    story.append(Paragraph(
        "<b>Goal:</b> Derive the image-to-text InfoNCE loss starting from the principle of maximizing the log-probability of the matching caption under a softmax distribution over candidate captions in a batch of size <i>N</i>.",
        body_style
    ))
    story.append(Paragraph(
        "Let <i>s<sub>ij</sub></i> denote the cosine similarity between normalized image embedding <b>I</b><sub>i</sub> and text embedding <b>T</b><sub>j</sub>, and let &tau; &gt; 0 be the scalar temperature hyperparameter.<br/>"
        "Under the categorical softmax distribution, the conditional probability that text caption <i>j</i> matches image query <i>i</i> is:",
        body_style
    ))
    story.append(Paragraph(
        "<b>Softmax Probability:</b><br/>"
        "<i>p(T<sub>j</sub> | I<sub>i</sub>) = exp(s<sub>ij</sub> / &tau;) / &sum;<sub>k=1..N</sub> exp(s<sub>ik</sub> / &tau;)</i>",
        math_box_style
    ))
    story.append(Paragraph(
        "For candidate <i>i</i>, the ground-truth matching caption is <i>j = i</i>. The log-likelihood of the correct pair is:",
        body_style
    ))
    story.append(Paragraph(
        "<b>Log-Likelihood:</b><br/>"
        "log <i>p(T<sub>i</sub> | I<sub>i</sub>) = (s<sub>ii</sub> / &tau;) - log &sum;<sub>k=1..N</sub> exp(s<sub>ik</sub> / &tau;)</i>",
        math_box_style
    ))
    story.append(Paragraph(
        "The image-to-text InfoNCE loss &Lscript;<sub>i2t</sub> is defined as the negative expected log-likelihood across all <i>N</i> batch items:",
        body_style
    ))
    story.append(Paragraph(
        "<b>Explicit Summation Formula:</b><br/>"
        "&Lscript;<sub>i2t</sub> = - (1 / N) &sum;<sub>i=1..N</sub> log <i>p(T<sub>i</sub> | I<sub>i</sub>)</i><br/>"
        "&Lscript;<sub>i2t</sub> = (1 / N) &sum;<sub>i=1..N</sub> [ - (s<sub>ii</sub> / &tau;) + log &sum;<sub>k=1..N</sub> exp(s<sub>ik</sub> / &tau;) ]",
        math_box_style
    ))
    
    # -------------------------------------------------------------
    # Question 2
    # -------------------------------------------------------------
    story.append(Paragraph("2. Temperature Limiting Behavior (&tau; &rarr; 0 and &tau; &rarr; &infin;)", h2_style))
    story.append(Paragraph(
        "<b>Case A: Limit as &tau; &rarr; 0 (Zero Temperature / Sharp Softmax)</b><br/>"
        "As &tau; approaches 0, the exponential term with the largest similarity <i>s<sub>i,max</sub> = max<sub>k</sub> s<sub>ik</sub></i> completely dominates the log-sum-exp term:<br/>"
        "&tau; &middot; log &sum;<sub>k=1..N</sub> exp(s<sub>ik</sub> / &tau;) &rarr; max<sub>k</sub> s<sub>ik</sub>.<br/>"
        "Multiplying &Lscript;<sub>i2t</sub> by &tau; yields:<br/>"
        "lim<sub>&tau; &rarr; 0</sub> &tau; &middot; &Lscript;<sub>i2t</sub> = (1 / N) &sum;<sub>i=1..N</sub> ( max<sub>k</sub> s<sub>ik</sub> - s<sub>ii</sub> ).<br/>"
        "To achieve zero loss, we require <i>max<sub>k</sub> s<sub>ik</sub> = s<sub>ii</sub></i>, implying <i>s<sub>ii</sub> &gt; s<sub>ik</sub></i> for all <i>k &ne; i</i>. "
        "This acts like a hard max-margin constraint: the correct pair similarity must strictly exceed all negative candidate similarities.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Case B: Limit as &tau; &rarr; &infin; (Infinite Temperature / Flat Softmax)</b><br/>"
        "As &tau; &rarr; &infin;, all scaled similarities <i>s<sub>ik</sub> / &tau; &rarr; 0</i>. Therefore exp(s<sub>ik</sub> / &tau;) &rarr; 1 for all <i>k</i>.<br/>"
        "The log-sum term becomes log(&sum;<sub>k=1..N</sub> 1) = log(N).<br/>"
        "Substituting this back into the loss:<br/>"
        "lim<sub>&tau; &rarr; &infin;</sub> &Lscript;<sub>i2t</sub> = - (1 / N) &sum;<sub>i=1..N</sub> ( 0 - log N ) = <b>log N</b>.<br/>"
        "<b>Intuition:</b> At infinite temperature, the softmax distribution flattens into a uniform distribution over <i>N</i> classes where <i>p = 1/N</i>. The model retains zero discriminative power, collapsing the loss to maximum uncertainty log(N).",
        body_style
    ))

    # -------------------------------------------------------------
    # Question 3
    # -------------------------------------------------------------
    story.append(Paragraph("3. Analytical Gradients & Physical Interpretation of Signs", h2_style))
    story.append(Paragraph(
        "Let <i>P<sub>ij</sub> = exp(s<sub>ij</sub> / &tau;) / &sum;<sub>k=1..N</sub> exp(s<sub>ik</sub> / &tau;)</i> denote the softmax weight assigned to pair (i, j).",
        body_style
    ))
    story.append(Paragraph(
        "<b>Gradient w.r.t Correct Pair Similarity (s<sub>ii</sub>):</b><br/>"
        "&part;&Lscript;<sub>i2t</sub> / &part;s<sub>ii</sub> = (1 / N &tau;) &middot; ( P<sub>ii</sub> - 1 ) &nbsp;&le;&nbsp; 0",
        math_box_style
    ))
    story.append(Paragraph(
        "<b>Gradient w.r.t Incorrect Pair Similarity (s<sub>ij</sub> for j &ne; i):</b><br/>"
        "&part;&Lscript;<sub>i2t</sub> / &part;s<sub>ij</sub> = (1 / N &tau;) &middot; P<sub>ij</sub> &nbsp;&gt;&nbsp; 0",
        math_box_style
    ))
    story.append(Paragraph(
        "<b>Interpretation of Gradient Signs:</b><br/>"
        "• <b>Correct Pair (s<sub>ii</sub>):</b> The gradient &part;&Lscript; / &part;s<sub>ii</sub> is strictly negative (since <i>P<sub>ii</sub> &le; 1</i>). Under gradient descent (<i>s<sub>ii</sub> &larr; s<sub>ii</sub> - &eta; &nabla;</i>), subtracting a negative quantity <b>increases s<sub>ii</sub></b>. This pulls matching image and text embeddings closer together.<br/>"
        "• <b>Incorrect Pair (s<sub>ij</sub>):</b> The gradient &part;&Lscript; / &part;s<sub>ij</sub> is strictly positive (since <i>P<sub>ij</sub> &gt; 0</i>). Under gradient descent, subtracting a positive gradient <b>decreases s<sub>ij</sub></b>. This pushes mismatched image and text embeddings apart in the joint embedding space.",
        body_style
    ))
    
    doc.build(story)
    print(f"Successfully generated {pdf_path}")

if __name__ == '__main__':
    build_pdf()

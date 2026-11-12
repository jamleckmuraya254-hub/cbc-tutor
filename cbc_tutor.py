import hashlib
import io
import json
import re
import time
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types

# ---------- Settings ----------
MODEL = "gemini-3.5-flash"   # If Google renames models, change it here only
MAX_HISTORY = 20             # messages sent back to the model each turn
CURRICULUM_DIR = Path(__file__).parent / "curriculum"   # put KICD PDFs here
FILE_TTL = 60 * 60 * 40      # Gemini keeps uploaded files 48h; refresh at 40h

# Approximate CBC performance bands (percent). Edit to match your school's rubric.
LEVEL_BANDS = [
    (75, "Exceeds Expectation"),
    (41, "Meets Expectation"),
    (21, "Approaches Expectation"),
    (0, "Below Expectation"),
]

st.set_page_config(page_title="CBC AI Tutor Kenya", page_icon="📚", layout="centered")

# ---------- Optional password (set APP_PASSWORD in secrets to turn on) ----------
if "APP_PASSWORD" in st.secrets:
    if not st.session_state.get("authed"):
        pw = st.text_input("Enter access password", type="password")
        if pw and pw == st.secrets["APP_PASSWORD"]:
            st.session_state.authed = True
            st.rerun()
        st.stop()


@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


client = get_client()

# ---------- CBC structure ----------
RE = "Religious Education (CRE/IRE/HRE)"

LEARNING_AREAS = {
    "Playgroup (age 3)": [
        "Language Activities", "Mathematical Activities", "Environmental Activities",
        "Psychomotor & Creative Activities", "Religious Activities",
    ],
    "PP1 (age 4)": [
        "Language Activities", "Mathematical Activities", "Environmental Activities",
        "Psychomotor & Creative Activities", "Religious Education Activities",
    ],
    "PP2 (age 5)": [
        "Language Activities", "Mathematical Activities", "Environmental Activities",
        "Psychomotor & Creative Activities", "Religious Education Activities",
    ],
    "Lower Primary (Grade 1-3)": [
        "English (Literacy)", "Kiswahili (Kusoma na Kuandika)", "Indigenous Language",
        "Mathematical Activities", "Environmental Activities", "Hygiene & Nutrition",
        RE, "Movement & Creative Activities",
    ],
    "Upper Primary (Grade 4-6)": [
        "English", "Kiswahili", "Mathematics", "Science & Technology", "Social Studies",
        "Agriculture & Nutrition", RE, "Creative Arts", "Physical & Health Education",
    ],
    "Junior Secondary (Grade 7-9)": [
        "English", "Kiswahili", "Mathematics", "Integrated Science", "Social Studies",
        "Pre-Technical Studies", "Agriculture & Nutrition", RE,
        "Creative Arts & Sports", "Business Studies", "Computer Science",
    ],
    "Senior Secondary (Grade 10-12)": [
        "English", "Kiswahili", "Core Mathematics", "Community Service Learning",
        "Physical Education",
        "STEM: Biology", "STEM: Chemistry", "STEM: Physics", "STEM: Advanced Mathematics",
        "STEM: Computer Studies", "STEM: Agriculture", "STEM: Home Science",
        "Social Sciences: Geography", "Social Sciences: History & Citizenship",
        "Social Sciences: Business Studies", "Social Sciences: Literature in English",
        "Social Sciences: " + RE,
        "Arts & Sports: Music & Dance", "Arts & Sports: Fine Arts",
        "Arts & Sports: Theatre & Film", "Arts & Sports: Sports & Recreation",
    ],
}

MODES = {
    "Explain a topic": "Explain the topic simply with Kenyan examples, then check understanding with 3 short questions.",
    "Lesson plan": "Write a full CBC lesson plan: strand, sub-strand, specific learning outcomes, key inquiry question, core competencies, values, PCIs, learning resources, learning experiences (introduction, development, conclusion), and assessment.",
    "Notes for learners": "Write clear, well-organised learner notes with headings, key terms, examples and a short summary.",
    "Quiz / practice questions": "Create a quiz with an answer key and a CBC-style rubric (Exceeds, Meets, Approaches, Below Expectations).",
    "Activities & projects": "Suggest hands-on learning activities, group work, and a small project using locally available materials.",
    "Scheme of work": "Create a term scheme of work table: week, lesson, strand/sub-strand, outcomes, activities, resources, assessment, remarks.",
    "Homework help": "Guide the learner step by step. Do not just give the final answer; ask guiding questions and let them try.",
}

LANGUAGES = ["English", "Kiswahili", "Simple English + Kiswahili terms"]
AUDIENCES = ["Learner", "Teacher", "Parent"]

DOCS_RULE = """
Official curriculum documents (KICD curriculum designs) are attached. Base strands, sub-strands, specific learning outcomes, suggested learning experiences, key inquiry questions and assessment on them, and mention the section or page you used. If the topic is not covered in the documents, say so clearly before using general knowledge."""


def build_system_prompt(level, area, mode, language, audience, has_docs=False):
    return f"""You are a friendly, patient AI tutor for Kenya's Competency Based Curriculum (CBC), aligned to KICD curriculum designs.

Context chosen by the user:
- Level: {level}
- Learning area: {area}
- Task type: {mode}
- Audience: {audience}
- Response language: {language}

Task guidance: {MODES[mode]}

Rules:
- Match vocabulary, sentence length and depth to the level. For Playgroup, PP1 and PP2 content must be play-based, using songs, stories, movement, counting objects, and short activities. Address teachers or parents, not the child directly, unless the audience is Learner.
- Use CBC language: strands, sub-strands, specific learning outcomes, learning experiences, key inquiry questions, the 7 core competencies (communication and collaboration, critical thinking and problem solving, creativity and imagination, citizenship, digital literacy, learning to learn, self-efficacy), values, and Pertinent and Contemporary Issues.
- Use Kenyan context: shillings, local foods, matatus, markets, farms, local places, familiar names.
- Emphasise doing, applying and competence, not memorising.
- Assessment uses the four CBC levels: Exceeds Expectation, Meets Expectation, Approaches Expectation, Below Expectation.
- If you are unsure of the exact official sub-strand wording or grade placement, say so and advise checking the KICD curriculum design.
- Keep content safe, age-appropriate and respectful of Kenya's diverse cultures and religions.
- Format with headings, short paragraphs and bullet points. Be accurate and say so when you are unsure.
{DOCS_RULE if has_docs else ""}"""


# ---------- Curriculum PDFs (Gemini Files API) ----------
def wait_active(f):
    while getattr(f.state, "name", "") == "PROCESSING":
        time.sleep(2)
        f = client.files.get(name=f.name)
    return f


@st.cache_resource(ttl=FILE_TTL, show_spinner="Loading curriculum document...")
def upload_repo_pdf(name):
    path = CURRICULUM_DIR / name
    f = client.files.upload(
        file=str(path),
        config=types.UploadFileConfig(mime_type="application/pdf", display_name=name),
    )
    return wait_active(f)


@st.cache_resource(ttl=FILE_TTL, show_spinner="Uploading your document...")
def upload_bytes(name, data):
    f = client.files.upload(
        file=io.BytesIO(data),
        config=types.UploadFileConfig(mime_type="application/pdf", display_name=name),
    )
    return wait_active(f)


def file_parts(files):
    return [types.Part.from_uri(file_uri=f.uri, mime_type=f.mime_type) for f in files]


# ---------- Chat helpers ----------
def to_contents(messages, files):
    recent = messages[-MAX_HISTORY:]
    while recent and recent[0]["role"] == "model":   # history must start with user
        recent = recent[1:]
    contents = [
        types.Content(role=m["role"], parts=[types.Part.from_text(text=m["text"])])
        for m in recent
    ]
    if files and contents:
        contents[-1].parts = file_parts(files) + list(contents[-1].parts)
    return contents


def stream_reply(system_prompt, files):
    stream = client.models.generate_content_stream(
        model=MODEL,
        contents=to_contents(st.session_state.messages, files),
        config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.7),
    )
    for chunk in stream:
        if chunk.text:
            yield chunk.text


def transcribe(data):
    r = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=data, mime_type="audio/wav"),
            "Transcribe this audio exactly. The speaker may use English, Kiswahili or both. "
            "Return only the transcript, nothing else.",
        ],
    )
    return (r.text or "").strip()


# ---------- Quiz helpers ----------
def make_quiz(topic, n, system_prompt, files):
    prompt = (
        f'Create {n} multiple-choice questions on "{topic}" for this level and learning area. '
        'Return ONLY a JSON list. Each item must be: {"question": string, '
        '"options": [exactly 4 short strings without A/B/C/D letters], '
        '"answer": index 0-3 of the correct option, "explanation": 1-2 sentences}. '
        "Vary the position of the correct answer. Test understanding and application, not just recall."
    )
    resp = client.models.generate_content(
        model=MODEL,
        contents=file_parts(files) + [prompt],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            temperature=0.5,
        ),
    )
    items = json.loads(resp.text)
    return [
        q for q in items
        if isinstance(q, dict) and len(q.get("options", [])) == 4 and q.get("answer") in range(4)
    ]


def cbc_level(pct):
    for floor, name in LEVEL_BANDS:
        if pct >= floor:
            return name


# ---------- Export helpers (Word / PDF) ----------
SYM = {"≥": ">=", "≤": "<=", "≠": "!=", "≈": "~", "√": "sqrt", "π": "pi", "→": "->",
       "−": "-", "∞": "infinity", "∠": "angle ", "Δ": "delta "}
SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def parse_blocks(md):
    blocks, table = [], []
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(re.fullmatch(r":?-+:?", c) for c in cells):
                continue
            table.append(cells)
            continue
        if table:
            blocks.append(("table", table))
            table = []
        if not s:
            continue
        m = re.match(r"(#{1,4})\s+(.*)", s)
        if m:
            blocks.append(("h", len(m.group(1)), m.group(2)))
            continue
        m = re.match(r"[-*•]\s+(.*)", s)
        if m:
            blocks.append(("li", m.group(1)))
            continue
        blocks.append(("p", s))
    if table:
        blocks.append(("table", table))
    return blocks


def add_runs(par, text, bold_all=False):
    text = text.replace("<br>", "\n").replace("<br/>", "\n")
    for i, part in enumerate(text.split("**")):
        if part:
            par.add_run(part.replace("*", "")).bold = bold_all or (i % 2 == 1)


def to_docx(md, title):
    from docx import Document

    d = Document()
    d.add_heading(title, 0)
    for b in parse_blocks(md):
        if b[0] == "h":
            d.add_heading(b[2].replace("**", ""), min(b[1], 4))
        elif b[0] == "li":
            add_runs(d.add_paragraph(style="List Bullet"), b[1])
        elif b[0] == "table":
            n = max(len(r) for r in b[1])
            t = d.add_table(rows=len(b[1]), cols=n)
            t.style = "Table Grid"
            for i, row in enumerate(b[1]):
                for j, c in enumerate(row):
                    add_runs(t.cell(i, j).paragraphs[0], c, bold_all=(i == 0))
            d.add_paragraph()
        else:
            add_runs(d.add_paragraph(), b[1])
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def pdf_clean(t):
    t = t.replace("<br>", "\n").replace("<br/>", "\n")
    for a, b in SYM.items():
        t = t.replace(a, b)
    t = t.translate(SUB).encode("cp1252", "replace").decode("cp1252")
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t).replace("*", "")
    return t.replace("\n", "<br/>")


def to_pdf(md, title):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=ss["BodyText"], fontSize=8, leading=10)
    li = ParagraphStyle("li", parent=ss["BodyText"], leftIndent=16, bulletIndent=6)
    story = [Paragraph(pdf_clean(title), ss["Title"])]
    for b in parse_blocks(md):
        if b[0] == "h":
            story += [Spacer(1, 6), Paragraph(pdf_clean(b[2]), ss[f"Heading{min(b[1], 3)}"])]
        elif b[0] == "li":
            story.append(Paragraph(pdf_clean(b[1]), li, bulletText="•"))
        elif b[0] == "table":
            n = max(len(r) for r in b[1])
            w = (A4[0] - 72) / n
            data = [[Paragraph(pdf_clean(c), small) for c in r + [""] * (n - len(r))] for r in b[1]]
            t = Table(data, colWidths=[w] * n, repeatRows=1)
            t.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story += [t, Spacer(1, 6)]
        else:
            story.append(Paragraph(pdf_clean(b[1]), ss["BodyText"]))
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=36, rightMargin=36,
                      topMargin=36, bottomMargin=36, title=title).build(story)
    return buf.getvalue()


def export_buttons(md, title, key):
    c1, c2, c3 = st.columns(3)
    try:
        c1.download_button("⬇️ Word", to_docx(md, title), "cbc_content.docx", key=f"w{key}")
    except Exception as e:
        c1.caption(f"Word export failed: {e}")
    try:
        c2.download_button("⬇️ PDF", to_pdf(md, title), "cbc_content.pdf", key=f"p{key}")
    except Exception as e:
        c2.caption(f"PDF export failed: {e}")
    c3.download_button("⬇️ Text", md, "cbc_content.md", key=f"m{key}")


# ---------- Sidebar ----------
selected_files = []
with st.sidebar:
    st.header("Learning setup")
    level = st.selectbox("Level", list(LEARNING_AREAS.keys()), index=4)
    area = st.selectbox("Learning area", LEARNING_AREAS[level])
    mode = st.selectbox("What do you need?", list(MODES.keys()))
    audience = st.radio("I am a", AUDIENCES, horizontal=True)
    language = st.selectbox("Language", LANGUAGES)

    with st.expander("📖 Curriculum documents (KICD)"):
        st.caption("Pick 1-3 documents for the level you teach. The tutor will then answer from them.")
        repo_pdfs = sorted(p.name for p in CURRICULUM_DIR.glob("*.pdf")) if CURRICULUM_DIR.exists() else []
        chosen = st.multiselect("From the app's library", repo_pdfs, max_selections=3)
        uploads = st.file_uploader("Or upload a PDF", type="pdf", accept_multiple_files=True)
        try:
            for name in chosen:
                selected_files.append(upload_repo_pdf(name))
            for u in uploads or []:
                selected_files.append(upload_bytes(u.name, u.getvalue()))
        except Exception as e:
            st.error(f"Could not load a document: {e}")
        if selected_files:
            st.success(f"{len(selected_files)} document(s) active")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.caption("Tip: never type learners' full names or personal details.")

# ---------- Main page ----------
st.title("📚 CBC AI Tutor Kenya")
st.caption(f"{level} · {area}")
view = st.radio("View", ["💬 Tutor", "📝 Quiz"], horizontal=True, label_visibility="collapsed")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ===== Tutor view =====
if view == "💬 Tutor":
    voice_prompt = None
    with st.expander("🎤 Ask by voice"):
        audio = st.audio_input("Record your question", label_visibility="collapsed")
        if audio is not None:
            data = audio.getvalue()
            h = hashlib.md5(data).hexdigest()
            if h != st.session_state.get("last_audio"):
                st.session_state.last_audio = h
                try:
                    with st.spinner("Listening..."):
                        voice_prompt = transcribe(data) or None
                except Exception as e:
                    st.error(f"Could not understand the audio: {e}")

    if not st.session_state.messages:
        st.info("Pick your level and learning area in the sidebar (top-left arrow), then type or speak a topic. Example: 'Fractions' or 'Parts of a plant'.")

    for m in st.session_state.messages:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.markdown(m["text"])

    typed = st.chat_input("Type a topic or question...")
    prompt = typed or voice_prompt
    if prompt:
        st.session_state.messages.append({"role": "user", "text": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        system_prompt = build_system_prompt(level, area, mode, language, audience, bool(selected_files))
        with st.chat_message("assistant"):
            try:
                reply = st.write_stream(stream_reply(system_prompt, selected_files))
                st.session_state.messages.append({"role": "model", "text": reply})
            except Exception as e:
                st.session_state.messages.pop()  # remove the unanswered question
                st.error(f"Sorry, something went wrong: {e}")

    last = next((m for m in reversed(st.session_state.messages) if m["role"] == "model"), None)
    if last:
        export_buttons(last["text"], f"{area} - {level}", len(st.session_state.messages))

# ===== Quiz view =====
else:
    if level.startswith(("Playgroup", "PP")):
        st.info("Quizzes work best from Grade 1 upwards. For PP levels, use the Tutor tab for play-based activities.")
    topic = st.text_input("Quiz topic", placeholder="e.g. Fractions, Parts of a plant")
    n = st.select_slider("Number of questions", options=[5, 10], value=5)

    if st.button("Generate quiz"):
        if not topic.strip():
            st.warning("Type a topic first.")
        else:
            sys_prompt = build_system_prompt(level, area, "Quiz / practice questions", language, audience, bool(selected_files))
            try:
                with st.spinner("Writing your quiz..."):
                    items = make_quiz(topic.strip(), n, sys_prompt, selected_files)
                if items:
                    st.session_state.quiz = {"id": int(time.time() * 1000), "items": items, "answers": None}
                else:
                    st.error("The quiz came back empty. Please try again.")
            except Exception as e:
                st.error(f"Could not create the quiz: {e}")

    quiz = st.session_state.get("quiz")
    if quiz:
        with st.form(f"quizform{quiz['id']}"):
            picks = [
                st.radio(f"{i + 1}. {q['question']}", q["options"], index=None, key=f"q{quiz['id']}_{i}")
                for i, q in enumerate(quiz["items"])
            ]
            if st.form_submit_button("Submit answers"):
                quiz["answers"] = picks

        if quiz["answers"] is not None:
            items, given = quiz["items"], quiz["answers"]
            score = sum(1 for q, a in zip(items, given) if a == q["options"][q["answer"]])
            pct = round(100 * score / len(items))
            st.subheader(f"Score: {score}/{len(items)} ({pct}%)")
            st.write(f"**CBC lev

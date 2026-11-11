import streamlit as st
from google import genai
from google.genai import types

# ---------- Settings ----------
MODEL = "gemini-3.5-flash"  # If Google renames models, change it here only
MAX_HISTORY = 20            # messages sent back to the model each turn

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


def build_system_prompt(level, area, mode, language, audience):
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
"""


def to_contents(messages):
    return [
        types.Content(role=m["role"], parts=[types.Part.from_text(text=m["text"])])
        for m in messages[-MAX_HISTORY:]
    ]


def stream_reply(system_prompt):
    stream = client.models.generate_content_stream(
        model=MODEL,
        contents=to_contents(st.session_state.messages),
        config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.7),
    )
    for chunk in stream:
        if chunk.text:
            yield chunk.text


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Learning setup")
    level = st.selectbox("Level", list(LEARNING_AREAS.keys()), index=4)
    area = st.selectbox("Learning area", LEARNING_AREAS[level])
    mode = st.selectbox("What do you need?", list(MODES.keys()))
    audience = st.radio("I am a", AUDIENCES, horizontal=True)
    language = st.selectbox("Language", LANGUAGES)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("Tip: never type learners' full names or personal details.")

# ---------- Main page ----------
st.title("📚 CBC AI Tutor Kenya")
st.caption(f"{level} · {area} · {mode}")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info("Pick your level and learning area in the sidebar (top-left arrow), then type a topic below. Example: 'Fractions' or 'Parts of a plant'.")

for m in st.session_state.messages:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.markdown(m["text"])

if prompt := st.chat_input("Type a topic or question..."):
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    system_prompt = build_system_prompt(level, area, mode, language, audience)
    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(stream_reply(system_prompt))
            st.session_state.messages.append({"role": "model", "text": reply})
        except Exception as e:
            st.session_state.messages.pop()  # remove the unanswered question
            st.error(f"Sorry, something went wrong: {e}")

# ---------- Download last answer ----------
last = next((m for m in reversed(st.session_state.messages) if m["role"] == "model"), None)
if last:
    st.download_button("⬇️ Download last answer", last["text"], file_name="cbc_content.md")

st.divider()
st.caption("AI-generated content. Teachers should check against official KICD curriculum designs before use.")

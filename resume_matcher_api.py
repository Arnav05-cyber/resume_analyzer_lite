import os
import fitz  # PyMuPDF
import docx
import spacy
from spacy.matcher import Matcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Import CORS

# --- 1. Load NLP Models and Define Skills ---

print("✅ Loading spaCy model...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("❌ spaCy model not found. Please run: python -m spacy download en_core_web_sm")
    exit()
print("✅ spaCy model loaded.")

print("✅ Loading SentenceTransformer model...")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ SentenceTransformer model loaded.")

SKILL_LIST = [  # (unchanged, keeping your full list)
    # ... your entire skill list unchanged ...
    'communication', 'teamwork', 'leadership', 'problem solving', 'critical thinking',
    'analytical skills', 'collaboration', 'time management', 'adaptability'
]

# --- 2. Define Helper Functions ---

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_skills(text):
    doc = nlp(str(text).lower())
    matcher = Matcher(nlp.vocab)
    for skill in SKILL_LIST:
        pattern = [{"LOWER": word} for word in skill.split()]
        matcher.add(skill, [pattern])
    matches = matcher(doc)
    found_skills = {doc[start:end].text for _, start, end in matches}
    return list(found_skills)

def calculate_semantic_match_score(resume_text, job_description_text):
    embeddings = semantic_model.encode([resume_text, job_description_text])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(score)

def generate_suggestions(resume_skills, job_skills):
    resume_skills_set = set(resume_skills)
    job_skills_set = set(job_skills)
    missing_skills = list(job_skills_set - resume_skills_set)

    if not missing_skills:
        return "Excellent match! Your resume contains all the key skills mentioned in the job description."

    suggestions = "To improve your match score, consider highlighting the following skills from the job description if you have relevant experience:\n"
    for skill in missing_skills:
        suggestions += f"- {skill.title()}\n"

    suggestions += "\nTip: Make sure to mention these skills in the context of your projects or work experience to demonstrate your expertise."
    return suggestions

# --- 3. Flask App ---

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Resume Matcher API is live!"  # ✅ Optional root endpoint

@app.route('/match', methods=['POST'])
def match_resume_jd():
    try:
        resume_file = request.files['resume']
        jd_file = request.files.get('job_description')

        if not resume_file or not jd_file:
            return jsonify({"error": "Both resume and job_description files are required."}), 400

        temp_path = f"temp_resume.{resume_file.filename.split('.')[-1]}"
        resume_file.save(temp_path)

        if temp_path.endswith('.pdf'):
            resume_text = extract_text_from_pdf(temp_path)
        elif temp_path.endswith('.docx'):
            resume_text = extract_text_from_docx(temp_path)
        else:
            return jsonify({"error": "Unsupported resume file type"}), 400

        os.remove(temp_path)

        if jd_file.filename.endswith('.txt'):
            jd_text = jd_file.read().decode('utf-8')
        else:
            return jsonify({"error": "Job description must be a .txt file"}), 400

        score = calculate_semantic_match_score(resume_text, jd_text)
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(jd_text)
        suggestions = generate_suggestions(resume_skills, jd_skills)

        return jsonify({
            "score": round(score * 100, 2),
            "resume_skills": resume_skills,
            "jd_skills": jd_skills,
            "suggestions": suggestions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Required for Render to detect port
if __name__ == "_main_":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting app on port {port}...")
    app.run(host="0.0.0.0", port=port)
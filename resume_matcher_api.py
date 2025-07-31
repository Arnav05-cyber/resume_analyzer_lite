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

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
    exit()

print("Loading Sentence Transformer model...")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully.")

SKILL_LIST = [
    # --- Technical & Programming ---
    'python', 'java', 'javascript', 'c++', 'sql', 'r', 'go', 'php', 'ruby', 'swift', 'kotlin',
    'html', 'css', 'jquery', 'typescript', 'bash', 'powershell',
    
    # --- Web Development & Frameworks ---
    'react', 'angular', 'vue', 'node.js', 'express.js', 'django', 'flask', 'laravel', 
    'ruby on rails', 'asp.net', 'spring boot', 'bootstrap', 'tailwind css',
    
    # --- Databases ---
    'mysql', 'postgresql', 'mongodb', 'sql server', 'oracle', 'sqlite', 'redis',
    'cassandra', 'hbase', 'elasticsearch', 'dynamodb',
    
    # --- Data Science & Machine Learning ---
    'machine learning', 'deep learning', 'data science', 'data analysis', 'nlp',
    'natural language processing', 'computer vision', 'data visualization',
    'tensorflow', 'keras', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'matplotlib',
    'seaborn', 'scipy', 'opencv', 'tableau', 'power bi', 'kibana', 'ggplot', 'd3.js',
    'statistics', 'statistical modeling', 'a/b testing', 'econometrics',
    'regression', 'classification', 'clustering', 'svm', 'naive bayes', 'knn', 'random forest', 
    'decision trees', 'gradient boosting', 'xgboost', 'lightgbm', 'cluster analysis', 
    'word embedding', 'sentiment analysis', 'dimensionality reduction', 'topic modelling', 
    'lda', 'nmf', 'pca', 'neural networks', 'rnn', 'lstm', 'transformer',
    
    # --- Big Data & Data Engineering ---
    'hadoop', 'spark', 'apache spark', 'hive', 'pig', 'etl', 'data warehousing',
    'informatica', 'data stage', 'airflow', 'kafka',
    
    # --- Cloud & DevOps ---
    'aws', 'azure', 'google cloud', 'gcp', 'docker', 'kubernetes', 'git', 'github', 
    'gitlab', 'ci/cd', 'terraform', 'ansible', 'serverless',
    
    # --- Business & Finance ---
    'business analysis', 'requirement gathering', 'business intelligence', 'financial analysis',
    'financial modeling', 'accounting', 'auditing', 'sap', 'erp', 'excel',
    'project management', 'agile', 'scrum', 'kanban', 'jira', 'product management',
    'operations management', 'supply chain', 'logistics', 'pmp', 'six sigma',
    
    # --- Sales & Marketing ---
    'sales', 'business development', 'marketing', 'digital marketing', 'seo', 'sem',
    'content marketing', 'social media marketing', 'email marketing', 'google analytics',
    'crm', 'salesforce', 'negotiation', 'lead generation',
    
    # --- HR & Recruiting ---
    'human resources', 'recruiting', 'talent acquisition', 'screening', 'interviewing',
    'employee relations', 'employee engagement', 'hris', 'onboarding', 'hr',
    
    # --- Design & Creative ---
    'ui', 'ux', 'ui/ux', 'design thinking', 'figma', 'sketch', 'adobe xd',
    'adobe photoshop', 'illustrator', 'indesign', 'graphic design', 'motion graphics',
    'video editing', 'after effects', 'premiere pro',
    
    # --- Engineering (Non-Software) ---
    'mechanical engineering', 'autocad', 'solidworks', 'catia', 'civil engineering',
    'staad pro', 'electrical engineering', 'matlab',
    
    # --- Other Professional Skills ---
    'testing', 'quality assurance', 'qa', 'automation testing', 'manual testing', 'selenium', 'qtp',
    'blockchain', 'solidity', 'ethereum', 'advocate', 'legal', 'drafting', 'litigation',
    'health', 'fitness', 'nutrition', 'training', 'yoga',
    
    # --- Soft Skills (Crucial for all roles) ---
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
    return float(score)  # 👈 FIX: Ensure score is a native Python float

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
CORS(app)  # ✅ Allow all origins (for development)


@app.route('/match', methods=['POST'])
def match_resume_jd():
    try:
        resume_file = request.files['resume']
        jd_file = request.files.get('job_description')

        if not resume_file or not jd_file:
            return jsonify({"error": "Both resume and job_description files are required."}), 400

        # Save resume temporarily
        temp_path = f"temp_resume.{resume_file.filename.split('.')[-1]}"
        resume_file.save(temp_path)

        # Extract resume text
        if temp_path.endswith('.pdf'):
            resume_text = extract_text_from_pdf(temp_path)
        elif temp_path.endswith('.docx'):
            resume_text = extract_text_from_docx(temp_path)
        else:
            return jsonify({"error": "Unsupported resume file type"}), 400

        # Clean up
        os.remove(temp_path)

        # Read job description from txt
        if jd_file.filename.endswith('.txt'):
            jd_text = jd_file.read().decode('utf-8')
        else:
            return jsonify({"error": "Job description must be a .txt file"}), 400

        # Match logic
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
    
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

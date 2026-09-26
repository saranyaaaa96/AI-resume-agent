# ==============================================================================
# RESUME AGENT - AI Resume Analyzer & Career Coach (< 300 Lines)
# Single file: Python, LangChain, LangGraph, Dynamic NLP Analysis, and Flask
# ==============================================================================
import os, io, json, re
from typing import TypedDict, Dict, Any
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string
import pypdf
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

load_dotenv()
app = Flask(__name__)
API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

CAREER_KNOWLEDGE = {
    "Machine Learning Engineer": ["Python", "SQL", "PyTorch", "TensorFlow", "Scikit-Learn", "Docker", "FastAPI", "MLOps", "Git"],
    "Data Scientist": ["Python", "SQL", "Pandas", "NumPy", "Statistics", "Machine Learning", "Data Visualization", "Git"],
    "Full Stack Developer": ["JavaScript", "TypeScript", "React", "Node.js", "SQL", "REST APIs", "Docker", "Git", "HTML/CSS"],
    "DevOps / Cloud": ["Linux", "Docker", "Kubernetes", "AWS", "CI/CD", "Terraform", "Python", "Git", "Bash"],
    "AI Engineer": ["Python", "LLMs", "LangChain", "PyTorch", "FastAPI", "Docker", "Vector DBs", "Git"]
}
ALL_SKILLS = ["python", "sql", "pytorch", "tensorflow", "scikit-learn", "docker", "fastapi", "mlops", "git", "pandas",
              "numpy", "statistics", "machine learning", "javascript", "typescript", "react", "node.js", "rest apis",
              "linux", "kubernetes", "aws", "ci/cd", "terraform", "llms", "langchain", "vector dbs", "java", "c++", "c#", "go"]

@tool
def calculate_score(skills_count: int, has_projects: bool, role_match_pct: int) -> int:
    """Calculates AI estimated resume score (0-100)."""
    return min(95, max(45, 45 + min(20, skills_count * 2) + (15 if has_projects else 5) + int(role_match_pct * 0.2)))

def extract_text(file_storage) -> str:
    if not file_storage or not file_storage.filename: return ""
    try:
        if file_storage.filename.lower().endswith(".txt"):
            return file_storage.read().decode("utf-8", errors="ignore").strip()
        reader = pypdf.PdfReader(file_storage)
        return "\n".join([p.extract_text() or "" for p in reader.pages]).strip()
    except Exception:
        file_storage.seek(0)
        return file_storage.read().decode("utf-8", errors="ignore").strip()

class ResumeState(TypedDict):
    resume_text: str
    target_role: str
    analysis: Dict[str, Any]

def parse_resume_node(state: ResumeState): return {"resume_text": state["resume_text"].strip()}

def analyze_resume_node(state: ResumeState):
    text, role = state["resume_text"], state["target_role"]
    if API_KEY:
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY, temperature=0.2)
            prompt = f"Analyze actual resume text for '{role}':\n{text[:3500]}\nReturn STRICT JSON ONLY (no markdown ticks):\n{{\"score\": <0-100 score>, \"match_pct\": <0-100 match>, \"strong_matches\": [\"✓ <found skill>\"], \"missing_skills\": [\"⚠ <missing skill>\"], \"recommendation\": \"<recommendation>\", \"improvements\": [\"<actionable bullet improvement>\"], \"roadmap\": [\"Step 1: ...\", \"Step 2: ...\", \"Step 3: ...\", \"Step 4: ...\"]}}"
            raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
            raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
            return {"analysis": json.loads(raw)}
        except Exception: pass

    # Dynamic local analysis from actual resume text
    low = text.lower()
    found = [s.title() for s in ALL_SKILLS if re.search(r'\b' + re.escape(s) + r'\b', low)]
    expected = CAREER_KNOWLEDGE.get(role, [r.strip() for r in role.split() if len(r)>2] + ["Git", "Docker", "SQL"])
    strong = [f"✓ {s}" for s in expected if any(s.lower() == f.lower() for f in found)] or ([f"✓ {s}" for s in found[:4]] if found else ["✓ Foundational Experience"])
    missing = [f"⚠ {s}" for s in expected if not any(s.lower() == f.lower() for f in found)] or ["⚠ Advanced Cloud/MLOps"]
    match_pct = int((len(strong) / max(len(expected), 1)) * 100)
    has_proj = any(w in low for w in ["project", "github", "portfolio", "built", "implemented", "developed"])
    score = calculate_score.invoke({"skills_count": len(found), "has_projects": has_proj, "role_match_pct": match_pct})
    m_names = [m.replace("⚠ ", "") for m in missing]
    return {"analysis": {
        "score": score, "match_pct": match_pct, "strong_matches": strong, "missing_skills": missing[:5],
        "recommendation": f"To optimize your profile for {role}, focus on building hands-on projects featuring {', '.join(m_names[:2]) if m_names else 'cloud deployment'}.",
        "improvements": [
            "Quantify bullet points with metrics (e.g. latency reduced by X%, handled Y users)",
            f"Demonstrate practical implementation of {m_names[0] if m_names else 'unit testing'} in a featured project",
            "Structure achievements with Google X-Y-Z formula: Accomplished [X] measured by [Y] by doing [Z]"
        ],
        "roadmap": [
            f"Step 1: Highlight core strengths ({', '.join([s.replace('✓ ', '') for s in strong[:2]])})",
            f"Step 2: Learn role requirement: {m_names[0] if m_names else 'System Design'}",
            f"Step 3: Practice {m_names[1] if len(m_names)>1 else 'CI/CD Pipelines'}",
            f"Step 4: Deploy an end-to-end portfolio project tailored for {role}"
        ]
    }}

workflow = StateGraph(ResumeState)
workflow.add_node("parse_resume", parse_resume_node)
workflow.add_node("analyze_resume", analyze_resume_node)
workflow.add_edge(START, "parse_resume")
workflow.add_edge("parse_resume", "analyze_resume")
workflow.add_edge("analyze_resume", END)
agent_graph = workflow.compile()

# 5. UI & ROUTES
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Resume Agent — AI Resume Analyzer & Career Coach</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root { --bg: #090d16; --card: rgba(19, 27, 46, 0.8); --border: rgba(255, 255, 255, 0.08); --primary: #6366f1; --success: #10b981; --warning: #f59e0b; --text: #f8fafc; --muted: #94a3b8; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Plus Jakarta Sans', system-ui, sans-serif; background: radial-gradient(ellipse at top, #1e1b4b 0%, #090d16 60%, #05070c 100%); color: var(--text); min-height: 100vh; padding: 28px 16px; }
    .container { max-width: 860px; margin: 0 auto; display: flex; flex-direction: column; gap: 18px; }
    .card { background: var(--card); backdrop-filter: blur(16px); border: 1px solid var(--border); border-radius: 16px; padding: 22px; box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5); transition: border-color 0.2s; }
    .card:hover { border-color: rgba(99, 102, 241, 0.35); }
    .header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; }
    .brand { display: flex; align-items: center; gap: 12px; }
    .logo-icon { width: 44px; height: 44px; border-radius: 12px; background: linear-gradient(135deg, #6366f1, #8b5cf6); display: flex; align-items: center; justify-content: center; font-size: 22px; box-shadow: 0 6px 18px rgba(99, 102, 241, 0.4); }
    h1 { font-size: 1.45rem; font-weight: 800; letter-spacing: -0.02em; }
    .subtitle { color: var(--muted); font-size: 0.84rem; }
    .badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 9999px; font-size: 0.74rem; font-weight: 600; background: rgba(99, 102, 241, 0.12); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.25); }
    .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--success); box-shadow: 0 0 8px var(--success); }
    .form-group { margin-bottom: 16px; }
    label { display: block; font-size: 0.78rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 7px; }
    .file-drop { border: 1.5px dashed rgba(255,255,255,0.15); border-radius: 12px; padding: 20px; text-align: center; background: rgba(15, 23, 42, 0.4); cursor: pointer; transition: 0.2s; }
    .file-drop:hover { border-color: var(--primary); background: rgba(99, 102, 241, 0.06); }
    input[type="text"] { width: 100%; padding: 11px 14px; border-radius: 10px; border: 1px solid var(--border); background: rgba(15, 23, 42, 0.7); color: #fff; font-size: 0.95rem; outline: none; transition: 0.2s; }
    input[type="text"]:focus { border-color: var(--primary); box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2); }
    .role-pills { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
    .pill-btn { background: rgba(255,255,255,0.04); border: 1px solid var(--border); color: var(--muted); font-size: 0.75rem; padding: 4px 9px; border-radius: 6px; cursor: pointer; transition: 0.2s; }
    .pill-btn:hover { background: rgba(99, 102, 241, 0.2); color: #fff; border-color: var(--primary); }
    .btn { padding: 12px 24px; border-radius: 10px; font-weight: 600; font-size: 0.92rem; border: none; cursor: pointer; display: inline-flex; align-items: center; gap: 8px; transition: 0.2s; background: linear-gradient(135deg, #6366f1, #4f46e5); color: #fff; box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35); }
    .btn:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5); }
    .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; margin-bottom: 14px; }
    .metric-box { background: rgba(15, 23, 42, 0.5); border: 1px solid var(--border); border-radius: 12px; padding: 16px; display: flex; align-items: center; gap: 16px; }
    .score-circle { width: 62px; height: 62px; border-radius: 50%; background: conic-gradient(var(--success) calc(var(--score) * 1%), rgba(255,255,255,0.1) 0); display: flex; align-items: center; justify-content: center; position: relative; flex-shrink: 0; }
    .score-inner { width: 48px; height: 48px; border-radius: 50%; background: #0b1120; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.1rem; color: #fff; }
    .tag { display: inline-flex; align-items: center; gap: 4px; padding: 4px 10px; border-radius: 7px; margin: 3px; font-size: 0.8rem; font-weight: 500; }
    .tag-match { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.25); }
    .tag-miss { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.25); }
    .callout { background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(168, 85, 247, 0.05)); border-left: 3px solid var(--primary); padding: 12px 14px; border-radius: 0 10px 10px 0; margin: 12px 0; font-size: 0.9rem; }
    .checklist { list-style: none; display: flex; flex-direction: column; gap: 7px; margin-top: 6px; }
    .checklist li { display: flex; align-items: flex-start; gap: 8px; font-size: 0.86rem; color: #cbd5e1; }
    .checklist li::before { content: "→"; color: var(--primary); font-weight: bold; }
    .roadmap-steps { display: flex; flex-direction: column; gap: 7px; margin-top: 8px; }
    .step-item { display: flex; align-items: center; gap: 12px; padding: 9px 12px; background: rgba(15, 23, 42, 0.5); border: 1px solid var(--border); border-radius: 8px; font-size: 0.84rem; }
    .step-num { width: 22px; height: 22px; border-radius: 50%; background: var(--primary); color: #fff; display: flex; align-items: center; justify-content: center; font-size: 0.72rem; font-weight: 700; flex-shrink: 0; }
    .chat-box { height: 180px; overflow-y: auto; background: rgba(11, 17, 32, 0.6); border-radius: 12px; padding: 12px; border: 1px solid var(--border); display: flex; flex-direction: column; gap: 8px; }
    .msg { padding: 8px 12px; border-radius: 12px; max-width: 84%; font-size: 0.86rem; line-height: 1.45; }
    .msg-ai { align-self: flex-start; background: rgba(30, 41, 59, 0.7); border: 1px solid var(--border); color: #e2e8f0; }
    .msg-user { align-self: flex-end; background: linear-gradient(135deg, #6366f1, #4f46e5); color: #fff; }
    .quick-prompt { font-size: 0.74rem; padding: 3px 8px; border-radius: 6px; background: rgba(255,255,255,0.05); border: 1px solid var(--border); color: var(--muted); cursor: pointer; }
    .quick-prompt:hover { color: #fff; border-color: var(--primary); }
    .alert-error { background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.3); color: #fca5a5; padding: 10px 14px; border-radius: 8px; font-size: 0.86rem; margin-bottom: 12px; }
  </style>
</head>
<body>
  <div class="container">
    <div class="card header">
      <div class="brand"><div class="logo-icon">📄</div><div><h1>Resume Agent</h1><div class="subtitle">AI-Powered Resume Analyzer & Career Coach</div></div></div>
      <div class="badge"><span class="dot"></span> LangGraph • AI Agent</div>
    </div>
    {% if error %}<div class="alert-error">⚠️ {{ error }}</div>{% endif %}
    <div class="card">
      <form method="POST" enctype="multipart/form-data" id="analyzeForm">
        <div class="form-group">
          <label>Upload Resume (PDF / TXT)</label>
          <div class="file-drop" onclick="document.getElementById('fileInput').click()">
            <span id="fileLabel">📁 Select your actual PDF or TXT resume to begin</span>
            <input type="file" id="fileInput" name="resume_file" accept=".pdf,.txt" style="display:none" onchange="updateFileName(this)" required>
          </div>
        </div>
        <div class="form-group">
          <label>Target Job Role</label>
          <input type="text" id="roleInput" name="target_role" value="{{ target_role or 'Machine Learning Engineer' }}" required>
          <div class="role-pills">
            {% for r in ['Machine Learning Engineer', 'Full Stack Developer', 'Data Scientist', 'DevOps / Cloud', 'AI Engineer'] %}
            <span class="pill-btn" onclick="setRole('{{ r }}')">{{ r }}</span>
            {% endfor %}
          </div>
        </div>
        <button type="submit" class="btn" id="submitBtn">⚡ Analyze My Resume</button>
      </form>
    </div>
    {% if analysis %}
    <div class="card">
      <div class="dashboard-grid">
        <div class="metric-box">
          <div class="score-circle" style="--score: {{ analysis.score }};"><div class="score-inner">{{ analysis.score }}</div></div>
          <div>
            <div style="font-size:0.75rem; color:var(--muted); text-transform:uppercase; font-weight:700;">AI Resume Score</div>
            <div style="font-weight:700; color:#10b981;">{{ 'High ATS Readiness' if analysis.score >= 80 else ('Good Foundation' if analysis.score >= 65 else 'Needs Optimization') }}</div>
            <div style="font-size:0.75rem; color:var(--muted);">out of 100 (Calculated)</div>
          </div>
        </div>
        <div class="metric-box">
          <div class="score-circle" style="--score: {{ analysis.match_pct }}; background: conic-gradient(#38bdf8 calc(var(--score)*1%), rgba(255,255,255,0.1) 0);"><div class="score-inner" style="color:#38bdf8;">{{ analysis.match_pct }}%</div></div>
          <div>
            <div style="font-size:0.75rem; color:var(--muted); text-transform:uppercase; font-weight:700;">Role Compatibility</div>
            <div style="font-weight:700; color:#f8fafc;">{{ target_role }}</div>
            <div style="font-size:0.75rem; color:var(--muted);">skills benchmarked</div>
          </div>
        </div>
      </div>
      <div style="margin: 12px 0;"><label>Verified Matches in Resume</label><div>{% for s in analysis.strong_matches %}<span class="tag tag-match">{{ s }}</span>{% endfor %}</div></div>
      <div style="margin: 12px 0;"><label>Skill Gaps for {{ target_role }}</label><div>{% for m in analysis.missing_skills %}<span class="tag tag-miss">{{ m }}</span>{% endfor %}</div></div>
      <div class="callout"><b>🎯 Strategic Recommendation:</b> {{ analysis.recommendation }}</div>
      <label>Actionable Resume Improvements</label>
      <ul class="checklist">{% for imp in analysis.improvements %}<li>{{ imp }}</li>{% endfor %}</ul>
      <div style="margin-top: 16px;"><label>Personalized Career Roadmap</label>
        <div class="roadmap-steps">{% for step in analysis.roadmap %}<div class="step-item"><span class="step-num">{{ loop.index }}</span><span>{{ step }}</span></div>{% endfor %}</div>
      </div>
    </div>
    <div class="card" style="display:flex; flex-direction:column; gap:10px;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div><h3 style="font-size:1.1rem; font-weight:700;">💬 Interactive AI Career Coach</h3><div class="subtitle">Personalized feedback based on your uploaded resume</div></div>
        <span class="badge"><span class="dot"></span> Active</span>
      </div>
      <div style="display:flex; flex-wrap:wrap; gap:6px;">
        <span class="quick-prompt" onclick="quickAsk('How can I boost my score to 90+?')">How to boost score?</span>
        <span class="quick-prompt" onclick="quickAsk('Suggest a standout portfolio project for this role')">Suggest project</span>
        <span class="quick-prompt" onclick="quickAsk('How should I frame my resume bullets using Google X-Y-Z?')">Rewrite bullets</span>
      </div>
      <div class="chat-box" id="chatBox"><div class="msg msg-ai">👋 Hello! I evaluated your resume against <b>{{ target_role }}</b> requirements (Calculated score: <b>{{ analysis.score }}/100</b>). Ask me how to improve your bullet points or bridge your skill gaps!</div></div>
      <div style="display:flex; gap:8px;">
        <input type="text" id="chatInput" placeholder="Ask your AI coach anything..." onkeydown="if(event.key==='Enter') sendChat()">
        <button class="btn" onclick="sendChat()" id="sendBtn" style="padding:10px 18px;">Send</button>
      </div>
    </div>
    {% endif %}
  </div>
  <script>
    function updateFileName(input) { if (input.files && input.files[0]) document.getElementById('fileLabel').innerText = '📄 ' + input.files[0].name; }
    function setRole(role) { document.getElementById('roleInput').value = role; }
    function quickAsk(text) { document.getElementById('chatInput').value = text; sendChat(); }
    document.getElementById('analyzeForm').addEventListener('submit', () => { const b = document.getElementById('submitBtn'); b.innerHTML = '⏳ Analyzing...'; b.style.opacity = '0.7'; });
    {% if analysis %}
    async function sendChat() {
      const input = document.getElementById('chatInput'), box = document.getElementById('chatBox'), msg = input.value.trim();
      if (!msg) return;
      box.innerHTML += `<div class="msg msg-user">${msg}</div>`;
      input.value = ''; box.scrollTop = box.scrollHeight;
      const btn = document.getElementById('sendBtn'); btn.disabled = true; btn.innerText = '...';
      try {
        const res = await fetch('/chat', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ message: msg, target_role: '{{ target_role }}', score: '{{ analysis.score }}', missing: {{ analysis.missing_skills|tojson }} }) });
        const data = await res.json();
        box.innerHTML += `<div class="msg msg-ai">${data.reply}</div>`;
      } catch (err) {
        box.innerHTML += `<div class="msg msg-ai" style="color:#f87171;">Error connecting to assistant. Please try again.</div>`;
      }
      btn.disabled = false; btn.innerText = 'Send'; box.scrollTop = box.scrollHeight;
    }
    {% endif %}
  </script>
</body>
</html>"""

@app.route("/", methods=["GET", "POST"])
def index():
    analysis, target_role, error = None, "Machine Learning Engineer", None
    if request.method == "POST":
        target_role = request.form.get("target_role", "Machine Learning Engineer").strip()
        file = request.files.get("resume_file")
        text = extract_text(file) if file else ""
        if not text:
            error = "Please upload a valid PDF or TXT resume file with readable text."
        else:
            result = agent_graph.invoke({"resume_text": text, "target_role": target_role, "analysis": {}})
            analysis = result.get("analysis")
    return render_template_string(HTML, analysis=analysis, target_role=target_role, error=error)

@app.route("/chat", methods=["POST"])
def chat():
    d = request.get_json() or {}
    msg, role, score, missing = d.get("message", ""), d.get("target_role", "ML Engineer"), d.get("score", "70"), d.get("missing", [])
    if API_KEY:
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY, temperature=0.3)
            reply = llm.invoke([HumanMessage(content=f"You are an AI Resume Coach. Candidate score: {score}/100 for {role}. Skill gaps: {missing}. Answer briefly: {msg}")]).content
            return jsonify({"reply": reply})
        except Exception: pass
    m = ", ".join([x.replace("⚠ ", "") for x in missing[:2]]) if missing else "system design"
    if "boost" in msg.lower() or "score" in msg.lower():
        reply = f"To raise your score from {score}/100 for {role}, add verified projects showcasing {m} and quantify impact with Google X-Y-Z metrics."
    elif "project" in msg.lower():
        reply = f"For {role}, build an end-to-end portfolio project featuring {m} with containerization (Docker) and CI/CD."
    else:
        reply = f"For {role}, reframe your bullets to emphasize measurable outcomes (e.g. 'Improved efficiency by 25% by deploying {m}')."
    return jsonify({"reply": reply})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Resume Agent running at http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

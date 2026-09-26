# Resume Agent
> **AI-Powered Resume Analyzer & Career Assistant**

A lightweight, beginner-friendly AI Agent designed for job seekers to evaluate resume strength, extract skills, compare compatibility against target roles, identify skill gaps, and generate customized career roadmaps.

Built with **Google Gemini**, **LangChain**, **LangGraph**, and an **in-memory RAG knowledge base**, implemented entirely in a single file: [`app.py`](app.py).

---

## Architecture & How It Works

```
                        User Resume (PDF / DOCX / TXT)
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   Flask Server (app.py)   │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │    LangGraph Workflow     │
                        │                           │
                        │          [START]          │
                        │             │             │
                        │             ▼             │
                        │     [parse_resume]        │
                        │             │             │
                        │             ▼             │
                        │    [analyze_resume]       │
                        │             │             │
                        │             ▼             │
                        │   [match_target_role] <───┼── RAG Knowledge Base
                        │             │             │   (ATS Standards & Roles)
                        │             ▼             │
                        │ [generate_recommendations]│
                        │             │             │
                        │             ▼             │
                        │           [END]           │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
        ┌───────────────────────────────────────────────────────────┐
        │                 Interactive Dashboard & Chat              │
        │ - AI Estimated Resume Score (0-100)                       │
        │ - Categorized Competencies (Languages, DBs, Cloud, ML)    │
        │ - Strong Matches vs Missing Gaps (with warnings)          │
        │ - Actionable Rewrites (Google X-Y-Z Formula)              │
        │ - Visual Learning Roadmap & Suggested Projects            │
        │ - Contextual AI Chat Assistant                            │
        └───────────────────────────────────────────────────────────┘
```

---

## Features

- 📑 **Multi-Format Resume Parsing**: Supports PDF (`pypdf`), DOCX (`python-docx`), and plain text.
- 🤖 **Structured AI Profile Extraction**: Extracts contact details, education, work experience, projects, and categorizes technical competencies.
- 📊 **AI Estimated Resume Score (0–100)**: Evaluates content quality, technical breadth, projects, experience, ATS readiness, and impact (clearly framed as an AI estimate, not an official ATS audit).
- 🎯 **Target Job Role Matching**: Compares candidate skills against role benchmarks (e.g., Machine Learning Engineer, Data Scientist, Full Stack Developer, DevOps, AI Engineer) to identify strong matches and critical gaps.
- 💡 **Actionable Improvements**: Suggests specific bullet point optimizations using the Google X-Y-Z formula without fabricating accomplishments.
- 🗺️ **Career Roadmap & Project Ideas**: Recommends prioritized skills to learn, portfolio projects to bridge gaps, and an interactive step-by-step career path.
- 💬 **Interactive AI Resume Chat**: Chat with an AI coach that uses the candidate's resume and target role as grounding context to answer questions, improve summaries, and conduct mock interviews.
- ⚡ **Lightweight In-Memory RAG**: Curated built-in knowledge base on ATS formatting, role expectations, and impact metrics.
- 🛠️ **LangChain Tools Integration**: Core functions structured as LangChain `@tool` definitions (`resume_analyzer_tool`, `skill_matcher_tool`, `career_knowledge_retriever_tool`, `score_calculator_tool`).

---

## Technology Stack

- **Python 3.10+**
- **Flask**: Web application server and API endpoints
- **Google Gemini**: Large language model for reasoning, structured extraction, and conversational guidance
- **LangChain**: Prompt templates, structured outputs, and agent tools
- **LangGraph**: `StateGraph` multi-node agent workflow orchestration
- **PyPDF / python-docx**: Document text extraction
- **HTML5 / CSS3 / Vanilla JavaScript**: Modern, responsive dark-mode web dashboard embedded directly in `app.py`

---

## Project Structure

```
Resume-Agent/
├── app.py              # Entire application code (LangChain, LangGraph, RAG, Tools, Flask, UI)
├── requirements.txt    # Minimal dependencies
├── .env.example        # Environment variable template
└── README.md           # Documentation
```

---

## Quick Start Guide

### 1. Clone or Open the Repository
```bash
git clone https://github.com/your-username/Resume-Agent.git
cd Resume-Agent
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Your Gemini API Key
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```
> **Tip:** You can obtain a free API key from [Google AI Studio](https://aistudio.google.com/).  
> If you don't set it in `.env`, you can also enter your key directly via the **Configure API Key** button in the web interface header!

### 4. Run the Application
```bash
python app.py
```

### 5. Open in Your Browser
Navigate to:
```
http://127.0.0.1:5000
```

---

## 1-Click Demo Testing

Don't have a resume PDF on hand?
1. Open `http://127.0.0.1:5000`
2. Click **⚡ Load Sample Resume (1-Click Demo)**
3. The agent will load a sample technical profile and execute the LangGraph pipeline immediately.

---

## Ethical & Beginner-Friendly Design Rules

- **Zero Hallucination / No Fabrication**: The agent strictly avoids creating fictitious companies, percentages, or achievements.
- **Clear Disclaimers**: Resume scoring is labeled as an *AI Estimated Resume Score*, not a guarantee from an official ATS.
- **Single-File Architecture**: Everything is visible and understandable inside [`app.py`](app.py) for students and beginners learning GenAI, LangChain, and LangGraph.
"# AI-Resume-Agent" 

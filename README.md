---
title: SmartHireAI
emoji: 🤖
colorFrom: indigo
colorTo: green
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
---

# SmartHireAI Candidate Ranker

SmartHireAI is an end-to-end intelligent candidate discovery and ranking system designed for the Redrob Hackathon. It matches a pool of candidates against a specified job description for the **Senior AI Engineer — Founding Team** role.

---

## 🚀 Key Features

* **Two-Stage Candidate Retrieval**:
  * **Stage 1 (Heuristic Filter)**: Scores all 100,000 candidates using a fast, CPU-efficient heuristic function (combining title alignment, retrieval keywords, required skills, experience years, and behavioral engagement) in under 2 seconds.
  * **Stage 2 (Semantic Re-ranking)**: Extracts the top 2,500 candidates and ranks them using a local `SentenceTransformer` bi-encoder (`sentence-transformers/all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`), completing under 1.5 minutes.
* **Honeypot/Trap Guardrails**:
  * Explicitly filters out synthetic candidates with impossible profiles or key honeypot signals (e.g. skill durations exceeding experience years, summary keyword stuffing, non-tech titles with AI skills) using a strict threshold check (`honeypot_penalty >= 0.4`), achieving **0% honeypot rate** in the top 100.
* **Full CPU Constraints Compliance**:
  * Optimized to execute completely offline (no network API calls) on standard CPU within **1.5 minutes** (well below the 5-minute limit).

---

## 📁 Repository Structure

```text
SmartHireAI/
│
├── README.md
├── requirements.txt
├── rank.py
│
├── src/
│   ├── candidate_filter.py
│   ├── feature_engineering.py
│   ├── honeypot_detector.py
│   ├── jd_parser.py
│   ├── scoring.py
│   ├── ranker.py
│   └── submission_generator.py
│
├── output/
│   └── final_submission.csv
│
├── audit_honeypots.py
├── submission_metadata.yaml
└── docs/
    └── PPT.pdf
```

---

## 🛠️ Setup & Installation

1. Create and activate your virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate   # On Windows
   source venv/bin/activate  # On Linux/macOS
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Usage & Reproduction

Run the ranking pipeline:
```bash
python rank.py --model fast
```
* Use `--model fast` for the fast MiniLM embedding model (default, takes ~1.5 minutes).
* Use `--model quality` for the BGE-small embedding model.
* The output will be saved as `output/final_submission.csv`.

---

## 🔍 Verification

Run the honeypot audit tool to verify the quality and safety of the final top 100 candidates list:
```bash
python audit_honeypots.py
```
Expected output:
```text
Loading candidates...
Honeypots in Top100: 0
```

---

## 🌐 Streamlit Web Interface

A professional Streamlit web interface is available for interactive candidate ranking and discovery. It allows you to dynamically upload custom job descriptions and candidate datasets, run the AI ranking pipeline, inspect the top-ranked candidates, and download the resulting CSV.

### 🛠️ Running Locally

1. Ensure all dependencies (including Streamlit) are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. Open your browser and navigate to `http://localhost:8501`.

4. Upload your job description (in `.md` or `.txt` format) and the candidate profiles (`.jsonl`), select the embedding model, and click **Rank Candidates**.

---

### ☁️ Deploying to Hugging Face Spaces

You can deploy this application to Hugging Face Spaces in just a few steps:

1. **Create a New Space**:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
   - Set a name, choose **Streamlit** as the SDK, and select the owner/visibility.

2. **Commit/Upload Files**:
   - Upload the following files and directories to the root of your Space repository:
     * `app.py` (Streamlit entry point)
     * `requirements.txt` (dependencies list)
     * `src/` (the entire source code directory containing filters, features, and ranking code)
     * `rank.py` (command-line script)
     * `audit_honeypots.py` (honeypot auditor)
     * `submission_metadata.yaml` (metadata description)

3. **Automatic Deployment**:
   - Once the files are uploaded, Hugging Face will automatically install the requirements from `requirements.txt` and launch the Streamlit server.
   - You can access your live deployed application at: `https://huggingface.co/spaces/<your-username>/<your-space-name>`.


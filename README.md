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

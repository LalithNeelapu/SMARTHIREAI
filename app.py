#!/usr/bin/env python3
import sys
from pathlib import Path
import tempfile
import shutil
import traceback

import streamlit as st
import pandas as pd

# Add src to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ranker import run_ranking
import jd_parser


def local_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;700&display=swap');
        
        html, body, [class*="css"], .stApp {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        
        /* Glassmorphism Title Card */
        .title-card {
            background: linear-gradient(135deg, rgba(31, 41, 55, 0.9) 0%, rgba(17, 24, 39, 0.95) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 2.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
            text-align: center;
        }
        
        .title-card h1 {
            font-family: 'Outfit', sans-serif;
            font-weight: 800;
            background: linear-gradient(90deg, #6366F1 0%, #3B82F6 50%, #10B981 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            font-size: 2.8rem;
            letter-spacing: -0.02em;
        }
        
        .title-card p {
            color: #9CA3AF;
            font-size: 1.1rem;
            font-weight: 300;
        }

        /* Card panels for Uploaders and Configuration */
        .config-card {
            background: rgba(31, 41, 55, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }

        /* Subheadings */
        h2, h3 {
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            letter-spacing: -0.01em;
        }
        
        /* Premium Button styling overrides */
        div.stButton > button:first-child {
            background: linear-gradient(90deg, #4F46E5 0%, #2563EB 100%);
            color: #ffffff;
            font-weight: 600;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 8px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
            width: 100%;
        }
        
        div.stButton > button:first-child:hover {
            background: linear-gradient(90deg, #4338CA 0%, #1D4ED8 100%);
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
            transform: translateY(-2px);
        }

        div.stButton > button:first-child:active {
            transform: translateY(1px);
        }

        /* Success alerts */
        .stAlert {
            border-radius: 8px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def main():
    st.set_page_config(
        page_title="SmartHireAI – Intelligent Candidate Ranker",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    
    local_css()
    
    # Title / Header Card
    st.markdown(
        """
        <div class="title-card">
            <h1>SmartHireAI</h1>
            <p>Intelligent Candidate Ranker & Discovery Engine</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Grid Layout: Left Column for uploaders/config, Right Column for instructions & process trigger
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown('<div class="config-card">', unsafe_allow_html=True)
        st.subheader("📁 Upload Data Sources")
        
        jd_file = st.file_uploader(
            "Job Description (.md or .txt)",
            type=["md", "txt"],
            help="Upload the job description file. Markdown formatting is fully parsed."
        )
        
        candidates_file = st.file_uploader(
            "Candidate Profiles (.jsonl)",
            type=["jsonl"],
            help="Upload a candidate profiles dataset in line-delimited JSON format."
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_right:
        st.markdown('<div class="config-card">', unsafe_allow_html=True)
        st.subheader("⚙️ Pipeline Configuration")
        
        model_option = st.selectbox(
            "Embedding Model Selection",
            options=["fast", "quality"],
            index=0,
            format_func=lambda x: "sentence-transformers (all-MiniLM-L6-v2) - Fast" if x == "fast" else "BAAI (bge-small-en-v1.5) - Quality",
            help="Choose 'fast' for high-speed local processing (~1.5 min) or 'quality' for advanced semantic similarity (~4 min)."
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Rank Candidates Button
        rank_button = st.button(
            "Rank Candidates 🚀",
            disabled=not (jd_file and candidates_file)
        )
        
        if not (jd_file and candidates_file):
            st.info("💡 Please upload both files to activate the ranking pipeline.")
            
        st.markdown('</div>', unsafe_allow_html=True)

    # Execution phase
    if rank_button and jd_file and candidates_file:
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            with st.spinner("⚡ Running the candidate ranking pipeline... This may take up to a few minutes."):
                # Read uploaded Job Description
                try:
                    jd_content = jd_file.read().decode("utf-8")
                except UnicodeDecodeError:
                    jd_file.seek(0)
                    jd_content = jd_file.read().decode("latin-1", errors="ignore")
                
                # Write uploaded candidate JSONL file to temp directory
                temp_candidates_path = temp_dir / "candidates.jsonl"
                with open(temp_candidates_path, "wb") as f:
                    f.write(candidates_file.read())
                
                temp_output_path = temp_dir / "final_submission.csv"
                
                # Monkey-patch the jd_parser loading logic to use the uploaded job description text
                original_load_jd = jd_parser.load_job_description
                jd_parser.load_job_description = lambda: jd_content
                
                try:
                    # Map select box option to required model name
                    model_name = (
                        "BAAI/bge-small-en-v1.5"
                        if model_option == "quality"
                        else "sentence-transformers/all-MiniLM-L6-v2"
                    )
                    
                    # Execute the existing pipeline
                    submission = run_ranking(
                        candidates_path=str(temp_candidates_path),
                        output_path=str(temp_output_path),
                        show_progress=False,
                        model_name=model_name
                    )
                finally:
                    # Restore original function regardless of execution outcome
                    jd_parser.load_job_description = original_load_jd

                st.success("🎉 Ranking completed successfully!")
            
            # Results UI
            st.divider()
            
            # Display layout for results
            res_col1, res_col2 = st.columns([3, 1])
            
            with res_col1:
                st.subheader("🏆 Top 20 Ranked Candidates")
            
            with res_col2:
                # Download full submission CSV button
                csv_bytes = submission.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Full Rankings (CSV)",
                    data=csv_bytes,
                    file_name="final_submission.csv",
                    mime="text/csv",
                    key="download-csv"
                )
                
            # Table formatting
            top_20 = submission.head(20).copy()
            top_20["score"] = top_20["score"].round(4)
            top_20 = top_20[["rank", "candidate_id", "score", "reasoning"]]
            
            st.dataframe(
                top_20,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "rank": st.column_config.NumberColumn("Rank", format="%d"),
                    "candidate_id": st.column_config.TextColumn("Candidate ID"),
                    "score": st.column_config.NumberColumn("Final Score"),
                    "reasoning": st.column_config.TextColumn("Matching Reasoning"),
                }
            )
            
        except Exception as e:
            st.error(f"❌ An error occurred during the ranking process: {str(e)}")
            with st.expander("Show Detailed Stacktrace"):
                st.code(traceback.format_exc())
                
        finally:
            # Cleanup temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()

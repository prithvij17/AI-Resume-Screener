"""
app.py
------
Streamlit front-end for the AI-Powered Resume Screener & Job Matcher.

Ties together all four layers built earlier:
    Layer 1 (ML)     -> pipeline/ml_ranking.py       (TF-IDF + cosine similarity)
    Layer 2 (NLP)    -> pipeline/nlp_enrichment.py   (regex + spaCy extraction)
    Layer 3 (Agents) -> agents/crew_agents.py + agents/run_agents.py (CrewAI + Groq)
    Layer 4 (Email)  -> email/send_rejections.py     (SMTP rejection dispatch)

Run with:  streamlit run app.py
"""

import os
import sys
import ast
import smtplib
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup — makes agents/ and email/ importable as plain modules, and
# pipeline/ importable as a package. Order matters: see the collision check
# in the project notes — this combination was verified safe (the real
# stdlib `email` package always wins over our email/ folder because it's a
# regular package with __init__.py, while ours has none).
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "agents"))
sys.path.insert(0, os.path.join(BASE_DIR, "email"))

from pipeline.ml_ranking import rank_all_jobs, score_single_resume
from pipeline.nlp_enrichment import (
    load_nlp,
    build_skill_vocab,
    build_skill_matcher,
    enrich_resume_row,
    enrich_dataframe,
)
import crew_agents
import run_agents
import send_rejections

load_dotenv(os.path.join(BASE_DIR, ".env"))

DATA_DIR = os.path.join(BASE_DIR, "data")
RESUMES_CSV = os.path.join(DATA_DIR, "resumes.csv")
JOBS_CSV = os.path.join(DATA_DIR, "job_descriptions.csv")
RANKED_CSV = os.path.join(DATA_DIR, "ranked_candidates.csv")
ENRICHED_CSV = os.path.join(DATA_DIR, "enriched_candidates.csv")
SCORED_CSV = os.path.join(DATA_DIR, "scored_candidates.csv")

LIST_COLUMNS = ["skills_extracted", "certifications_extracted", "languages_extracted"]


def parse_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    """CSV round-trips turn Python lists into repr strings — convert back."""
    df = df.copy()
    for col in LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: v if isinstance(v, list) else (ast.literal_eval(v) if isinstance(v, str) and v.startswith("[") else [])
            )
    return df


st.set_page_config(page_title="AI Resume Screener", layout="wide", page_icon="🧑‍💼")
st.title("🧑‍💼 AI-Powered Resume Screener & Job Matcher")
st.caption("ML ranking → NLP enrichment → CrewAI multi-agent scoring (Groq) → automated rejection emails")

# ---------------------------------------------------------------------------
# Sidebar: pipeline status + configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Pipeline status")
    st.write("✅ Bundled data found" if os.path.exists(RESUMES_CSV) else "❌ No bundled data in data/")
    st.write("✅ Ranked" if os.path.exists(RANKED_CSV) else "⬜ Not ranked yet")
    st.write("✅ Enriched" if os.path.exists(ENRICHED_CSV) else "⬜ Not enriched yet")
    st.write("✅ Agent-scored" if os.path.exists(SCORED_CSV) else "⬜ Not scored yet")

    st.divider()
    st.header("Configuration")

    if os.getenv("GROQ_API_KEY"):
        st.success("Groq API key loaded (.env)")
    else:
        entered_key = st.text_input(
            "GROQ_API_KEY", type="password",
            help="Session-only — not written to disk. Get a free key at console.groq.com/keys",
        )
        if entered_key:
            os.environ["GROQ_API_KEY"] = entered_key
            st.success("Key set for this session")

    if os.getenv("SMTP_USERNAME") and os.getenv("SMTP_PASSWORD"):
        st.success(f"SMTP configured ({os.getenv('SMTP_USERNAME')})")
    else:
        smtp_user_in = st.text_input("SMTP_USERNAME (Gmail address)")
        smtp_pass_in = st.text_input("SMTP_PASSWORD (Gmail App Password)", type="password")
        if smtp_user_in and smtp_pass_in:
            os.environ["SMTP_USERNAME"] = smtp_user_in
            os.environ["SMTP_PASSWORD"] = smtp_pass_in
            os.environ.setdefault("FROM_EMAIL", smtp_user_in)
            st.success("SMTP set for this session")

    if os.getenv("TEST_OVERRIDE_EMAIL"):
        st.warning(f"TEST_OVERRIDE_EMAIL active — all real sends go to {os.getenv('TEST_OVERRIDE_EMAIL')}")

    st.divider()
    if st.button("🗑️ Clear generated data (start fresh)"):
        for f in [RANKED_CSV, ENRICHED_CSV, SCORED_CSV]:
            if os.path.exists(f):
                os.remove(f)
        for k in ["ranked_df", "enriched_df", "live_candidate"]:
            st.session_state.pop(k, None)
        st.success("Cleared. Reload the page.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Rank & Enrich", "🎯 Live Candidate Test", "🤖 Agent Scoring", "✉️ Rejection Emails"]
)

# ---------------------------------------------------------------------------
# Tab 1 — Layer 1 (ML) + Layer 2 (NLP)
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Layers 1 + 2 — TF-IDF Ranking + spaCy Enrichment")
    st.caption("Uses the bundled sample dataset by default, or upload your own resumes/job-descriptions CSVs.")

    col1, col2 = st.columns(2)
    with col1:
        resumes_upload = st.file_uploader("Resumes CSV (optional)", type="csv", key="resumes_upload")
    with col2:
        jobs_upload = st.file_uploader("Job descriptions CSV (optional)", type="csv", key="jobs_upload")

    if st.button("▶ Run ML + NLP Pipeline", type="primary"):
        resumes = pd.read_csv(resumes_upload) if resumes_upload else pd.read_csv(RESUMES_CSV)
        jobs = pd.read_csv(jobs_upload) if jobs_upload else pd.read_csv(JOBS_CSV)

        with st.spinner("Ranking candidates (TF-IDF + cosine similarity)..."):
            ranked = rank_all_jobs(jobs, resumes)
            ranked.to_csv(RANKED_CSV, index=False)

        with st.spinner("Extracting structured fields (regex + spaCy PhraseMatcher)..."):
            enriched_resumes = enrich_dataframe(resumes, jobs)
            enrich_cols = [
                "candidate_name", "email", "phone", "location",
                "degree_extracted", "university_extracted", "grad_year_extracted",
                "job_title_extracted", "company_extracted", "years_exp_extracted", "has_prior_experience",
                "certifications_extracted", "languages_extracted", "skills_extracted",
            ]
            enriched = ranked.merge(enriched_resumes[enrich_cols], on="candidate_name", how="left")
            enriched.to_csv(ENRICHED_CSV, index=False)

        st.session_state["ranked_df"] = ranked
        st.session_state["enriched_df"] = enriched
        st.success(f"Done — ranked and enriched {len(enriched)} candidates across {jobs['role'].nunique()} jobs.")

    enriched_df = st.session_state.get("enriched_df")
    if enriched_df is None and os.path.exists(ENRICHED_CSV):
        enriched_df = pd.read_csv(ENRICHED_CSV)

    if enriched_df is not None and not enriched_df.empty:
        st.divider()
        job_roles = sorted(enriched_df["job_role"].dropna().unique())
        selected_role = st.selectbox("View top candidates for:", job_roles)
        top_n = st.slider("How many to show", 3, 20, 5)
        subset = enriched_df[enriched_df["job_role"] == selected_role].sort_values("rank").head(top_n)
        display_cols = [c for c in [
            "rank", "candidate_name", "match_percent", "years_exp_extracted",
            "degree_extracted", "fit_level",
        ] if c in subset.columns]
        st.dataframe(subset[display_cols], width='stretch', hide_index=True)

        if "fit_level" in enriched_df.columns:
            st.divider()
            st.markdown("**Validation — does the TF-IDF score line up with the labeled `fit_level`?**")
            fig = px.box(
                enriched_df, x="fit_level", y="match_score",
                category_orders={"fit_level": ["high", "medium", "low"]},
                title="TF-IDF match score by ground-truth fit level",
            )
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("Run the pipeline above to see ranked results.")

# ---------------------------------------------------------------------------
# Tab 2 — Live single-candidate test (paste text or upload a PDF)
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Test a single resume live")
    st.caption(
        "Section extraction (education/experience/certifications) works best on resumes following the "
        "Name/Email/Phone + EDUCATION/SKILLS/EXPERIENCE section template. Skill detection works on any text."
    )

    if not os.path.exists(JOBS_CSV) or not os.path.exists(RESUMES_CSV):
        st.warning("Bundled data/resumes.csv and data/job_descriptions.csv are required for this tab.")
    else:
        jobs_df = pd.read_csv(JOBS_CSV)
        resumes_df = pd.read_csv(RESUMES_CSV)

        selected_job_role = st.selectbox("Job to test against:", sorted(jobs_df["role"].unique()), key="live_job_select")
        job_row = jobs_df[jobs_df["role"] == selected_job_role].iloc[0].to_dict()

        input_mode = st.radio("Resume input:", ["Paste text", "Upload PDF"], horizontal=True)
        resume_text = None
        if input_mode == "Paste text":
            resume_text = st.text_area("Paste resume text here", height=220)
        else:
            pdf_file = st.file_uploader("Upload a resume PDF", type="pdf")
            if pdf_file:
                import pdfplumber

                with pdfplumber.open(pdf_file) as pdf:
                    resume_text = "\n".join((page.extract_text() or "") for page in pdf.pages)
                with st.expander("Extracted text (preview)"):
                    st.text(resume_text[:3000] if resume_text else "(no extractable text found in this PDF)")

        if resume_text and st.button("Score this candidate", type="primary"):
            with st.spinner("Computing TF-IDF match + extracting fields..."):
                match_pct = score_single_resume(resume_text, job_row, resumes_df)
                nlp, has_full_model = load_nlp()
                vocab = build_skill_vocab(resumes_df, jobs_df)
                matcher, lookup = build_skill_matcher(nlp, vocab)
                extracted = enrich_resume_row(resume_text, nlp, matcher, lookup)

            st.session_state["live_candidate"] = {
                "resume_text": resume_text, "job_row": job_row,
                "match_pct": match_pct, "extracted": extracted,
            }

        live = st.session_state.get("live_candidate")
        if live:
            st.divider()
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("TF-IDF Match Score", f"{live['match_pct']}%")
            with col2:
                shown = {k: v for k, v in live["extracted"].items() if k != "skills_extracted"}
                st.json(shown)
            st.write("**Skills detected:**", ", ".join(live["extracted"]["skills_extracted"]) or "none found")

            st.divider()
            st.markdown("**Optional: score this candidate with the CrewAI agent pipeline (uses real Groq credits)**")
            if not os.getenv("GROQ_API_KEY"):
                st.warning("Add your GROQ_API_KEY in the sidebar first.")
            elif st.button("🤖 Run AI Agent Scoring on this candidate"):
                candidate_payload = {
                    "candidate_name": "Live Test Candidate",
                    "degree_extracted": live["extracted"]["degree_extracted"],
                    "years_exp_extracted": live["extracted"]["years_exp_extracted"],
                    "skills_extracted": live["extracted"]["skills_extracted"],
                    "certifications_extracted": live["extracted"]["certifications_extracted"],
                    "match_percent": live["match_pct"],
                }
                job_payload = {
                    "role": live["job_row"].get("role"),
                    "category": live["job_row"].get("category"),
                    "required_skills": live["job_row"].get("required_skills", ""),
                    "preferred_skills": live["job_row"].get("preferred_skills", ""),
                    "experience_required": live["job_row"].get("experience_required", ""),
                    "education_required": live["job_row"].get("education_required", ""),
                    "job_description": live["job_row"].get("job_description", ""),
                }
                with st.spinner("Running Screening → Scoring → Feedback → Bias-Check agents..."):
                    try:
                        llm = crew_agents.get_llm()
                        agents = crew_agents.build_agents(llm)
                        result = run_agents.run_pipeline_for_candidate(agents, candidate_payload, job_payload)
                    except Exception as e:  # noqa: BLE001
                        result = {"agent_error": str(e)}

                if result.get("agent_error"):
                    st.error(f"Agent pipeline failed: {result['agent_error']}")
                else:
                    st.metric("AI Fit Score", result["fit_score"])
                    st.write("**Screening:**", result["screening_summary"])
                    st.write("**Scoring justification:**", result["scoring_justification"])
                    st.write("**Internal feedback:**", result["internal_feedback"])
                    st.write("**Candidate-facing feedback:**", result["final_candidate_feedback"])

# ---------------------------------------------------------------------------
# Tab 3 — Layer 3 (CrewAI agents, batch)
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Layer 3 — CrewAI Multi-Agent Scoring (Groq)")

    if not os.path.exists(ENRICHED_CSV):
        st.warning("Run the ML + NLP pipeline in the first tab first.")
    else:
        enriched_df = parse_list_columns(pd.read_csv(ENRICHED_CSV))

        max_per_job = st.slider("Candidates to score per job (cost control)", 1, 20, 3)
        process_all = st.checkbox("Process ALL candidates instead (slower, costs more)")

        if not os.getenv("GROQ_API_KEY"):
            st.warning("Add your GROQ_API_KEY in the sidebar to run this.")
        elif st.button("▶ Run Agent Scoring", type="primary"):
            done = run_agents.already_processed(SCORED_CSV)
            jobs_df = pd.read_csv(JOBS_CSV)
            jobs_by_category = {row["category"]: row.to_dict() for _, row in jobs_df.iterrows()}

            to_process = []
            for category, group in enriched_df.groupby("category"):
                job = jobs_by_category.get(category)
                if job is None:
                    continue
                group_sorted = group.sort_values("match_score", ascending=False)
                if not process_all:
                    group_sorted = group_sorted.head(max_per_job)
                for _, row in group_sorted.iterrows():
                    if (row["candidate_name"], job["role"]) not in done:
                        to_process.append((row, job))

            if not to_process:
                st.info("Nothing new to process — already scored, or no candidates match a posted job.")
            else:
                llm = crew_agents.get_llm()
                agents = crew_agents.build_agents(llm)

                progress = st.progress(0.0, text="Starting...")
                results = []
                for i, (row, job) in enumerate(to_process, start=1):
                    candidate_payload = run_agents.build_candidate_payload(row)
                    job_payload = run_agents.build_job_payload(job)
                    progress.progress(
                        i / len(to_process),
                        text=f"[{i}/{len(to_process)}] {candidate_payload['candidate_name']} → {job_payload['role']}",
                    )
                    result = run_agents.run_pipeline_for_candidate(agents, candidate_payload, job_payload)
                    record = {
                        "candidate_name": row["candidate_name"],
                        "email": row.get("email"),
                        "job_role": job["role"],
                        "category": row["category"],
                        "match_percent": row.get("match_percent"),
                        "fit_level": row.get("fit_level"),
                        **result,
                    }
                    record["status"] = (
                        "rejected"
                        if (record["fit_score"] is not None and record["fit_score"] < run_agents.REJECTION_THRESHOLD)
                        else "shortlisted" if record["fit_score"] is not None else "error"
                    )
                    results.append(record)

                new_df = pd.DataFrame(results)
                if os.path.exists(SCORED_CSV):
                    existing = pd.read_csv(SCORED_CSV)
                    new_keys = set(zip(new_df["candidate_name"], new_df["job_role"]))
                    existing = existing[
                        ~existing.apply(lambda r: (r["candidate_name"], r["job_role"]) in new_keys, axis=1)
                    ]
                    combined = pd.concat([existing, new_df], ignore_index=True)
                else:
                    combined = new_df
                combined.to_csv(SCORED_CSV, index=False)
                progress.progress(1.0, text="Done.")
                st.success(f"Scored {len(results)} candidate/job pair(s).")

        if os.path.exists(SCORED_CSV):
            st.divider()
            scored_df = pd.read_csv(SCORED_CSV)
            display_cols = [c for c in [
                "candidate_name", "job_role", "fit_score", "status", "scoring_justification",
            ] if c in scored_df.columns]
            st.dataframe(scored_df[display_cols], width='stretch', hide_index=True)

# ---------------------------------------------------------------------------
# Tab 4 — Layer 4 (rejection email dispatch)
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Layer 4 — Rejection Email Dispatch")
    st.caption(
        "The dataset's emails are all @example.com/.net/.org (IANA-reserved, never deliver to real inboxes) — "
        "safe to test with --send. Set TEST_OVERRIDE_EMAIL in .env to redirect real sends to your own inbox."
    )

    if not os.path.exists(SCORED_CSV):
        st.warning("Run agent scoring in the previous tab first.")
    else:
        scored_df = pd.read_csv(SCORED_CSV)
        for col in ["email_sent", "email_sent_at", "email_error"]:
            if col not in scored_df.columns:
                scored_df[col] = None
            else:
                scored_df[col] = scored_df[col].astype("object")

        pending = send_rejections.get_pending_rejections(scored_df)
        st.write(f"**{len(pending)} pending rejection email(s)**")

        test_override = os.getenv("TEST_OVERRIDE_EMAIL")

        if not pending.empty:
            with st.expander(f"Preview messages ({min(len(pending), 10)} of {len(pending)} shown)"):
                preview_config = {
                    "test_override_email": test_override,
                    "from_name": os.getenv("FROM_NAME", "Recruiting Team"),
                    "from_email": os.getenv("FROM_EMAIL", "your_email@example.com"),
                }
                for _, row in pending.head(10).iterrows():
                    _, to_address, subject, body = send_rejections.build_message(row, preview_config)
                    st.markdown(f"**To:** {to_address}  \n**Subject:** {subject}")
                    st.text(body)
                    st.divider()

            confirm = st.checkbox("I understand this will send real emails to these addresses.")
            if st.button("✉️ Send Rejection Emails", disabled=not confirm, type="primary"):
                try:
                    smtp_config = send_rejections.load_smtp_config()
                except RuntimeError as e:
                    st.error(str(e))
                else:
                    server = smtplib.SMTP(smtp_config["host"], smtp_config["port"])
                    server.starttls()
                    server.login(smtp_config["username"], smtp_config["password"])

                    sent, failed = 0, 0
                    progress = st.progress(0.0)
                    try:
                        for i, (idx, row) in enumerate(pending.iterrows(), start=1):
                            msg, to_address, _, _ = send_rejections.build_message(row, smtp_config)
                            try:
                                server.send_message(msg)
                                scored_df.loc[idx, "email_sent"] = True
                                scored_df.loc[idx, "email_sent_at"] = datetime.now(timezone.utc).isoformat()
                                scored_df.loc[idx, "email_error"] = None
                                sent += 1
                            except Exception as e:  # noqa: BLE001
                                scored_df.loc[idx, "email_sent"] = False
                                scored_df.loc[idx, "email_error"] = str(e)
                                failed += 1
                            progress.progress(i / len(pending))
                    finally:
                        server.quit()

                    scored_df.to_csv(SCORED_CSV, index=False)
                    st.success(f"Sent: {sent}, Failed: {failed}")
        else:
            st.info("No pending rejections to send.")

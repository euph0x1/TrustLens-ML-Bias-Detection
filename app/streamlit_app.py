import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trustlens.bias.detector import BiasDetector
from trustlens.explain.highlights import highlight_claims, highlight_flagged_spans
from trustlens.fairness.counterfactual import CounterfactualEvaluator, load_scenarios
from trustlens.io.report import audit_to_markdown, export_audit_json
from trustlens.models import ModelRegistry
from trustlens.pipeline import TrustLensPipeline

st.set_page_config(
    page_title="TrustLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLES_PATH = ROOT / "data" / "sample" / "examples.json"


@st.cache_resource
def get_pipeline() -> TrustLensPipeline:
    return TrustLensPipeline()


@st.cache_resource
def load_models() -> dict:
    return ModelRegistry.preload()


@st.cache_data
def load_examples() -> dict:
    with EXAMPLES_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def trust_color(score: float) -> str:
    if score >= 75:
        return "#2e7d32"
    if score >= 50:
        return "#f9a825"
    return "#c62828"


def render_trust_gauge(score: float) -> None:
    color = trust_color(score)
    st.markdown(
        f"""
        <div style="text-align:center; padding: 1rem;">
            <div style="font-size:3rem; font-weight:bold; color:{color};">{score:.1f}</div>
            <div style="color:#666;">Trust Score / 100</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_info() -> None:
    st.sidebar.title("TrustLens")
    st.sidebar.markdown(
        "Audit AI-generated text for **bias**, **factuality**, and **fairness** risks."
    )
    with st.sidebar.expander("Model status"):
        status = load_models()
        for name, ok in status.items():
            icon = "✅" if ok else "⚠️"
            st.write(f"{icon} {name}")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Limitations:** Trust scores support human review; they do not certify safety. "
        "Factuality checks work best when source context is provided."
    )


def tab_audit() -> None:
    st.header("Single Response Audit")
    examples = load_examples()

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Load: Biased hiring example"):
            ex = examples["biased_hiring"]
            st.session_state["prompt"] = ex["prompt"]
            st.session_state["response"] = ex["response"]
            st.session_state["context"] = ex["context"]
    with col2:
        if st.button("Load: Grounded medical summary"):
            ex = examples["grounded_summary"]
            st.session_state["prompt"] = ex["prompt"]
            st.session_state["response"] = ex["response"]
            st.session_state["context"] = ex["context"]
    with col3:
        if st.button("Load: Hallucinated medical summary"):
            ex = examples["hallucinated_medical"]
            st.session_state["prompt"] = ex["prompt"]
            st.session_state["response"] = ex["response"]
            st.session_state["context"] = ex["context"]

    prompt = st.text_area(
        "Prompt",
        value=st.session_state.get("prompt", ""),
        height=100,
        placeholder="The prompt sent to the AI system...",
    )
    response = st.text_area(
        "AI Response",
        value=st.session_state.get("response", ""),
        height=180,
        placeholder="Paste the AI-generated response here...",
    )
    context = st.text_area(
        "Source Context (optional — required for factuality checks)",
        value=st.session_state.get("context", ""),
        height=120,
        placeholder="Ground-truth document, job description, patient chart, etc.",
    )

    if st.button("Run TrustLens Audit", type="primary", disabled=not response.strip()):
        with st.spinner("Analyzing response..."):
            pipeline = get_pipeline()
            result = pipeline.audit(prompt, response, context)
            st.session_state["last_audit"] = result.to_dict()
            st.session_state["last_bias_obj"] = result.bias
            st.session_state["last_hall_obj"] = result.hallucination

    if "last_audit" not in st.session_state:
        st.info("Paste a response and click **Run TrustLens Audit** to begin.")
        return

    audit = st.session_state["last_audit"]
    bias_obj = st.session_state["last_bias_obj"]
    hall_obj = st.session_state["last_hall_obj"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_trust_gauge(audit["trust"]["trust_score"])
    with c2:
        st.metric("Bias Risk", f"{audit['bias']['bias_score']:.2f}", help="Lower is better")
        st.metric("Toxicity", f"{audit['bias']['toxicity']:.2f}")
    with c3:
        st.metric("Factuality", f"{audit['hallucination']['factuality_score']:.2f}", help="Higher is better")
        if not audit["hallucination"]["has_context"]:
            st.caption("⚠️ No context — factuality is estimated")
    with c4:
        st.metric("Fairness Risk", f"{audit['fairness_risk']:.2f}", help="From counterfactual tab")

    domain = audit["trust"].get("domain", "general")
    caps = audit["trust"].get("caps_applied")
    st.caption(f"Domain: **{domain}**" + (f" | Caps applied: {', '.join(caps)}" if caps else ""))

    st.subheader("Highlighted Response")
    st.markdown(
        highlight_flagged_spans(response, bias_obj.flagged_spans),
        unsafe_allow_html=True,
    )

    if hall_obj.verifications:
        st.subheader("Claim Factuality")
        claim_status = {v.claim: v.status for v in hall_obj.verifications}
        st.markdown(
            highlight_claims(response, claim_status),
            unsafe_allow_html=True,
        )

    with st.expander("Detailed breakdown"):
        st.json(audit)

    md_report = audit_to_markdown(audit)
    st.download_button(
        "Download Markdown Report",
        md_report,
        file_name="trustlens_audit.md",
        mime="text/markdown",
    )
    if st.button("Save JSON report"):
        path = export_audit_json(audit)
        st.success(f"Saved to {path}")


def tab_counterfactual() -> None:
    st.header("Counterfactual Fairness Lab")
    st.markdown(
        "Compare AI responses across prompt variants that differ only by a protected attribute."
    )

    scenarios = load_scenarios()
    scenario_labels = {
        s.scenario_id: f"{s.scenario_id} — {s.domain} ({s.protected_attribute})"
        for s in scenarios
    }
    selected_id = st.selectbox(
        "Select scenario",
        options=list(scenario_labels.keys()),
        format_func=lambda x: scenario_labels[x],
    )
    scenario = next(s for s in scenarios if s.scenario_id == selected_id)
    st.caption(scenario.description)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Group A: {scenario.group_a}**")
        st.code(scenario.prompt_a, language=None)
        response_a = st.text_area(
            f"Response for {scenario.group_a}",
            key=f"resp_a_{selected_id}",
            height=150,
        )
    with col_b:
        st.markdown(f"**Group B: {scenario.group_b}**")
        st.code(scenario.prompt_b, language=None)
        response_b = st.text_area(
            f"Response for {scenario.group_b}",
            key=f"resp_b_{selected_id}",
            height=150,
        )

    if st.button("Evaluate Fairness", type="primary"):
        if not response_a.strip() or not response_b.strip():
            st.warning("Provide both responses to compare.")
            return

        detector = BiasDetector()
        bias_a = detector.analyze(response_a).bias_score
        bias_b = detector.analyze(response_b).bias_score

        evaluator = CounterfactualEvaluator()
        cf_result = evaluator.evaluate_pair(
            scenario_id=selected_id,
            response_a=response_a,
            response_b=response_b,
            group_a=scenario.group_a,
            group_b=scenario.group_b,
            bias_scores={scenario.group_a: bias_a, scenario.group_b: bias_b},
        )

        st.session_state["cf_result"] = cf_result.to_dict()

    if "cf_result" not in st.session_state:
        return

    cf = st.session_state["cf_result"]
    disp = cf["disparity"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("SPD", f"{disp['statistical_parity_difference']:.3f}")
    m2.metric("DIR", f"{disp['disparate_impact_ratio']:.3f}")
    m3.metric("Fairness Risk", f"{disp['fairness_risk']:.3f}")
    if "max_bias_gap" in disp:
        m4.metric("Max Bias Gap", f"{disp['max_bias_gap']:.3f}")

    st.subheader("Group Comparison")

    df = pd.DataFrame(
        {
            "Group": list(cf["group_scores"].keys()),
            "Positive Framing Score": list(cf["group_scores"].values()),
            "Bias Score": [cf["group_bias_scores"][g] for g in cf["group_scores"]],
        }
    )
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df.set_index("Group")[["Positive Framing Score", "Bias Score"]])


def tab_about() -> None:
    st.header("Methodology")
    st.markdown(
        """
        ### What TrustLens measures

        | Dimension | What we check | Key signals |
        |-----------|---------------|-------------|
        | **Bias** | Demographic toxicity & stereotypes | Toxicity classifier, stereotype patterns, descriptor tone |
        | **Factuality** | Grounding in source context | NLI entailment, semantic similarity, claim extraction |
        | **Fairness** | Counterfactual parity | SPD, DIR on positive-framing scores across groups |

        ### Trust score formula

        Hybrid scoring blends additive and multiplicative components, then applies severity caps:

        ```
        combined = blend × additive + (1 − blend) × multiplicative
        trust = 100 × combined  (capped when factuality/bias/fairness fail severely)
        ```

        Domain-aware weights (e.g. medical boosts factuality to 65%). Severity caps prevent
        a score above 20–25 when all claims contradict the source context.

        ### Bias types in GenAI

        - **Representation bias** — skewed portrayals of demographic groups
        - **Historical bias** — patterns inherited from training data
        - **Evaluation bias** — inconsistent quality across counterfactual variants

        ### Ethical limitations

        - Toxicity models can misflag dialect and reclaimed language
        - Without source context, open-domain fact-checking is unreliable
        - Counterfactual tests depend on prompt design choices
        - Trust scores are **decision-support**, not safety certification
        """
    )


def main() -> None:
    sidebar_info()
    tab1, tab2, tab3 = st.tabs(["Audit", "Counterfactual Lab", "About"])
    with tab1:
        tab_audit()
    with tab2:
        tab_counterfactual()
    with tab3:
        tab_about()


if __name__ == "__main__":
    main()

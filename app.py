import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import requests
import time
from ai_module import (
    clean_turbine_data,
    add_engineered_features,
    inject_synthetic_anomalies,
    split_data,
    grid_search_isolation_forest,
    train_final_model,
    prepare_input_data,
    predict_single_turbine,
    summarize_prediction,
    evaluate_predictions,
    check_design_violations,
    get_chatgpt_summary,
    get_training_stats,
    visualize_feature_distributions,
    compute_feature_anomaly_correlation,
    show_top_correlated_features,
    plot_top_feature_relationships,
    plot_feature_correlation_heatmap,
)

zapier_webhook_url = st.secrets.get("ZAPIER_WEBHOOK_URL")
if not zapier_webhook_url:
    raise RuntimeError("ZAPIER_WEBHOOK_URL is not set in Streamlit Secrets or env.")

st.set_page_config(page_title="Permit Flow", page_icon="🌀", layout="wide")
if "btn_lock_until" not in st.session_state:
    st.session_state.btn_lock_until = 0


@st.cache_resource(show_spinner=True)
def train_model():
    df = clean_turbine_data(filepath="wind_turbines.csv", save_cleaned=False)
    df = inject_synthetic_anomalies(df)
    df = add_engineered_features(df)

    X_train, X_test, y_train, y_test = split_data(df)

    param_grid = {
        "n_estimators": [100, 200],
        "max_samples": ["auto", 0.5],
        "contamination": [0.05],
        "max_features": [1.0],
    }
    best_params = grid_search_isolation_forest(X_train, X_test, y_test, param_grid)
    model = train_final_model(X_train, best_params)

    mean, std = get_training_stats(df)

    y_pred = model.predict(X_test)
    y_pred_binary = np.array([1 if p == -1 else 0 for p in y_pred])
    metrics = evaluate_predictions(y_test, y_pred)
    return model, mean, std, df, (X_train, X_test, y_train, y_test), metrics


model, training_mean, training_std, train_df, splits, test_metrics = train_model()
X_train, X_test, y_train, y_test = splits


def notify_slack_anomaly(
    turbine_name: str,
    summary: str,
    violations: list,
    anomaly_score: float,
    workspace: dict,
):
    payload = {
        "name": turbine_name,
        "message": summary,
        "anomaly_score": round(anomaly_score, 4),
        "violations": violations or [],
        "city": workspace.get("city", ""),
        "country": workspace.get("country", ""),
        "notes": workspace.get("notes", ""),
    }
    try:
        resp = requests.post(zapier_webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        st.success("Slack notified about anomaly", icon="✅")
    except requests.RequestException as e:
        st.warning(f"Couldn't notify Slack (Zapier webhook): {e}")


if "history" not in st.session_state:
    st.session_state.history = []
if "workspace" not in st.session_state:
    st.session_state.workspace = {"city": "", "country": "", "notes": ""}


st.sidebar.markdown("### Permit Flow")
nav = st.sidebar.radio(
    "Navigate",
    ["Home", "Anomaly Detection", "View Archive", "Contact Us"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Training Snapshot**",
    help="Model trained on real wind turbine data and anomalies.",
)
st.sidebar.caption(f"- Test Precision: **{test_metrics['precision']:.2f}**")
st.sidebar.caption(f"- Test Recall: **{test_metrics['recall']:.2f}**")
st.sidebar.caption(f"- Test F1: **{test_metrics['f1']:.2f}**")


def violation_severity_map(v: str) -> str:
    if v.startswith("Total height"):
        return "High"
    if v.startswith("Hub height") or v.startswith("Rotor diameter"):
        return "Medium"
    if v.startswith("Swept area"):
        return "Low"
    return "Low"


def severity_badge(sev: str) -> str:
    if sev == "High":
        return '<span class="sev-high">High</span>'
    if sev == "Medium":
        return '<span class="sev-med">Medium</span>'
    return '<span class="sev-low">Low</span>'


def make_report(record: dict) -> str:
    lines = [
        f"Permit Flow — Turbine Analysis Report",
        f"Name/ID: {record['Turbine Name']}",
        f"Location: {record.get('City','')}, {record.get('Country','')}",
        "",
        f"Capacity (kW): {record['Capacity (kW)']}",
        f"Hub Height (m): {record['Hub Height (m)']}",
        f"Rotor Diameter (m): {record['Rotor Diameter (m)']}",
        f"Swept Area (m²): {record['Swept Area (m²)']}",
        f"Total Height (m): {record['Total Height (m)']}",
        "",
        f"Prediction: {record['Prediction']}   |   Anomaly Score: {record['Anomaly Score']}",
        f"Violations: {record['Violations']}",
        "",
        "AI Summary:",
        record["GPT Summary"],
        "",
        "Notes:",
        record.get("Notes", ""),
    ]
    return "\n".join(lines)


if nav == "Home":
    st.markdown("## Seamlessly bridging design to permitting.")
    st.write(
        "AI analyzes CAD and specs to speed up the permitting process while keeping teams in sync."
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("⚡ Faster Permitting")
        st.caption("Compress review cycles and reduce waiting time.")
    with c2:
        st.subheader("🧭 Automated Compliance")
        st.caption("Instant checks against practical design rules.")
    with c3:
        st.subheader("🧩 Operational Efficiency")
        st.caption("Centralized data + repeatable workflows.")

    st.write("")
    st.markdown("### Real Energy Data — Quick Glance")
    with st.expander("Show training feature distributions & relationships"):

        corrs = compute_feature_anomaly_correlation(model, X_train)
        scores = model.decision_function(X_train)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            plot_feature_correlation_heatmap(corrs, use_streamlit=True)

        figs = plot_top_feature_relationships(
            X_train, scores, corrs, top_n=3, use_streamlit=False 
        )

        with c2:
            st.pyplot(figs[0])
        with c3:
            st.pyplot(figs[1])
        with c4:
            st.pyplot(figs[2])

elif nav == "Anomaly Detection":
    st.markdown("### Smart Design Workplace")

    st.markdown("#### Location Metadata")
    colA, colB = st.columns(2)
    with colA:
        st.session_state.workspace["city"] = st.text_input(
            "City", value=st.session_state.workspace.get("city", "")
        )
    with colB:
        st.session_state.workspace["country"] = st.text_input(
            "Country", value=st.session_state.workspace.get("country", "")
        )
    st.session_state.workspace["notes"] = st.text_area(
        "Location Specifics / Notes", value=st.session_state.workspace.get("notes", "")
    )

    turbine_name = st.text_input("Turbine Name / ID", placeholder="e.g., T-001")
    capacity = st.number_input("Capacity (kW)", min_value=0.0, value=2000.0, step=100.0)
    hub_height = st.number_input("Hub Height (m)", min_value=0.0, value=90.0, step=1.0)
    rotor_diameter = st.number_input(
        "Rotor Diameter (m)", min_value=0.0, value=100.0, step=1.0
    )
    swept_area = st.number_input(
        "Swept Area (m²)", min_value=0.0, value=7850.0, step=10.0
    )
    total_height = st.number_input(
        "Total Height (m)", min_value=0.0, value=140.0, step=1.0
    )

    locked = time.time() < st.session_state.btn_lock_until
    run = st.button(
        "Run Compliance Check",
        type="primary",
        use_container_width=True,
        disabled=locked,
    )

    if run and not locked:
        st.session_state.btn_lock_until = time.time() + 1.0
        if not turbine_name:
            st.error("Please enter a turbine name or ID.")
        else:
            turbine_input = prepare_input_data(
                capacity=capacity,
                hub_height=hub_height,
                rotor_diameter=rotor_diameter,
                swept_area=swept_area,
                total_height=total_height,
                training_mean=training_mean,
                training_std=training_std,
            )
            prediction = predict_single_turbine(model, turbine_input)
            anomaly_score = float(model.decision_function(turbine_input)[0])

            violations = check_design_violations(
                capacity, hub_height, rotor_diameter, swept_area, total_height
            )
            v_rows = []
            for v in violations:
                sev = violation_severity_map(v)
                v_rows.append({"Issue": v, "Severity": sev})

            summary = get_chatgpt_summary(
                turbine_input, prediction, anomaly_score, violations
            )

            st.markdown("---")
            cols = st.columns([1, 2])
            with cols[0]:
                if prediction == -1:
                    st.error("Anomaly detected")
                    pred_label = "Anomaly"
                else:
                    st.success("Normal turbine design")
                    pred_label = "Normal"
                st.metric("Anomaly Score", f"{anomaly_score:.4f}")
            with cols[1]:
                st.markdown("**AI Summary**")
                st.info(summary)

            st.markdown("#### Regulation Checker")
            if v_rows:
                dfv = pd.DataFrame(v_rows)
                dfv["Severity Badge"] = dfv["Severity"].apply(
                    lambda s: severity_badge(s)
                )
                st.write(
                    dfv[["Issue", "Severity Badge"]].to_html(escape=False, index=False),
                    unsafe_allow_html=True,
                )
            else:
                st.success("No engineering rule violations detected.")

            if prediction == -1:
                pred_label = "Anomaly"
                notify_slack_anomaly(
                    turbine_name,
                    summary,
                    violations,
                    anomaly_score,
                    st.session_state.workspace,
                )
            else:
                pred_label = "Normal"

            st.markdown("#### Context — Real Data Scatter")
            with st.spinner("Scoring dataset for context..."):

                ref = add_engineered_features(clean_turbine_data(save_cleaned=False))
                Xref = ref[[c for c in turbine_input.columns]]
                ref_scores = model.decision_function(Xref)
                ref_pred = model.predict(Xref)
                ref_plot = ref.copy()
                ref_plot["score"] = ref_scores
                ref_plot["is_anom"] = ref_pred == -1

                ref_plot = ref_plot.sample(min(1500, len(ref_plot)), random_state=42)

                pt = {
                    "capacity_kw": capacity,
                    "hub_height_m": hub_height,
                    "rotor_diameter_m": rotor_diameter,
                    "swept_area_m2": swept_area,
                    "total_height_m": total_height,
                    "score": anomaly_score,
                    "is_anom": (prediction == -1),
                }
                pt_df = pd.DataFrame([pt])

                import altair as alt

                base_chart = (
                    alt.Chart(ref_plot)
                    .mark_circle(size=60)
                    .encode(
                        x=alt.X("rotor_diameter_m", title="Rotor Diameter (m)"),
                        y=alt.Y("capacity_kw", title="Capacity (kW)"),
                        color=alt.condition(
                            "datum.is_anom", alt.value("#d62728"), alt.value("#2ca02c")
                        ),
                        tooltip=[
                            "capacity_kw",
                            "hub_height_m",
                            "rotor_diameter_m",
                            "total_height_m",
                            "score",
                            "is_anom",
                        ],
                    )
                )

                highlight_chart = (
                    alt.Chart(pt_df)
                    .mark_point(
                        size=300,
                        shape="diamond",
                        filled=True,
                        color="gold",
                        stroke="black",
                        strokeWidth=2,
                    )
                    .encode(
                        x="rotor_diameter_m",
                        y="capacity_kw",
                        tooltip=[
                            "capacity_kw",
                            "hub_height_m",
                            "rotor_diameter_m",
                            "total_height_m",
                            "score",
                            "is_anom",
                        ],
                    )
                )

                label = (
                    alt.Chart(pt_df)
                    .mark_text(dx=8, dy=-8)
                    .encode(
                        x="rotor_diameter_m",
                        y="capacity_kw",
                        text=alt.value("Your input"),
                    )
                )

                chart = (
                    (base_chart + highlight_chart + label)
                    .interactive()
                    .properties(height=380)
                )

                st.altair_chart(chart, use_container_width=True)

            record = {
                "Turbine Name": turbine_name,
                "City": st.session_state.workspace.get("city", ""),
                "Country": st.session_state.workspace.get("country", ""),
                "Notes": st.session_state.workspace.get("notes", ""),
                "Capacity (kW)": capacity,
                "Hub Height (m)": hub_height,
                "Rotor Diameter (m)": rotor_diameter,
                "Swept Area (m²)": swept_area,
                "Total Height (m)": total_height,
                "Prediction": pred_label,
                "Anomaly Score": round(anomaly_score, 4),
                "Violations": (
                    "; ".join(
                        [
                            f"{violation} (sev: {violation_severity_map(violation)})"
                            for violation in violations
                        ]
                    )
                    if violations
                    else "None"
                ),
                "GPT Summary": summary,
            }
            st.session_state.history.append(record)

            st.markdown("---")
            st.markdown("#### Export")
            txt = make_report(record)
            st.download_button(
                "Download Report (.txt)",
                data=txt,
                file_name=f"{turbine_name}_report.txt",
            )
            hist_df = pd.DataFrame(st.session_state.history)
            st.download_button(
                "Download History (.csv)",
                data=hist_df.to_csv(index=False),
                file_name="turbine_history.csv",
            )

elif nav == "View Archive":
    st.markdown("### All Analyses")
    if not st.session_state.history:
        st.info("No turbines have been analyzed yet.")
    else:
        dfh = pd.DataFrame(st.session_state.history)
        st.dataframe(dfh, use_container_width=True)
        st.download_button(
            "Download History (.csv)",
            data=dfh.to_csv(index=False),
            file_name="turbine_history.csv",
        )

elif nav == "Contact Us":
    st.markdown("### Contact Us")
    st.write("For inquiries, feedback, or support, please reach out to us")
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    message = st.text_area("Message")

    if st.button("Send"):
        if not name or not email or not message:
            st.error("Please fill in all fields before sending.")
        else:
            st.success("✅ Thank you for reaching out! We will get back to you soon.")

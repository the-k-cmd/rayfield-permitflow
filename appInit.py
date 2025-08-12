import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix


from ai_module import (
    clean_turbine_data, add_engineered_features, inject_synthetic_anomalies,
    split_data, grid_search_isolation_forest, train_final_model, 
    prepare_input_data, predict_single_turbine, summarize_prediction, evaluate_predictions, check_design_violations, 
    get_chatgpt_summary, get_training_stats
)


# Train the model once when the web app starts
def train_model():
    df = clean_turbine_data(save_cleaned=False)
    df = add_engineered_features(df)
    df = inject_synthetic_anomalies(df)

    X_train, X_test, y_train, y_test = split_data(df)

    param_grid = {
        'n_estimators': [100, 200],
        'max_samples': ['auto', 0.5],
        "contamination": [0.05],
        "max_features": [1.0]
    }

    best_params = grid_search_isolation_forest(X_train, X_test, y_test, param_grid)
    model = train_final_model(X_train, best_params)

    mean, std = get_training_stats(df)

    # Predictions on test set
    y_pred = model.predict(X_test)

    # Convert predictions: -1 → anomaly (1), 1 → normal (0)
    y_pred_binary = [1 if p == -1 else 0 for p in y_pred]

    # Evaluation
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_binary))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_binary, target_names=["Normal", "Anomaly"]))


    return model, mean, std

# Train model
model, training_mean, training_std = train_model()

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Streamlit UI

# Navigation bar
st.set_page_config(page_title="Permit Flow", layout="wide")
page = st.sidebar.radio("Navigate", ["Analyze Turbine", "View All"])

if page == "Analyze Turbine":
    st.title("Permit Flow - Seamlessly bridging design to permitting.")
    st.write("Enter turbine specifications below to check for anomalies and compliance.")

    # Take input
    turbine_name = st.text_input("Turbine Name / ID", placeholder="e.g., T-001")

    capacity = st.number_input("Capacity (kW)", min_value=0.0, value=2000.0, step=100.0)
    hub_height = st.number_input("Hub Height (m)", min_value=0.0, value=90.0, step=1.0)
    rotor_diameter = st.number_input("Rotor Diameter (m)", min_value=0.0, value=100.0, step=1.0)
    swept_area = st.number_input("Swept Area (m²)", min_value=0.0, value=7850.0, step=10.0)
    total_height = st.number_input("Total Height (m)", min_value=0.0, value=140.0, step=1.0)

    # Pass to preperation function if button clicked
    if st.button("Run Prediction"):
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
                training_std=training_std
            )

            prediction = predict_single_turbine(model, turbine_input)
            anomaly_score = model.decision_function(turbine_input)[0]
            violations = check_design_violations(capacity, hub_height, rotor_diameter, swept_area, total_height)
            summary = get_chatgpt_summary(turbine_input, prediction, anomaly_score, violations)

            # Display results
            st.subheader("Analyzed Result")
            if prediction == -1:
                st.error("Anomaly detected")
                pred_label = "Anomaly"
            else:
                st.success("Normal turbine design")
                pred_label = "Normal"

            st.write(f"**Anomaly Score:** {anomaly_score:.4f}")
            if violations:
                st.warning("Engineering Violations:\n" + "\n".join(violations))

            st.subheader("AI Summary")
            st.info(summary)

            st.session_state.history.append({
                    "Turbine Name": turbine_name,
                    "Capacity (kW)": capacity,
                    "Hub Height (m)": hub_height,
                    "Rotor Diameter (m)": rotor_diameter,
                    "Swept Area (m²)": swept_area,
                    "Total Height (m)": total_height,
                    "Prediction": pred_label,
                    "Anomaly Score": round(anomaly_score, 4),
                    "Violations": "; ".join(violations) if violations else "None",
                    "GPT Summary": summary
                })


elif page == "View All":
    st.title("Permit Flow - Seamlessly bridging design to permitting.")
    st.write("View all of your company's past turbine design analyses in one place.")

    for i, record in enumerate(st.session_state.history):
            with st.container():
                cols = st.columns([2, 2, 2, 2, 1])

                # Turbine Name
                cols[0].write(f"**{record['Turbine Name']}**")

                # Prediction
                pred_color = "🟢" if record["Prediction"] == "Normal" else "🔴"
                cols[1].write(f"{pred_color} {record['Prediction']}")

                # Violations
                cols[2].write("None" if record["Violations"] == "None" else f"{record['Violations']}")

                # Expandable Summary
                with cols[3].expander("Summary"):
                    st.write(record["GPT Summary"])

                # View Specs Button
                if cols[4].button("View Specs", key=f"view_{i}"):
                    st.session_state.selected_turbine = record
                    st.session_state.page = "Specs"
                    st.rerun()

    else:
        st.info("No turbines have been analyzed yet.")

if "page" in st.session_state and st.session_state.page == "Specs":
    record = st.session_state.selected_turbine
    st.title(f"Turbine Specs - {record['Turbine Name']}")
    st.table({
        "Capacity (kW)": [record["Capacity (kW)"]],
        "Hub Height (m)": [record["Hub Height (m)"]],
        "Rotor Diameter (m)": [record["Rotor Diameter (m)"]],
        "Swept Area (m²)": [record["Swept Area (m²)"]],
        "Total Height (m)": [record["Total Height (m)"]],
        "Prediction": [record["Prediction"]],
        "Anomaly Score": [record["Anomaly Score"]],
        "Violations": [record["Violations"]]
    })
    if st.button("Back to List"):
        st.session_state.page = "View All"
        st.rerun()

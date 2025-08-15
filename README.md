# Wind Turbine Anomaly Detection

A machine learning pipeline for detecting unusual wind turbine configurations, helping engineers identify potential issues early to improve efficiency and safety

---

## Project Overview
This project detects anomalies in turbine performance using **Isolation Forest**. Users can input turbine specifications through a **Streamlit frontend**, see visual comparisons to the dataset, and receive a **ChatGPT-generated summary** when anomalies are detected. **Zapier notifications** alert stakeholders automatically through Slack

---

## Features
- **Interactive Frontend:** Input turbine specs and receive instant anomaly predictions
- **Visualizations:** Altair, Matplotlib, and Seaborn charts for trend and feature comparisons
- **Backend Processing:** Dataset cleaning, synthetic anomaly injection (5% of turbines), and feature engineering
- **Engineered Features:** Rotor-to-height ratio, capacity-to-rotor ratio, and other meaningful derived metrics
- **Model:** Isolation Forest for unsupervised anomaly detection
- **Evaluation:** Precision, recall, and F1 calculated using synthetic anomalies

---

## Dataset
- Source: CORGIS Wind Turbines Dataset  
- Preprocessing: Missing values handled, unnecessary columns removed, and data types converted
- Perturbing Real Data: Injecting small controlled anomalies into valid turbine designs (values that just exceed known thresholds or break design rules)

---

## Model & Method
- **Algorithm:** Isolation Forest chosen for unsupervised anomaly detection
- **Synthetic Anomalies:** Injected into 5% of turbines to simulate unusual or guideline-exceeding configurations
- **Train/Test Split:** Ensures generalization, synthetic anomalies in test set allow evaluation
- **Feature Scaling:** Standardized to prevent bias from differing magnitudes

---

## Frontend & Integration
- **Streamlit UI:** Rapid prototyping with interactive inputs, visualizations, and real-time predictions 
- **ChatGPT Summary:** Explains why a turbine is flagged anomalous and highlights which metrics deviate
- **Zapier Notifications:** Triggered automatically when anomalies are detected

---

## Prototypes
- **Low-Fidelity:** [Figma Low-Fidelity Prototype](https://www.figma.com/design/lfldGk2tLeKLEyeLj5Fiyi/Low-fidelity-prototype?node-id=0-1&t=9uj1vhmZt9U3tO1J-1)  
- **High-Fidelity Design:** [Figma High-Fidelity Design](https://www.figma.com/design/BOoc7sa7qoimscir7yHfZa/High-fidelity-prototype-by-Adham-Kayed?node-id=0-1&t=5xARh3OmmXGTGBBv-1)  
- **Interactive Prototype:** [High-Fidelity Prototype](https://www.figma.com/proto/BOoc7sa7qoimscir7yHfZa/High-fidelity-prototype-by-Adham-Kayed?node-id=2040-161&p=f&t=8WSbTCpF4om4FO8b-1&scaling=contain&content-scaling=fixed&page-id=0%3A1)  

---

## Team & Workflow
- **Team Members:** Harshitha, Adham, Kiyan, Alonso
- **Week 1:** Individual deliverables to align on project vision
- **Week 2:** Dataset selection, cleaning, and preprocessing
- **Week 3:** ML model development and backend integration
- **Week 4:** Frontend completion and dashboard integration
- **Week 5:** Future development planning and presentation preparation

---

## Limitations & Future Work
- **CAD Drag & Drop:** Backend ready, frontend integration pending (full code on CAD-Future branch)
- **Model Accuracy:** Could improve with more real anomalies, expanded synthetic dataset, or experiment with different models
- **Next Steps:** Expand synthetic anomalies, experiment with different datasets, add engineered features, explore ensemble or autoencoder-based models, rebuild frontend as a full-featured web app beyond Streamlitand, and adapt to new turbine types or regions

---

## Key Learnings
- **Technical:** Data preprocessing, feature engineering, ML pipeline integration
- **Soft Skills:** Teamwork, communication, aligning on shared vision
- **Industry Insight:** Understanding gaps in wind energy engineering workflows and regulatory considerations

---

## GitHub & Resources
- **Initial AI/ML Pipeline:** [Pipeline Diagram](https://drive.google.com/file/d/1URmhdtMe8t9sX95ouDqOLQzJ-9exz9g2/view) 
- **User Workflow:** [Miro Board](https://miro.com/welcomeonboard/U2QrTHA4UDlTNlJWdWRRcjNaV2RKcFdWbEdMc3FRY2tZeCtvcVVZMWtLK2FUaEFkeCt6S3U0WXYxRGFMWW9tMkt6dkZ4dFJJaEtWRllYQU9JVVVFS2oyNlZaU1oxVkVTL2tERFFTUld3eGRlUWpFQm9Hb0JiaGJQcDQ5eWljcW5Bd044SHFHaVlWYWk0d3NxeHNmeG9BPT0hdjE=?share_link_id=884020073028)  

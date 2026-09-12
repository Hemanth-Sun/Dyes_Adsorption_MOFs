import io
import joblib
import pandas as pd
import streamlit as st

# --- 1. Page Configuration ---
st.set_page_config(page_title="MOF Adsorption Predictor", layout="wide")

# Styling for forced light background and clean UI
st.markdown(
    """
    <style>
        /* Force background to white */
        .stApp {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
        }

        /* 1. Force Input Field Labels to Dark Blue/Black */
        label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {
            color: #0F172A !important;
            font-weight: 600 !important;
        }

        /* 2. Force Streamlit Metrics to Dark Text */
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
            color: #0F172A !important;
        }

        /* 3. Input Text Boxes Styling */
        input[type="number"] {
            text-align: center !important;
            background-color: #F8FAFC !important;
            color: #0F172A !important;
            border: 1px solid #94A3B8 !important;
            font-weight: 600 !important;
        }

        /* 4. Hide + / - step buttons */
        button[title="Step up"], button[title="Step down"],
        button[aria-label="Step up"], button[aria-label="Step down"],
        [data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {
            display: none !important;
        }

        /* 5. Hide standard spin arrows */
        input[type="number"]::-webkit-outer-spin-button,
        input[type="number"]::-webkit-inner-spin-button {
            -webkit-appearance: none !important;
            margin: 0 !important;
        }

        /* 6. Fix Inline Code Tags (`.csv`, `.xlsx`) */
        code {
            background-color: #F1F5F9 !important;
            color: #0284C7 !important;
            border: 1px solid #CBD5E1 !important;
            font-weight: bold !important;
        }

        /* 7. Fix File Uploader Container & Dropzone */
        [data-testid="stFileUploader"] {
            background-color: #FFFFFF !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            background-color: #F8FAFC !important;
            border: 2px dashed #94A3B8 !important;
            color: #0F172A !important;
        }

        [data-testid="stFileUploaderDropzone"] span, 
        [data-testid="stFileUploaderDropzone"] div,
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzone"] p {
            color: #334155 !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #94A3B8 !important;
        }

        /* Make primary action buttons pop */
        button[kind="primary"] {
            font-weight: bold !important;
            box-shadow: 0 4px 6px rgba(0, 82, 204, 0.2) !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)


# --- 2. Load Frozen Artifacts ---
@st.cache_resource
def load_artifacts():
    try:
        artifacts = {
            "model": joblib.load("models/gbr_model_MB.joblib"),
            "scaler": joblib.load("models/scaler_MB.joblib"),
            "features": joblib.load("models/feature_names_MB.joblib"),
        }
        return artifacts
    except Exception as e:
        return str(e)


app_data = load_artifacts()

if isinstance(app_data, str):
    st.error(f"⚠️ Artifact Loading Error: {app_data}")
    st.info(
        "Ensure 'gbr_model_MB.joblib', 'scaler_MB.joblib', and 'feature_names_MB.joblib' exist in the 'models/' folder."
    )
    st.stop()

active_model = app_data["model"]
active_scaler = app_data["scaler"]
active_features = app_data["features"]

# --- 3. App Header ---
st.markdown(
    """
    <div style="background-color: #FFFFFF; 
                padding: 25px; 
                border-radius: 12px; 
                margin-bottom: 25px; 
                border: 2px solid #E2E8F0; 
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);">
        <h1 style="color: #0F172A; text-align: center; margin-bottom: 0; font-weight: 800;">MOF Adsorption Prediction Platform</h1>
        <p style="color: #0052CC; text-align: center; font-size: 19px; font-weight: 600; margin-top: 5px;">Methylene Blue (MB) Equilibrium Capacity Estimation (Train-Only GBR Model)</p>
    </div>
""",
    unsafe_allow_html=True,
)

# --- 4. Dashboard Layout ---
left_pane, right_pane = st.columns([1.2, 1], gap="large")

# LEFT PANE: MANUAL FORM ENTRY
with left_pane:
    st.subheader("Single Sample Interface")

    with st.form("prediction_form", border=True):
        st.markdown("**1. MOF Structural Parameters**")
        col1, col2, col3 = st.columns(3)
        with col1:
            bet = st.number_input("BET (m²/g)", min_value=0.0, value=150.0)
        with col2:
            pore_vol = st.number_input(
                "Pore Vol (cm³/g)", min_value=0.0, value=0.05
            )
        with col3:
            pore_size = st.number_input(
                "Pore Size (nm)", min_value=0.0, value=2.5
            )

        st.markdown("**2. Operational Conditions**")
        col4, col5 = st.columns(2)
        with col4:
            dosage = st.number_input("Dosage (g/L)", min_value=0.0, value=1.0)
        with col5:
            temp = st.number_input("Temp (K)", min_value=0.0, value=298.15)
        with col4:
            time = st.number_input("Time (h)", min_value=0.0, value=24.0)

        st.markdown("**3. Solution Chemistry**")
        col6, col7 = st.columns(2)
        with col6:
            c0 = st.number_input(
                "Initial Conc. (mg/L)", min_value=0.0, value=50.0
            )
        with col7:
            ph = st.number_input(
                "pH", min_value=0.0, max_value=14.0, value=7.0
            )

        submitted = st.form_submit_button(
            "Compute Adsorption Capacity",
            type="primary",
            use_container_width=True,
        )

# RIGHT PANE: OUTPUTS & BATCH PROCESSING
with right_pane:
    st.subheader("Output Analytics")
    metric_container = st.container(border=True)

    if submitted:
        input_dict = {
            active_features[0]: bet,
            active_features[1]: pore_vol,
            active_features[2]: pore_size,
            active_features[3]: dosage,
            active_features[4]: temp,
            active_features[5]: time,
            active_features[6]: c0,
            active_features[7]: ph,
        }
        input_df = pd.DataFrame([input_dict])[active_features]

        # Transform using the scaler fitted ONLY on X_train
        scaled_input = active_scaler.transform(input_df)
        prediction = active_model.predict(scaled_input)[0]

        with metric_container:
            st.metric(
                label="Predicted $Q_e$ for Methylene Blue",
                value=f"{prediction:.2f} mg/g",
            )
    else:
        with metric_container:
            st.info(
                "Awaiting input parameters. Click 'Compute' to generate $Q_e$ prediction."
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Permanently Exposed High-Throughput Batch Section
    with st.container(border=True):
        st.subheader("High-Throughput Batch Processing")
        st.write("Upload a `.csv` or `.xlsx` matrix to evaluate multiple MOFs.")

        # Display Required Format Directly without dark code box
        st.caption("**Required File Format (Header Columns):**")
        feature_text = ", ".join(active_features)
        st.markdown(
            f"""
            <div style="background-color: #F8FAFC; 
                        padding: 10px 14px; 
                        border-radius: 6px; 
                        border: 1px solid #CBD5E1; 
                        color: #0F172A; 
                        font-family: monospace; 
                        font-size: 13px; 
                        font-weight: 600;
                        margin-bottom: 15px;
                        word-wrap: break-word;">
                {feature_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload Matrix", type=["xlsx", "csv"], label_visibility="collapsed"
        )

        if uploaded_file is not None:
            try:
                df_upload = (
                    pd.read_csv(uploaded_file)
                    if uploaded_file.name.endswith(".csv")
                    else pd.read_excel(uploaded_file)
                )
                missing_cols = [
                    col for col in active_features if col not in df_upload.columns
                ]

                if missing_cols:
                    st.error(f"Missing columns: {', '.join(missing_cols)}")
                else:
                    if st.button(
                        "Execute Batch Evaluation", use_container_width=True
                    ):
                        with st.spinner("Processing matrix..."):
                            process_df = df_upload[active_features]
                            scaled_batch = active_scaler.transform(process_df)
                            batch_predictions = active_model.predict(
                                scaled_batch
                            )

                            results_df = df_upload.copy()
                            results_df["Predicted_Qe_MB_mg_g"] = (
                                batch_predictions
                            )

                            st.success("Evaluation complete.")
                            st.dataframe(results_df.head(3))

                            buffer = io.BytesIO()
                            with pd.ExcelWriter(
                                buffer, engine="openpyxl"
                            ) as writer:
                                results_df.to_excel(writer, index=False)

                            st.download_button(
                                label="Download Results",
                                data=buffer.getvalue(),
                                file_name="Batch_Predictions_MB_GBR.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True,
                            )
            except Exception as e:
                st.error(f"Error reading file: {e}")

# --- 5. Application Footer ---
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("---")

footer_left, footer_right = st.columns(2)

with footer_left:
    st.markdown("### Institution")
    st.markdown(
        "**Department of Chemical Engineering,**<br>"
        "B.M.S. College of Engineering,<br>"
        "Bengaluru, India",
        unsafe_allow_html=True,
    )

with footer_right:
    st.markdown("### About the Tool")
    st.markdown(
        "This materials informatics platform utilizes a **Gradient Boosting Regressor (GBR)** model "
        "trained strictly on independent training set data to predict the Methylene Blue equilibrium adsorption capacity ($Q_e$) of Metal-Organic Frameworks (MOFs)."
    )
import streamlit as st
import pandas as pd
import joblib
import io

# --- 1. Page Configuration ---
st.set_page_config(page_title="MOF Adsorption Predictor", layout="wide")

# Injecting aggressive CSS to completely kill the +/- buttons everywhere
st.markdown("""
    <style>
        /* Nuke the Streamlit step buttons by aria-label, title, and test-id */
        button[title="Step up"], button[title="Step down"],
        button[aria-label="Step up"], button[aria-label="Step down"],
        [data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {
            display: none !important;
        }

        /* Center the text inside the input to make it look like a clean data field */
        input[type="number"] {
            text-align: center !important;
        }

        /* Remove standard browser native spin buttons */
        input[type="number"]::-webkit-outer-spin-button,
        input[type="number"]::-webkit-inner-spin-button {
            -webkit-appearance: none !important;
            margin: 0 !important;
        }
    </style>
""", unsafe_allow_html=True)


# --- 2. Load Frozen Models & Scalers ---
@st.cache_resource
def load_artifacts():
    try:
        artifacts = {
            "Methylene Blue (MB)": {
                "model": joblib.load("models/stacking_model_MB.joblib"),
                "scaler": joblib.load("models/scaler_MB.joblib"),
                "features": joblib.load("models/feature_names_MB.joblib")
            },
            "Methyl Orange (MO)": {
                "model": joblib.load("models/stacking_model_MO.joblib"),
                "scaler": joblib.load("models/scaler_MO.joblib"),
                "features": joblib.load("models/feature_names_MO.joblib")
            }
        }
        return artifacts
    except FileNotFoundError:
        st.error("Missing .joblib files. Please ensure the 'models' folder contains all 6 required files.")
        st.stop()


app_data = load_artifacts()

# --- 3. Main App Header & Global Selector ---
# Custom HTML/CSS Banner matching the Neon/Dark Blue theme
# Also includes CSS to completely hide the + / - buttons on number inputs
st.markdown("""
    <style>
        /* Hide the Streamlit +/- buttons */
        div[data-testid="stNumberInputStepUp"] {display: none !important;}
        div[data-testid="stNumberInputStepDown"] {display: none !important;}

        /* Center the text inside the input box to make it look cleaner without the buttons */
        input[type="number"] {text-align: center;}
    </style>

    <div style="background-color: #0B1B3D; padding: 25px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #39FF14;">
        <h1 style="color: #F8FAFC; text-align: center; margin-bottom: 0;">MOF Adsorption Prediction Platform</h1>
        <p style="color: #39FF14; text-align: center; font-size: 18px; margin-top: 5px;">Intelligent Equilibrium Capacity Estimation</p>
    </div>
""", unsafe_allow_html=True)

with st.container(border=True):
    selected_dye = st.radio(
        "**Select Target Adsorbate Configuration:**",
        ["Methylene Blue (MB)", "Methyl Orange (MO)"],
        horizontal=True
    )
st.markdown("<br>", unsafe_allow_html=True)

# Route variables based on selection
active_model = app_data[selected_dye]["model"]
active_scaler = app_data[selected_dye]["scaler"]
active_features = app_data[selected_dye]["features"]

# --- 4. Dashboard Layout ---
left_pane, right_pane = st.columns([1.2, 1], gap="large")

# ==========================================
# LEFT PANE: MANUAL FORM ENTRY
# ==========================================
with left_pane:
    st.subheader("️Single Sample Interface")

    with st.form("prediction_form", border=True):
        st.markdown("**1. MOF Structural Parameters**")
        col1, col2, col3 = st.columns(3)
        with col1: bet = st.number_input("BET (m²/g)", min_value=0.0, value=150.0)
        with col2: pore_vol = st.number_input("Pore Vol (cm³/g)", min_value=0.0, value=0.05)
        with col3: pore_size = st.number_input("Pore Size (nm)", min_value=0.0, value=2.5)

        st.markdown("**2. Operational Conditions**")
        col4, col5 = st.columns(2)
        with col4: dosage = st.number_input("Dosage (g/L)", min_value=0.0, value=1.0)
        with col5: temp = st.number_input("Temp (K)", min_value=0.0, value=298.15)
        with col4: time = st.number_input("Time (h)", min_value=0.0, value=24.0)

        st.markdown("**3. Solution Chemistry**")
        col6, col7 = st.columns(2)
        with col6: c0 = st.number_input("Initial Conc. (mg/L)", min_value=0.0, value=50.0)
        with col7: ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)

        submitted = st.form_submit_button("Compute Adsorption Capacity", type="primary", use_container_width=True)

# ==========================================
# RIGHT PANE: OUTPUTS & BATCH PROCESSING
# ==========================================
with right_pane:
    st.subheader("Output Analytics")

    metric_container = st.container(border=True)

    if submitted:
        input_dict = {
            active_features[0]: bet, active_features[1]: pore_vol, active_features[2]: pore_size,
            active_features[3]: dosage, active_features[4]: temp, active_features[5]: time,
            active_features[6]: c0, active_features[7]: ph
        }
        input_df = pd.DataFrame([input_dict])

        scaled_input = active_scaler.transform(input_df)
        prediction = active_model.predict(scaled_input)[0]

        with metric_container:
            st.metric(label=f"Predicted $Q_e$ for {selected_dye[:2]}", value=f"{prediction:.2f} mg/g")
    else:
        with metric_container:
            st.info("Awaiting input parameters. Click 'Compute' to generate $Q_e$ prediction.")

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("High-Throughput Batch Processing"):
        st.write("Upload a `.csv` or `.xlsx` matrix to evaluate multiple MOFs.")

        with st.popover("View Required Format"):
            st.code(", ".join(active_features))

        uploaded_file = st.file_uploader("Upload Matrix", type=["xlsx", "csv"], label_visibility="collapsed")

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_upload = pd.read_csv(uploaded_file)
                else:
                    df_upload = pd.read_excel(uploaded_file)

                missing_cols = [col for col in active_features if col not in df_upload.columns]

                if missing_cols:
                    st.error(f"Missing columns: {', '.join(missing_cols)}")
                else:
                    if st.button("Execute Batch Evaluation", use_container_width=True):
                        with st.spinner("Processing matrix..."):
                            process_df = df_upload[active_features]
                            scaled_batch = active_scaler.transform(process_df)
                            batch_predictions = active_model.predict(scaled_batch)

                            results_df = df_upload.copy()
                            results_df[f"Predicted_Qe_{selected_dye[:2]}_mg_g"] = batch_predictions

                            st.success("Evaluation complete.")
                            st.dataframe(results_df.head(3))

                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                results_df.to_excel(writer, index=False)

                            st.download_button(
                                label="Download Results",
                                data=buffer.getvalue(),
                                file_name=f"Batch_Predictions_{selected_dye[:2]}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
            except Exception as e:
                st.error(f"Error reading file: {e}")

# --- 5. Application Footer ---
st.markdown("<br><br><br>", unsafe_allow_html=True)  # Push footer to the bottom
st.markdown("---")  # Visual separator

footer_left, footer_right = st.columns(2)

with footer_left:
    st.markdown("### Institution")
    st.markdown(
        "**Department of Chemical Engineering**<br>"
        "B.M.S. College of Engineering<br>"
        "Bengaluru, India",
        unsafe_allow_html=True
    )

with footer_right:
    st.markdown("### About the Tool")
    st.markdown(
        "This materials informatics platform utilizes a stacked CatBoost and Gradient Boosting Regressor"
        "ensemble to predict the equilibrium adsorption capacity ($Q_e$) of Metal-Organic Frameworks (MOFs)."
    )
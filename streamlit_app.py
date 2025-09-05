import streamlit as st
import os
import json
import pandas as pd
import numpy as np
import re
from datetime import datetime
import sys

# Add the project's root directory to the Python path
# This is a robust way to handle imports in different environments
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.auth import authenticate, get_user_store, register_user, get_current_user
from app.db import Study, Form, Submission, init_db, get_session
from app.form_utils import TEMPLATE_FIELDS, save_form_schema
from app.sample_size import calculate_sample_size_buderer, calculate_auc_n, calculate_epv, calculate_design_effect
from app.analysis import (
    run_roc_analysis, run_delong_test, run_logistic_regression,
    run_linear_regression, run_cox_ph, generate_km_plot,
    plot_calibration_curve, plot_decision_curve
)
from app.reporting import generate_docx_report, generate_pdf_report
from cryptography.fernet import Fernet
from sklearn.metrics import confusion_matrix

# --- Project Settings ---
PROJECT_NAME = "Neonatal Sepsis Biomarker Study"
DOMAIN = "Clinical"
PRIMARY_USE = "Data collection + sample size + analytics + reports"
BINARY_OUTCOME_COL = "Sepsis"
SURVIVAL_COLS = {'duration': 'time_to_death', 'event': 'death_event'}

# --- App Initialization and Setup ---
st.set_page_config(layout="wide", page_title=PROJECT_NAME)

# Create a `data` directory if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("knowledge_notes.md"):
    with open("knowledge_notes.md", "w") as f:
        f.write("# Knowledge Notes\n\n- Start writing your notes here!")

init_db()

# --- Authentication Logic ---
def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.session_state['authenticated'] = True
            st.session_state['username'] = username
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Invalid username or password.")

def register_page():
    st.title("Register New User")
    username = st.text_input("New Username")
    password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    if st.button("Register"):
        if password == confirm_password:
            if register_user(username, password):
                st.success("User registered successfully! You can now log in.")
            else:
                st.error("Username already exists.")
        else:
            st.error("Passwords do not match.")

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    st.sidebar.title("Access")
    choice = st.sidebar.radio("Navigation", ["Login", "Register"])
    if choice == "Login":
        login_page()
    elif choice == "Register":
        register_page()
    st.stop()

# --- Main App Navigation ---
st.sidebar.title(PROJECT_NAME)
current_user = get_current_user()
st.sidebar.write(f"Logged in as: **{current_user}**")
if st.sidebar.button("Logout"):
    st.session_state['authenticated'] = False
    st.session_state['username'] = None
    st.rerun()

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["📚 Knowledge & Notes", "🧰 Form Designer", "📝 Data Entry", "📊 Dataset & Export",
     "📐 Sample Size", "📈 Analysis", "📑 Reports & Exports"]
)

session = get_session()

# --- Pages ---

if page == "📚 Knowledge & Notes":
    st.title("📚 Knowledge & Notes")
    st.markdown("Use this space to jot down notes, protocols, or ideas. The notes are saved automatically.")
    with open("knowledge_notes.md", "r") as f:
        notes = f.read()

    edited_notes = st.text_area("Your notes:", notes, height=600)

    if edited_notes != notes:
        with open("knowledge_notes.md", "w") as f:
            f.write(edited_notes)
        st.success("Notes saved!")

elif page == "🧰 Form Designer":
    st.title("🧰 Form Designer")
    st.markdown("Create a new study and design forms (CRFs).")

    st.subheader("1. Study Management")
    studies = session.query(Study).all()
    study_titles = {s.id: s.title for s in studies}
    selected_study_id = st.selectbox("Select a Study", options=list(study_titles.keys()), format_func=lambda x: study_titles[x])
    
    new_study_title = st.text_input("New Study Title")
    new_study_desc = st.text_area("New Study Description")
    if st.button("Create New Study"):
        if new_study_title:
            new_study = Study(title=new_study_title, description=new_study_desc)
            session.add(new_study)
            session.commit()
            st.success("Study created successfully!")
            st.rerun()
        else:
            st.error("Study title cannot be empty.")
    
    st.markdown("---")
    st.subheader("2. Form (CRF) Designer")
    if selected_study_id:
        form_name = st.text_input("Form Name", "CRF 1")
        
        form_template_option = st.radio("Start with a template?", ("No, start from scratch", "Yes, load template"))
        
        schema = st.session_state.get('current_schema', [])

        if form_template_option == "Yes, load template":
            st.session_state['current_schema'] = TEMPLATE_FIELDS
            schema = TEMPLATE_FIELDS
            st.warning("Template loaded. Edit fields below.")
            
        def render_field_editor(field, index):
            with st.expander(f"Field #{index + 1}: {field['label']}", expanded=True):
                st.markdown(f"**Type:** `{field['type']}`")
                field['label'] = st.text_input("Label", field['label'], key=f"label_{index}")
                field['name'] = st.text_input("Name (unique ID)", field.get('name', f"field_{index}"), key=f"name_{index}")
                field['required'] = st.checkbox("Required", field.get('required', False), key=f"required_{index}")
                
                if field['type'] in ["number", "slider"]:
                    field['min'] = st.number_input("Min", value=field.get('min', 0), key=f"min_{index}")
                    field['max'] = st.number_input("Max", value=field.get('max', 100), key=f"max_{index}")
                
                if field['type'] in ["text"]:
                    field['pattern'] = st.text_input("Regex Pattern (optional)", value=field.get('pattern', ""), key=f"pattern_{index}")

                if field['type'] in ["select", "multiselect"]:
                    choices_str = st.text_area("Choices (comma-separated)", value=", ".join(field.get('choices', [])), key=f"choices_{index}")
                    field['choices'] = [c.strip() for c in choices_str.split(',')]
                
                show_if_field = st.text_input("Show if... Field Name (e.g., 'Mortality')", value=field.get('show_if', {}).get('field', ''), key=f"show_if_field_{index}")
                show_if_value = st.text_input("...equals this value (e.g., '1' or 'M')", value=field.get('show_if', {}).get('value', ''), key=f"show_if_value_{index}")
                
                if show_if_field and show_if_value:
                    field['show_if'] = {'field': show_if_field, 'value': show_if_value}
                else:
                    field.pop('show_if', None)

                if st.button(f"Remove Field #{index + 1}", key=f"remove_{index}"):
                    schema.pop(index)
                    st.session_state['current_schema'] = schema
                    st.rerun()

        for i, field in enumerate(schema):
            render_field_editor(field, i)
        
        st.markdown("---")
        st.subheader("Add New Field")
        
        col1, col2 = st.columns(2)
        with col1:
            new_field_label = st.text_input("New Field Label")
        with col2:
            new_field_type = st.selectbox("Type", ["text", "number", "select", "multiselect", "date", "slider"])

        if st.button("Add Field"):
            if new_field_label:
                new_field = {'label': new_field_label, 'name': new_field_label.lower().replace(' ', '_'), 'type': new_field_type}
                schema.append(new_field)
                st.session_state['current_schema'] = schema
                st.rerun()
            else:
                st.error("New field label cannot be empty.")
        
        st.markdown("---")
        if st.button("Save Form to Study"):
            if form_name:
                save_form_schema(session, selected_study_id, form_name, schema)
                st.success(f"Form '{form_name}' saved successfully!")
                st.session_state['current_schema'] = [] # Reset schema
                st.rerun()
            else:
                st.error("Form name cannot be empty.")
    else:
        st.warning("Please create a study first.")

elif page == "📝 Data Entry":
    st.title("📝 Data Entry")
    st.markdown("Enter data for a selected form.")

    studies = session.query(Study).all()
    study_titles = {s.id: s.title for s in studies}
    selected_study_id = st.selectbox("Select a Study", options=list(study_titles.keys()), format_func=lambda x: study_titles[x])
    
    if selected_study_id:
        forms = session.query(Form).filter(Form.study_id == selected_study_id).all()
        form_names = {f.id: f.name for f in forms}
        selected_form_id = st.selectbox("Select a Form", options=list(form_names.keys()), format_func=lambda x: form_names[x])

        if selected_form_id:
            form = session.query(Form).filter_by(id=selected_form_id).first()
            schema = json.loads(form.schema_json)
            
            st.subheader(f"Data Entry for: {form.name}")
            
            submission_data = {}
            valid_submission = True

            # Use a dictionary to store the current state of all input widgets
            input_values = {}
            for field in schema:
                name = field['name']
                input_values[name] = st.session_state.get(f"entry_{name}", None)

            # Render fields with skip-logic
            for field in schema:
                name = field['name']
                field_label = field['label']
                field_type = field['type']
                required = field.get('required', False)
                
                # Apply skip-logic
                show = True
                if 'show_if' in field:
                    show_if_field = field['show_if']['field']
                    show_if_value = field['show_if']['value']
                    
                    # Check if the condition field exists and its value matches
                    if show_if_field in input_values and str(input_values[show_if_field]) != str(show_if_value):
                        show = False
                
                if show:
                    try:
                        if field_type == 'text':
                            value = st.text_input(field_label, key=f"entry_{name}")
                            if required and not value:
                                valid_submission = False
                                st.warning(f"{field_label} is required.")
                            if 'pattern' in field and field['pattern'] and not re.match(field['pattern'], value):
                                valid_submission = False
                                st.warning(f"{field_label} does not match the required pattern.")
                            submission_data[name] = value
                        elif field_type == 'number':
                            value = st.number_input(field_label, min_value=field.get('min'), max_value=field.get('max'), key=f"entry_{name}")
                            submission_data[name] = value
                        elif field_type == 'select':
                            value = st.selectbox(field_label, options=field['choices'], key=f"entry_{name}")
                            submission_data[name] = value
                        elif field_type == 'multiselect':
                            value = st.multiselect(field_label, options=field['choices'], key=f"entry_{name}")
                            submission_data[name] = value
                        elif field_type == 'date':
                            value = st.date_input(field_label, datetime.now().date(), key=f"entry_{name}")
                            submission_data[name] = str(value)
                        elif field_type == 'slider':
                            value = st.slider(field_label, min_value=field.get('min'), max_value=field.get('max'), key=f"entry_{name}")
                            submission_data[name] = value
                    except Exception as e:
                        st.error(f"Error rendering field {field_label}: {e}")
                        valid_submission = False
            
            if st.button("Submit Data"):
                if valid_submission:
                    try:
                        new_submission = Submission(form_id=selected_form_id, data_json=json.dumps(submission_data))
                        session.add(new_submission)
                        session.commit()
                        st.success("Data submitted successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error saving submission: {e}")
                else:
                    st.error("Please correct the errors in the form before submitting.")
        else:
            st.warning("Please create a form for this study first.")
    else:
        st.warning("Please create a study first.")

elif page == "📊 Dataset & Export":
    st.title("📊 Dataset & Export")
    st.markdown("View all submitted data and export it as a CSV file.")
    
    studies = session.query(Study).all()
    study_titles = {s.id: s.title for s in studies}
    selected_study_id = st.selectbox("Select a Study", options=list(study_titles.keys()), format_func=lambda x: study_titles[x])
    
    if selected_study_id:
        forms = session.query(Form).filter(Form.study_id == selected_study_id).all()
        form_names = {f.id: f.name for f in forms}
        selected_form_id = st.selectbox("Select a Form", options=list(form_names.keys()), format_func=lambda x: form_names[x])

        if selected_form_id:
            submissions = session.query(Submission).filter(Submission.form_id == selected_form_id).all()
            if submissions:
                all_data = [json.loads(s.data_json) for s in submissions]
                df = pd.DataFrame(all_data)
                
                st.subheader("Dataset Preview")
                st.dataframe(df)

                csv_file = df.to_csv(index=False).encode('utf-8')
                
                st.markdown("---")
                st.subheader("Export Options")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download as CSV",
                        data=csv_file,
                        file_name=f"{form_names[selected_form_id]}_data.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    st.subheader("Secure Export (Fernet Encryption)")
                    if 'fernet_key' not in st.session_state:
                        st.session_state['fernet_key'] = Fernet.generate_key().decode()
                    
                    st.text_area("Your Encryption Key (SAVE THIS!)", st.session_state['fernet_key'], height=100)
                    
                    fernet = Fernet(st.session_state['fernet_key'].encode())
                    encrypted_data = fernet.encrypt(csv_file)
                    
                    st.download_button(
                        label="Download Encrypted .enc",
                        data=encrypted_data,
                        file_name=f"{form_names[selected_form_id]}_data.enc",
                        mime="application/octet-stream"
                    )
            else:
                st.warning("No submissions found for this form.")
        else:
            st.warning("Please select a form with data.")
    else:
        st.warning("Please select a study.")

elif page == "📐 Sample Size":
    st.title("📐 Sample Size Planning")
    st.markdown("Calculate the required sample size for your study design.")
    
    st.markdown("---")
    st.subheader("Diagnostic Accuracy (Buderer's Method)")
    se_target = st.number_input("Target Sensitivity (%)", min_value=0.0, max_value=100.0, value=95.0) / 100
    sp_target = st.number_input("Target Specificity (%)", min_value=0.0, max_value=100.0, value=90.0) / 100
    prev = st.number_input("Prevalence (%)", min_value=0.0, max_value=100.0, value=10.0) / 100
    ci_half_width = st.number_input("Confidence Interval Half-Width", min_value=0.0, max_value=1.0, value=0.05)
    alpha = st.number_input("Alpha", min_value=0.0, max_value=1.0, value=0.05)
    
    if st.button("Calculate Diagnostic Sample Size"):
        n_pos, n_neg = calculate_sample_size_buderer(se_target, sp_target, prev, ci_half_width, alpha)
        st.info(f"Required Positive Subjects: **{int(n_pos)}**")
        st.info(f"Required Negative Subjects: **{int(n_neg)}**")
    
    st.markdown("---")
    st.subheader("AUC Confidence Interval Half-Width")
    auc_target = st.number_input("Target AUC", min_value=0.5, max_value=1.0, value=0.8)
    ci_half_width_auc = st.number_input("CI Half-Width for AUC", min_value=0.0, max_value=1.0, value=0.1)
    
    if st.button("Calculate AUC Sample Size"):
        n_auc = calculate_auc_n(auc_target, ci_half_width_auc)
        st.info(f"Required total sample size: **{int(n_auc)}**")
    
    st.markdown("---")
    st.subheader("Events Per Variable (EPV)")
    num_vars = st.number_input("Number of Variables in Model", min_value=1, value=10)
    event_rate = st.number_input("Outcome Event Rate (%)", min_value=0.0, max_value=100.0, value=5.0) / 100
    epv_rule = st.number_input("Events Per Variable (EPV) Rule of Thumb", min_value=1, value=10)

    if st.button("Calculate Sample Size by EPV"):
        required_events, total_n = calculate_epv(num_vars, event_rate, epv_rule)
        st.info(f"Required events: **{int(required_events)}**")
        st.info(f"Required total sample size (N): **{int(total_n)}**")
        st.markdown("_A minimum of 10 EPV is often recommended for stable logistic regression models._")

    st.markdown("---")
    st.subheader("Design Effect for Clustering")
    m = st.number_input("Average Cluster Size (m)", min_value=1, value=30)
    icc = st.number_input("Intra-Class Correlation (ICC)", min_value=0.0, max_value=1.0, value=0.01)

    if st.button("Calculate Design Effect"):
        design_effect = calculate_design_effect(m, icc)
        st.info(f"Design Effect (DEFF): **{design_effect:.2f}**")
        st.markdown(f"**Interpretation:** To achieve the same precision as a simple random sample, you'll need to increase your sample size by a factor of **{design_effect:.2f}**.")


elif page == "📈 Analysis":
    st.title("📈 Statistical Analysis")
    st.markdown("Perform statistical analysis on your collected data.")
    
    studies = session.query(Study).all()
    study_titles = {s.id: s.title for s in studies}
    selected_study_id = st.selectbox("Select a Study", options=list(study_titles.keys()), format_func=lambda x: study_titles[x])
    
    if selected_study_id:
        forms = session.query(Form).filter(Form.study_id == selected_study_id).all()
        form_names = {f.id: f.name for f in forms}
        selected_form_id = st.selectbox("Select a Form", options=list(form_names.keys()), format_func=lambda x: form_names[x])
        
        if selected_form_id:
            submissions = session.query(Submission).filter(Submission.form_id == selected_form_id).all()
            if submissions:
                all_data = [json.loads(s.data_json) for s in submissions]
                df = pd.DataFrame(all_data)
                
                # Dynamic column selection
                numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                st.markdown("---")
                st.subheader("ROC Analysis & Youden's Index")
                outcome_col_roc = st.selectbox("Select Binary Outcome Column", categorical_cols, index=categorical_cols.index(BINARY_OUTCOME_COL) if BINARY_OUTCOME_COL in categorical_cols else 0)
                marker_cols_roc = st.multiselect("Select Biomarkers (Numerical)", numerical_cols)
                
                if st.button("Run ROC Analysis"):
                    if marker_cols_roc:
                        for marker in marker_cols_roc:
                            try:
                                result = run_roc_analysis(df, outcome_col_roc, marker)
                                st.subheader(f"Results for `{marker}`")
                                st.write(result['summary'])
                                st.pyplot(result['fig'])
                            except Exception as e:
                                st.error(f"Error with ROC analysis for {marker}: {e}")
                    else:
                        st.warning("Please select at least one marker.")

                st.markdown("---")
                st.subheader("DeLong Test for Comparing Two AUCs")
                marker1 = st.selectbox("Select Marker 1", numerical_cols, key='delong_m1')
                marker2 = st.selectbox("Select Marker 2", numerical_cols, key='delong_m2')
                
                if st.button("Run DeLong Test"):
                    try:
                        delong_result = run_delong_test(df, outcome_col_roc, marker1, marker2)
                        st.write(delong_result)
                    except Exception as e:
                        st.error(f"Error running DeLong test: {e}")
                
                st.markdown("---")
                st.subheader("Logistic Regression")
                outcome_col_log = st.selectbox("Select Binary Outcome Column", categorical_cols, key='log_outcome')
                predictor_cols_log = st.multiselect("Select Predictor Variables", numerical_cols + categorical_cols, key='log_predictors')
                
                if st.button("Run Logistic Regression"):
                    if predictor_cols_log:
                        try:
                            log_result = run_logistic_regression(df, outcome_col_log, predictor_cols_log)
                            st.subheader("Logistic Regression Results")
                            st.write(log_result['summary'])
                            
                            st.subheader("Calibration Curve")
                            fig_cal = plot_calibration_curve(log_result['model'], log_result['X_test'], log_result['y_test'])
                            st.pyplot(fig_cal)
                            
                            st.subheader("Decision Curve Analysis")
                            fig_dca = plot_decision_curve(log_result['y_test'], log_result['y_pred_proba'])
                            st.pyplot(fig_dca)

                        except Exception as e:
                            st.error(f"Error with logistic regression: {e}")
                    else:
                        st.warning("Please select at least one predictor.")
                
                st.markdown("---")
                st.subheader("Linear Regression")
                outcome_col_lin = st.selectbox("Select Continuous Outcome Column", numerical_cols, key='lin_outcome')
                predictor_cols_lin = st.multiselect("Select Predictor Variables", numerical_cols + categorical_cols, key='lin_predictors')
                
                if st.button("Run Linear Regression"):
                    if predictor_cols_lin:
                        try:
                            lin_result = run_linear_regression(df, outcome_col_lin, predictor_cols_lin)
                            st.subheader("Linear Regression Results")
                            st.write(lin_result['summary'])
                        except Exception as e:
                            st.error(f"Error with linear regression: {e}")
                    else:
                        st.warning("Please select at least one predictor.")
                        
                st.markdown("---")
                st.subheader("Survival Analysis (Cox PH & Kaplan-Meier)")
                duration_col = st.selectbox("Select Duration Column", numerical_cols, index=numerical_cols.index(SURVIVAL_COLS['duration']) if SURVIVAL_COLS['duration'] in numerical_cols else 0)
                event_col = st.selectbox("Select Event Column", categorical_cols, index=categorical_cols.index(SURVIVAL_COLS['event']) if SURVIVAL_COLS['event'] in categorical_cols else 0)
                grouping_var = st.selectbox("Select Grouping Variable for KM Plot", categorical_cols)
                
                if st.button("Run Survival Analysis"):
                    try:
                        # Cox PH Model
                        st.subheader("Cox Proportional Hazards Model")
                        cox_result = run_cox_ph(df, duration_col, event_col, predictor_cols_lin)
                        st.write(cox_result)
                        
                        # Kaplan-Meier Plot
                        st.subheader("Kaplan-Meier Plot")
                        km_fig = generate_km_plot(df, duration_col, event_col, grouping_var)
                        st.pyplot(km_fig)

                    except Exception as e:
                        st.error(f"Error with survival analysis: {e}")
            else:
                st.warning("No submissions found for this form. Please enter data first.")
        else:
            st.warning("Please select a form with data to analyze.")
    else:
        st.warning("Please select a study.")

elif page == "📑 Reports & Exports":
    st.title("📑 Reports & Exports")
    st.markdown("Generate comprehensive reports in DOCX or PDF format.")
    
    studies = session.query(Study).all()
    study_titles = {s.id: s.title for s in studies}
    selected_study_id = st.selectbox("Select a Study", options=list(study_titles.keys()), format_func=lambda x: study_titles[x])
    
    if selected_study_id:
        forms = session.query(Form).filter(Form.study_id == selected_study_id).all()
        form_names = {f.id: f.name for f in forms}
        selected_form_id = st.selectbox("Select a Form", options=list(form_names.keys()), format_func=lambda x: form_names[x])

        report_title = st.text_input("Report Title", f"Report for {study_titles[selected_study_id]}")
        report_notes = st.text_area("Report Notes", "Summary of key findings goes here.")
        include_dataset_table = st.checkbox("Include Dataset Table in Report", value=True)
        
        report_data = None
        if include_dataset_table and selected_form_id:
            submissions = session.query(Submission).filter(Submission.form_id == selected_form_id).all()
            if submissions:
                all_data = [json.loads(s.data_json) for s in submissions]
                report_data = pd.DataFrame(all_data)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate DOCX Report"):
                docx_buffer = generate_docx_report(report_title, report_notes, report_data)
                st.download_button(
                    label="Download DOCX",
                    data=docx_buffer.getvalue(),
                    file_name=f"{report_title.replace(' ', '_')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
        
        with col2:
            if st.button("Generate PDF Report"):
                pdf_buffer = generate_pdf_report(report_title, report_notes, report_data)
                st.download_button(
                    label="Download PDF",
                    data=pdf_buffer.getvalue(),
                    file_name=f"{report_title.replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )
    else:
        st.warning("Please create a study first.")

import json
from app.db import Form

# Example template based on the Neonatal Sepsis case
# You can customize these fields
TEMPLATE_FIELDS = [
    {"label": "Patient ID", "name": "patient_id", "type": "text", "required": True},
    {"label": "Enroll Date", "name": "enroll_date", "type": "date", "required": True},
    {"label": "Gestational Age (weeks)", "name": "ga_weeks", "type": "number", "min": 20, "max": 45},
    {"label": "Birth Weight (g)", "name": "bw_g", "type": "number", "min": 500, "max": 6000},
    {"label": "Sex", "name": "sex", "type": "select", "choices": ["M", "F", "Other"]},
    {"label": "EOS or LOS", "name": "sepsis_type", "type": "select", "choices": ["EOS", "LOS"]},
    {"label": "Sepsis", "name": "Sepsis", "type": "select", "choices": ["0", "1"]},
    {"label": "CRP (mg/L)", "name": "crp_mg_l", "type": "number"},
    {"label": "PCT (ng/mL)", "name": "pct_ng_ml", "type": "number"},
    {"label": "IL-6 (pg/mL)", "name": "il6_pg_ml", "type": "number"},
    {"label": "Mortality", "name": "Mortality", "type": "select", "choices": ["0", "1"]},
    {"label": "Time to Death (days)", "name": "time_to_death", "type": "number", "show_if": {"field": "Mortality", "value": "1"}},
    {"label": "Death Event", "name": "death_event", "type": "select", "choices": ["0", "1"], "show_if": {"field": "Mortality", "value": "1"}}
]

def save_form_schema(session, study_id, form_name, schema):
    """Saves a new form schema to the database."""
    form = Form(study_id=study_id, name=form_name, schema_json=json.dumps(schema))
    session.add(form)
    session.commit()
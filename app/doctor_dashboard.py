"""
VerdictMed AI - Doctor Dashboard
Streamlit interface for clinical decision support.
"""

import streamlit as st
import requests
import json
import os
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000"
API_KEY = os.getenv("API_KEY", "dev-test-key")

# Page config
st.set_page_config(
    page_title="VerdictMed AI - CDSS",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E40AF;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        margin-top: 0;
    }
    .diagnosis-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin: 1rem 0;
    }
    .alert-card {
        background: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .confidence-high { color: #059669; font-weight: bold; }
    .confidence-medium { color: #D97706; font-weight: bold; }
    .confidence-low { color: #DC2626; font-weight: bold; }
    .disclaimer {
        background: #FEF3C7;
        border: 1px solid #F59E0B;
        padding: 1rem;
        border-radius: 0.5rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def get_confidence_class(confidence: float) -> str:
    """Get CSS class based on confidence level"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    return "confidence-low"


def call_diagnosis_api(symptoms: str, patient_id: str = None) -> dict:
    """Call the diagnosis API"""
    try:
        response = requests.post(
            f"{API_URL}/api/v1/diagnose",
            json={
                "symptoms": symptoms,
                "patient_id": patient_id
            },
            headers={"X-API-Key": API_KEY},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def render_sidebar():
    """Render sidebar with settings and info"""
    with st.sidebar:
        st.image("https://via.placeholder.com/200x60?text=VerdictMed+AI", width=200)
        st.markdown("---")
        
        st.subheader("⚙️ Settings")
        st.text_input("API Endpoint", value=API_URL, disabled=True)
        st.selectbox("Language", ["English", "Hindi", "Spanish"])
        
        st.markdown("---")
        st.subheader("📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Diagnoses", st.session_state.get("diagnosis_count", 0))
        with col2:
            st.metric("Alerts", st.session_state.get("alert_count", 0))
        
        st.markdown("---")
        st.markdown("""
        <div class='disclaimer'>
        ⚠️ <b>Disclaimer</b>: This is an AI-assisted tool. 
        All outputs require physician verification.
        </div>
        """, unsafe_allow_html=True)


def render_diagnosis_input():
    """Render input form for patient symptoms"""
    st.markdown("<h1 class='main-header'>🏥 Clinical Decision Support</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>AI-Assisted Diagnostic Analysis</p>", unsafe_allow_html=True)
    
    with st.form("diagnosis_form"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            symptoms = st.text_area(
                "Patient Symptoms & Chief Complaint",
                placeholder="e.g., 45-year-old male presenting with fever x3 days, productive cough, SOB. Denies chest pain. Vitals: T 38.5°C, HR 95, BP 130/85, SpO2 94% on RA.",
                height=150
            )
        
        with col2:
            patient_id = st.text_input("Patient ID (optional)")
            st.selectbox("Urgency", ["Routine", "Urgent", "Emergency"])
        
        col3, col4 = st.columns([1, 4])
        with col3:
            submitted = st.form_submit_button("🔍 Analyze", use_container_width=True)
    
    return submitted, symptoms, patient_id


def render_diagnosis_result(result: dict):
    """Render diagnosis results"""
    if "error" in result:
        st.error(f"🚨 API Error: {result['error']}")
        return
    
    # Primary Diagnosis Card
    primary = result.get("primary_diagnosis", {})
    confidence = result.get("confidence", 0)
    conf_class = get_confidence_class(confidence)
    
    st.markdown(f"""
    <div class='diagnosis-card'>
        <h2 style='margin:0'>Primary Diagnosis</h2>
        <h1 style='margin:0.5rem 0'>{primary.get('disease', 'Unknown')}</h1>
        <p style='margin:0'>
            ICD-10: {primary.get('icd10', 'N/A')} | 
            Severity: {primary.get('severity', 'Unknown').title()} |
            Confidence: <span class='{conf_class}'>{confidence:.1%}</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Safety Alerts
    alerts = result.get("safety_alerts", [])
    requires_review = result.get("requires_review", False)
    
    if requires_review or alerts:
        st.subheader("⚠️ Safety Alerts")
        if requires_review:
            st.markdown("""
            <div class='alert-card'>
                🚨 <b>PHYSICIAN REVIEW REQUIRED</b>: This case has been flagged for mandatory review.
            </div>
            """, unsafe_allow_html=True)
        
        for alert in alerts:
            st.warning(f"⚠️ {alert}")
    
    # Differential Diagnoses
    differentials = result.get("differential_diagnoses", [])
    if differentials:
        st.subheader("📋 Differential Diagnoses")
        for i, dx in enumerate(differentials[:5], 1):
            conf = dx.get("confidence", 0)
            with st.expander(f"{i}. {dx.get('disease', 'Unknown')} ({conf:.1%})"):
                st.write(f"**ICD-10:** {dx.get('icd10', 'N/A')}")
                st.write(f"**Severity:** {dx.get('severity', 'Unknown')}")
                st.progress(conf)
    
    # Explanation
    explanation = result.get("explanation", "")
    if explanation:
        st.subheader("💡 AI Reasoning")
        st.text(explanation)
    
    # Metadata
    with st.expander("📊 Request Details"):
        st.json({
            "request_id": result.get("request_id"),
            "processing_time_ms": result.get("processing_time_ms"),
            "timestamp": datetime.now().isoformat()
        })


def render_feedback_form(request_id: str):
    """Render feedback form for physician input"""
    st.subheader("📝 Physician Feedback")
    
    with st.form("feedback_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            accuracy = st.radio(
                "Was the diagnosis accurate?",
                ["Correct", "Partially Correct", "Incorrect"],
                horizontal=True
            )
        
        with col2:
            final_diagnosis = st.text_input("Final Diagnosis (if different)")
        
        notes = st.text_area("Additional Notes", placeholder="Any observations for model improvement...")
        
        if st.form_submit_button("Submit Feedback"):
            st.success("✅ Feedback recorded. Thank you!")
            # TODO: Store feedback in database


def main():
    """Main app entry point"""
    # Initialize session state
    if "diagnosis_count" not in st.session_state:
        st.session_state.diagnosis_count = 0
    if "alert_count" not in st.session_state:
        st.session_state.alert_count = 0
    
    render_sidebar()
    
    # Main content
    tabs = st.tabs(["🔍 New Diagnosis", "📊 History", "📈 Analytics"])
    
    with tabs[0]:
        submitted, symptoms, patient_id = render_diagnosis_input()
        
        if submitted and symptoms:
            with st.spinner("Analyzing patient data..."):
                result = call_diagnosis_api(symptoms, patient_id)
                
                if "error" not in result:
                    st.session_state.diagnosis_count += 1
                    if result.get("requires_review"):
                        st.session_state.alert_count += 1
                
                render_diagnosis_result(result)
                
                if "error" not in result:
                    render_feedback_form(result.get("request_id", "unknown"))
    
    with tabs[1]:
        st.info("📋 Diagnosis history will be displayed here once database integration is complete.")
    
    with tabs[2]:
        st.info("📊 Analytics dashboard coming soon.")


if __name__ == "__main__":
    main()

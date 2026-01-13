import { useState, useEffect } from 'react'
import Header from './components/Header'
import PatientSummary from './components/PatientSummary'
import InputsPanel from './components/InputsPanel'
import AnalysisProgress from './components/AnalysisProgress'
import DiagnosisCard from './components/DiagnosisCard'
import Recommendations from './components/Recommendations'
import Explainability from './components/Explainability'
import SafetyAlerts from './components/SafetyAlerts'
import { diagnose } from './api/diagnosis'
import './index.css'

function App() {
  // Patient data state
  const [patientData, setPatientData] = useState({
    patientId: '',
    age: '',
    sex: '',
    chiefComplaint: '',
    vitals: {}
  })

  // Clinical inputs state
  const [clinicalInputs, setClinicalInputs] = useState({
    image: null,
    hpi: '',
    assessment: '',
    labs: [],
    comorbidities: '',
    medications: '',
    allergies: ''
  })

  // Analysis state
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [analysisStep, setAnalysisStep] = useState(0)
  const [error, setError] = useState(null)

  // Simulate step progress during loading
  useEffect(() => {
    if (loading) {
      setAnalysisStep(1)
      const intervals = [
        setTimeout(() => setAnalysisStep(2), 1500),
        setTimeout(() => setAnalysisStep(3), 3000),
        setTimeout(() => setAnalysisStep(4), 4500)
      ]
      return () => intervals.forEach(clearTimeout)
    } else {
      setAnalysisStep(0)
    }
  }, [loading])

  const handleAnalyze = async () => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      // Combine all inputs for API
      const symptoms = [
        patientData.chiefComplaint,
        clinicalInputs.hpi,
        clinicalInputs.assessment
      ].filter(Boolean).join('. ')

      const response = await diagnose({
        patient_id: patientData.patientId || 'P' + Date.now(),
        symptoms,
        age: parseInt(patientData.age) || undefined,
        sex: patientData.sex || undefined
      })

      // Parse and enhance the response
      setResult({
        ...response,
        recommendations: generateRecommendations(response),
        evidence: parseEvidence(response),
        reasoningChain: generateReasoningChain(response)
      })
    } catch (err) {
      setError(err.message || 'Analysis failed. Please check if the API is running.')
    } finally {
      setLoading(false)
    }
  }

  // Generate recommendations based on diagnosis
  const generateRecommendations = (response) => {
    const recs = []
    const diagnosis = response?.primary_diagnosis?.disease?.toLowerCase() || ''

    if (response?.requires_review) {
      recs.push('Urgent: Schedule physician review for this case')
    }
    if (diagnosis.includes('pneumonia')) {
      recs.push('Consider chest CT if hypoxia persists')
      recs.push('Start empiric antibiotics per hospital guidelines')
      recs.push('Order sputum culture before antibiotic initiation')
    }
    if (response?.confidence < 0.6) {
      recs.push('Consider additional diagnostic workup due to low confidence')
    }
    if (recs.length === 0) {
      recs.push('Continue monitoring and reassess as needed')
    }
    return recs
  }

  // Parse evidence from response
  const parseEvidence = (response) => {
    const findings = response?.extracted_findings || []
    return {
      text: findings.length > 0 ? findings : ['No entities extracted'],
      labs: clinicalInputs.labs?.length > 0
        ? clinicalInputs.labs.map(l => `${l.test}: ${l.value} ${l.flag === 'high' ? '↑' : l.flag === 'low' ? '↓' : ''}`).filter(Boolean)
        : null,
      kg: response?.primary_diagnosis?.icd10
        ? [`Matched pattern in knowledge graph: ${response.primary_diagnosis.disease}`]
        : null
    }
  }

  // Generate reasoning chain from response
  const generateReasoningChain = (response) => {
    const chain = []
    const findings = response?.extracted_findings || []

    if (findings.length > 0) {
      chain.push(`Extracted key clinical entities: ${findings.join(', ')}`)
    }
    if (response?.primary_diagnosis?.disease) {
      chain.push(`Matched symptom pattern to ${response.primary_diagnosis.disease} in medical knowledge base`)
    }
    if (response?.confidence) {
      chain.push(`Calculated confidence score of ${(response.confidence * 100).toFixed(0)}% based on symptom-disease correlation`)
    }
    if (response?.requires_review) {
      chain.push('Flagged for physician review due to confidence below threshold or critical condition indicators')
    }
    return chain
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-indigo-50">
      <Header />

      <main className="max-w-7xl mx-auto px-4 py-6">
        {/* 3-Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

          {/* Left: Patient Summary (3 cols) */}
          <div className="lg:col-span-3">
            <PatientSummary
              data={patientData}
              onChange={setPatientData}
            />
          </div>

          {/* Center: Inputs Panel (4 cols) */}
          <div className="lg:col-span-4">
            <InputsPanel
              data={clinicalInputs}
              onChange={setClinicalInputs}
              onAnalyze={handleAnalyze}
              loading={loading}
            />
          </div>

          {/* Right: AI Output (5 cols) */}
          <div className="lg:col-span-5 space-y-4">
            {/* Loading State */}
            {loading && (
              <AnalysisProgress currentStep={analysisStep} />
            )}

            {/* Error State */}
            {error && (
              <div className="bg-red-50 border-2 border-red-200 rounded-2xl p-6">
                <div className="flex items-center gap-3 text-red-700">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="font-semibold">Analysis Error</span>
                </div>
                <p className="mt-2 text-red-600">{error}</p>
              </div>
            )}

            {/* Results */}
            {result && !loading && (
              <>
                {/* Safety Alerts */}
                <SafetyAlerts
                  alerts={result.safety_alerts || []}
                  requiresReview={result.requires_review}
                />

                {/* Diagnosis Card */}
                <DiagnosisCard result={result} />

                {/* Recommendations */}
                <Recommendations recommendations={result.recommendations} />

                {/* Explainability */}
                <Explainability
                  evidence={result.evidence}
                  reasoningChain={result.reasoningChain}
                />
              </>
            )}

            {/* Empty State */}
            {!result && !loading && !error && (
              <div className="bg-white/50 rounded-2xl border-2 border-dashed border-gray-300 p-8 flex flex-col items-center justify-center min-h-[500px]">
                <svg className="w-20 h-20 text-gray-300 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
                <p className="text-gray-400 text-lg font-medium">AI Diagnosis Output</p>
                <p className="text-gray-300 text-sm mt-1">Enter patient data and run analysis</p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-8 py-6 border-t border-gray-200">
        <div className="max-w-4xl mx-auto px-4">
          <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-center">
            <p className="text-amber-800 text-sm">
              <span className="font-semibold">Important Notice:</span> This tool provides AI-assisted clinical decision support only.
              All diagnostic suggestions must be reviewed and verified by a qualified healthcare professional.
            </p>
          </div>
          <p className="text-center text-gray-400 text-xs mt-4">
            VerdictMed AI • Clinical Decision Support System • For Professional Use Only
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App

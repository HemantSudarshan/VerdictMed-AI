import ConfidenceGauge from './ConfidenceGauge'
import SafetyAlerts from './SafetyAlerts'

export default function DiagnosisResult({ result }) {
    const primary = result.primary_diagnosis || {}
    const confidence = result.confidence || 0
    const differentials = result.differential_diagnoses || []
    const alerts = result.safety_alerts || []
    const requiresReview = result.requires_review || false
    const explanation = result.explanation || ''

    const getSeverityColor = (severity) => {
        switch (severity?.toLowerCase()) {
            case 'critical': return 'text-red-600 bg-red-100'
            case 'high': return 'text-orange-600 bg-orange-100'
            case 'moderate': return 'text-yellow-600 bg-yellow-100'
            default: return 'text-green-600 bg-green-100'
        }
    }

    return (
        <div className="space-y-4">
            {/* Safety Alerts */}
            {(alerts.length > 0 || requiresReview) && (
                <SafetyAlerts alerts={alerts} requiresReview={requiresReview} />
            )}

            {/* Primary Diagnosis Card */}
            <div className="diagnosis-card bg-gradient-to-br from-indigo-600 to-purple-700 rounded-2xl p-6 text-white">
                <div className="flex items-start justify-between">
                    <div className="flex-1">
                        <p className="text-indigo-200 text-sm font-medium mb-1">Primary Diagnosis</p>
                        <h2 className="text-2xl font-bold mb-2">{primary.disease || 'Unknown'}</h2>
                        <div className="flex flex-wrap gap-2">
                            {primary.icd10 && (
                                <span className="bg-white/20 px-2 py-1 rounded text-xs font-medium">
                                    ICD-10: {primary.icd10}
                                </span>
                            )}
                            {primary.severity && (
                                <span className={`px-2 py-1 rounded text-xs font-medium ${getSeverityColor(primary.severity)}`}>
                                    {primary.severity}
                                </span>
                            )}
                        </div>
                    </div>
                    <ConfidenceGauge confidence={confidence} />
                </div>
            </div>

            {/* Differential Diagnoses */}
            {differentials.length > 0 && (
                <div className="bg-white rounded-2xl shadow-lg p-6">
                    <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                        <svg className="w-5 h-5 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                        </svg>
                        Differential Diagnoses
                    </h3>
                    <div className="space-y-3">
                        {differentials.slice(0, 5).map((dx, idx) => (
                            <div key={idx} className="flex items-center gap-4 p-3 bg-gray-50 rounded-xl hover:bg-gray-100 transition">
                                <span className="text-lg font-bold text-gray-400 w-6">{idx + 1}</span>
                                <div className="flex-1">
                                    <p className="font-medium text-gray-800">{dx.disease}</p>
                                    <p className="text-xs text-gray-500">
                                        {dx.icd10 && `ICD-10: ${dx.icd10}`}
                                        {dx.severity && ` • ${dx.severity}`}
                                    </p>
                                </div>
                                <div className="text-right">
                                    <span className="text-sm font-semibold text-indigo-600">
                                        {(dx.confidence * 100).toFixed(0)}%
                                    </span>
                                    <div className="w-20 h-2 bg-gray-200 rounded-full overflow-hidden mt-1">
                                        <div
                                            className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 transition-all duration-500"
                                            style={{ width: `${dx.confidence * 100}%` }}
                                        />
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* AI Reasoning */}
            {explanation && (
                <div className="bg-white rounded-2xl shadow-lg p-6">
                    <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                        <svg className="w-5 h-5 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                        </svg>
                        Analysis Summary
                    </h3>
                    <div className="space-y-3">
                        {/* Parse and display explanation in a friendly format */}
                        {explanation.split('\n').map((line, idx) => {
                            const cleanLine = line.trim()
                            if (!cleanLine) return null

                            // Format key findings
                            if (cleanLine.startsWith('KEY FINDINGS:') || cleanLine.startsWith('FINDINGS:')) {
                                return (
                                    <div key={idx} className="bg-blue-50 rounded-lg p-3">
                                        <p className="text-sm font-medium text-blue-800 mb-1">Key Findings Identified:</p>
                                        <p className="text-sm text-blue-700">
                                            {cleanLine.replace(/KEY FINDINGS:|FINDINGS:/g, '').trim() || 'No specific findings extracted'}
                                        </p>
                                    </div>
                                )
                            }

                            // Format diagnosis line
                            if (cleanLine.startsWith('DIAGNOSIS:')) {
                                return (
                                    <div key={idx} className="flex items-center gap-2">
                                        <span className="text-gray-500 text-sm">Suggested Diagnosis:</span>
                                        <span className="font-medium text-gray-800">
                                            {cleanLine.replace('DIAGNOSIS:', '').trim()}
                                        </span>
                                    </div>
                                )
                            }

                            // Format confidence line
                            if (cleanLine.startsWith('CONFIDENCE:')) {
                                const confValue = cleanLine.match(/(\d+\.?\d*)%/)
                                return (
                                    <div key={idx} className="flex items-center gap-2">
                                        <span className="text-gray-500 text-sm">Confidence Level:</span>
                                        <span className={`font-medium ${parseFloat(confValue?.[1]) >= 70 ? 'text-green-600' : parseFloat(confValue?.[1]) >= 55 ? 'text-yellow-600' : 'text-red-600'}`}>
                                            {confValue?.[0] || cleanLine.replace('CONFIDENCE:', '').trim()}
                                        </span>
                                    </div>
                                )
                            }

                            // Format alerts line more nicely
                            if (cleanLine.startsWith('ALERTS:')) {
                                const alertText = cleanLine.replace('ALERTS:', '').trim()
                                if (alertText.includes('LOW_CONFIDENCE')) {
                                    return (
                                        <div key={idx} className="flex items-center gap-2 text-amber-700 bg-amber-50 p-2 rounded-lg">
                                            <span className="text-sm">⚠️ The confidence level is below optimal. Physician verification recommended.</span>
                                        </div>
                                    )
                                }
                                return null // Hide raw alert text, already shown in SafetyAlerts
                            }

                            // Format bullet points
                            if (cleanLine.startsWith('•') || cleanLine.startsWith('-')) {
                                return (
                                    <div key={idx} className="flex items-start gap-2 ml-4">
                                        <span className="text-indigo-500">•</span>
                                        <span className="text-sm text-gray-700">{cleanLine.replace(/^[•-]\s*/, '')}</span>
                                    </div>
                                )
                            }

                            // Default: just show the line
                            return (
                                <p key={idx} className="text-sm text-gray-600">{cleanLine}</p>
                            )
                        })}
                    </div>
                </div>
            )}

            {/* Request Details */}
            <details className="bg-gray-50 rounded-xl p-4">
                <summary className="text-sm text-gray-500 cursor-pointer hover:text-gray-700">
                    Request Details
                </summary>
                <div className="mt-3 text-xs font-mono text-gray-600">
                    <p>Request ID: {result.request_id}</p>
                    <p>Processing Time: {result.processing_time_ms}ms</p>
                </div>
            </details>
        </div>
    )
}

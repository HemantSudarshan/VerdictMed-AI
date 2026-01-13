import ConfidenceGauge from './ConfidenceGauge'

export default function DiagnosisCard({ result }) {
    const primary = result?.primary_diagnosis || {}
    const confidence = result?.confidence || 0
    const differentials = result?.differential_diagnoses || []

    const getSeverityStyle = (severity) => {
        switch (severity?.toLowerCase()) {
            case 'critical': return 'bg-red-500 text-white'
            case 'high': return 'bg-orange-500 text-white'
            case 'moderate': return 'bg-yellow-500 text-white'
            default: return 'bg-green-500 text-white'
        }
    }

    const getUrgencyLabel = (confidence, severity) => {
        if (severity === 'critical') return { text: 'URGENT', color: 'bg-red-600' }
        if (confidence < 0.5) return { text: 'UNCERTAIN', color: 'bg-amber-500' }
        if (confidence > 0.8) return { text: 'HIGH CONFIDENCE', color: 'bg-green-600' }
        return null
    }

    const urgency = getUrgencyLabel(confidence, primary.severity)

    return (
        <div className="space-y-4">
            {/* Primary Diagnosis */}
            <div className="bg-gradient-to-br from-indigo-600 to-purple-700 rounded-2xl p-6 text-white shadow-xl">
                <div className="flex items-start justify-between">
                    <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                            <p className="text-indigo-200 text-sm font-medium">Most Likely Diagnosis</p>
                            {urgency && (
                                <span className={`${urgency.color} px-2 py-0.5 rounded text-xs font-bold`}>
                                    {urgency.text}
                                </span>
                            )}
                        </div>
                        <h2 className="text-2xl font-bold mb-3">
                            {primary.disease || 'Awaiting Analysis'}
                        </h2>
                        <div className="flex flex-wrap gap-2">
                            {primary.icd10 && (
                                <span className="bg-white/20 px-3 py-1 rounded-full text-xs font-medium">
                                    ICD-10: {primary.icd10}
                                </span>
                            )}
                            {primary.severity && (
                                <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getSeverityStyle(primary.severity)}`}>
                                    {primary.severity.toUpperCase()}
                                </span>
                            )}
                        </div>
                    </div>
                    <ConfidenceGauge confidence={confidence} />
                </div>
            </div>

            {/* Differential Diagnoses */}
            {differentials.length > 0 && (
                <div className="bg-white rounded-2xl shadow-lg p-5">
                    <h3 className="text-sm font-bold text-gray-700 mb-3 flex items-center gap-2">
                        <svg className="w-4 h-4 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 9l4-4 4 4m0 6l-4 4-4-4" />
                        </svg>
                        Differential Diagnoses
                    </h3>
                    <div className="space-y-2">
                        {differentials.slice(0, 3).map((dx, idx) => (
                            <div
                                key={idx}
                                className="flex items-center justify-between p-3 bg-gray-50 rounded-xl hover:bg-gray-100 transition"
                            >
                                <div className="flex items-center gap-3">
                                    <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${idx === 0 ? 'bg-indigo-100 text-indigo-600' : 'bg-gray-200 text-gray-500'
                                        }`}>
                                        {idx + 1}
                                    </span>
                                    <div>
                                        <p className="font-medium text-gray-800 text-sm">{dx.disease}</p>
                                        <p className="text-xs text-gray-500">
                                            {dx.icd10 && `${dx.icd10}`}
                                            {dx.severity && ` • ${dx.severity}`}
                                        </p>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <span className={`text-sm font-bold ${dx.confidence > 0.5 ? 'text-indigo-600' : 'text-gray-500'
                                        }`}>
                                        {(dx.confidence * 100).toFixed(0)}%
                                    </span>
                                    <div className="w-16 h-1.5 bg-gray-200 rounded-full mt-1">
                                        <div
                                            className="h-full bg-gradient-to-r from-indigo-400 to-purple-500 rounded-full"
                                            style={{ width: `${dx.confidence * 100}%` }}
                                        />
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}

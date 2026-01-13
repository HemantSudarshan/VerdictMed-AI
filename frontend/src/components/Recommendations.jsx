export default function Recommendations({ recommendations }) {
    if (!recommendations || recommendations.length === 0) return null

    const getSeverityStyle = (text) => {
        const lower = text.toLowerCase()
        if (lower.includes('urgent') || lower.includes('immediate') || lower.includes('critical')) {
            return 'bg-red-50 border-l-4 border-red-500 text-red-800'
        }
        if (lower.includes('consider') || lower.includes('may') || lower.includes('optional')) {
            return 'bg-blue-50 border-l-4 border-blue-500 text-blue-800'
        }
        return 'bg-green-50 border-l-4 border-green-500 text-green-800'
    }

    return (
        <div className="bg-white rounded-2xl shadow-lg p-6">
            <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Clinical Recommendations
            </h3>

            <div className="space-y-2">
                {recommendations.map((rec, idx) => (
                    <div key={idx} className={`p-3 rounded-r-lg text-sm ${getSeverityStyle(rec)}`}>
                        <div className="flex items-start gap-2">
                            <span className="mt-0.5">•</span>
                            <span>{rec}</span>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    )
}

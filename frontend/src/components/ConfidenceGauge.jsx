export default function ConfidenceGauge({ confidence }) {
    const percentage = Math.round(confidence * 100)

    const getColor = () => {
        if (percentage >= 80) return { stroke: '#059669', text: 'text-green-600' }
        if (percentage >= 60) return { stroke: '#d97706', text: 'text-yellow-600' }
        return { stroke: '#dc2626', text: 'text-red-600' }
    }

    const { stroke, text } = getColor()
    const circumference = 2 * Math.PI * 45
    const offset = circumference - (percentage / 100) * circumference

    return (
        <div className="confidence-gauge relative">
            <svg className="w-24 h-24" viewBox="0 0 100 100">
                {/* Background circle */}
                <circle
                    className="text-white/20"
                    strokeWidth="8"
                    stroke="currentColor"
                    fill="transparent"
                    r="45"
                    cx="50"
                    cy="50"
                />
                {/* Progress circle */}
                <circle
                    className="gauge-circle transition-all duration-1000 ease-out"
                    strokeWidth="8"
                    stroke={stroke}
                    fill="transparent"
                    r="45"
                    cx="50"
                    cy="50"
                    style={{
                        strokeDasharray: circumference,
                        strokeDashoffset: offset,
                        transform: 'rotate(-90deg)',
                        transformOrigin: 'center'
                    }}
                    strokeLinecap="round"
                />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className={`text-xl font-bold ${percentage >= 60 ? 'text-white' : 'text-white'}`}>
                    {percentage}%
                </span>
                <span className="text-xs text-white/70">Confidence</span>
            </div>
        </div>
    )
}

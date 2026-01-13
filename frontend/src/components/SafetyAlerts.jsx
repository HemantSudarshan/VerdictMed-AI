/**
 * Format raw alert messages into human-readable text
 */
function formatAlert(alert) {
    // LOW_CONFIDENCE pattern
    if (alert.includes('LOW_CONFIDENCE')) {
        const match = alert.match(/(\d+\.?\d*)%.*?(\d+\.?\d*)%/)
        if (match) {
            return `The AI confidence level (${match[1]}%) is below the recommended threshold. A physician should review this case.`
        }
        return 'Confidence level is below recommended threshold. Physician review recommended.'
    }

    // CRITICAL_CONDITION pattern
    if (alert.includes('CRITICAL_CONDITION') || alert.includes('critical')) {
        return '🚨 Potential critical condition detected. Immediate physician attention recommended.'
    }

    // SIGNAL_CONFLICT pattern
    if (alert.includes('SIGNAL_CONFLICT')) {
        return 'Conflicting signals detected between different analysis methods. Manual review needed.'
    }

    // ESCALATION pattern
    if (alert.includes('ESCALATION') || alert.includes('escalation')) {
        return 'This case has been escalated for senior physician review.'
    }

    // Return as-is if no pattern matches, but clean up underscores
    return alert.replace(/_/g, ' ').replace(/\s+/g, ' ').trim()
}

export default function SafetyAlerts({ alerts, requiresReview }) {
    if (!alerts.length && !requiresReview) return null

    return (
        <div className="space-y-3">
            {requiresReview && (
                <div className="bg-gradient-to-r from-red-50 to-orange-50 border-l-4 border-red-500 rounded-r-xl p-4 shadow-sm">
                    <div className="flex items-start gap-3">
                        <div className="flex-shrink-0 bg-red-100 p-2 rounded-full">
                            <svg className="w-5 h-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                            </svg>
                        </div>
                        <div>
                            <h4 className="text-red-800 font-semibold text-base">Physician Review Required</h4>
                            <p className="text-red-600 text-sm mt-1">
                                This case requires verification by a qualified physician before any clinical decisions are made.
                            </p>
                        </div>
                    </div>
                </div>
            )}

            {alerts.map((alert, idx) => (
                <div key={idx} className="bg-amber-50 border-l-4 border-amber-400 rounded-r-xl px-4 py-3 shadow-sm">
                    <div className="flex items-start gap-3">
                        <div className="flex-shrink-0 mt-0.5">
                            <svg className="w-5 h-5 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                        </div>
                        <p className="text-amber-800 text-sm leading-relaxed">
                            {formatAlert(alert)}
                        </p>
                    </div>
                </div>
            ))}
        </div>
    )
}

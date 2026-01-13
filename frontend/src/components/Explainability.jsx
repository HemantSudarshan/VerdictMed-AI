import { useState } from 'react'

export default function Explainability({ evidence, reasoningChain }) {
    const [isOpen, setIsOpen] = useState(false)

    const evidenceSources = [
        { key: 'image', label: 'From Imaging', icon: '🩻', color: 'indigo' },
        { key: 'text', label: 'From Clinical Text', icon: '📝', color: 'blue' },
        { key: 'labs', label: 'From Lab Results', icon: '🧪', color: 'green' },
        { key: 'kg', label: 'From Knowledge Graph', icon: '🔗', color: 'purple' }
    ]

    return (
        <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
            {/* Collapsible Header */}
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full p-4 flex items-center justify-between hover:bg-gray-50 transition"
            >
                <h3 className="text-lg font-bold text-gray-800 flex items-center gap-2">
                    <svg className="w-5 h-5 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Why This Diagnosis?
                </h3>
                <svg
                    className={`w-5 h-5 text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                </svg>
            </button>

            {/* Collapsible Content */}
            {isOpen && (
                <div className="p-4 pt-0 border-t border-gray-100">
                    {/* Evidence Sources */}
                    <div className="space-y-3 mb-6">
                        <p className="text-sm font-medium text-gray-600 mb-2">Key Supporting Evidence:</p>

                        {evidenceSources.map(source => {
                            const data = evidence?.[source.key]
                            if (!data) return null

                            return (
                                <div key={source.key} className={`bg-${source.color}-50 rounded-xl p-3`}>
                                    <p className={`text-xs font-semibold text-${source.color}-700 mb-1`}>
                                        {source.icon} {source.label}
                                    </p>
                                    <p className={`text-sm text-${source.color}-800`}>
                                        {Array.isArray(data) ? data.join(', ') : data}
                                    </p>
                                </div>
                            )
                        })}

                        {!evidence && (
                            <p className="text-sm text-gray-500 italic">No detailed evidence available for this analysis.</p>
                        )}
                    </div>

                    {/* Reasoning Chain */}
                    {reasoningChain && reasoningChain.length > 0 && (
                        <div>
                            <p className="text-sm font-medium text-gray-600 mb-3">Step-by-Step Reasoning:</p>
                            <div className="relative pl-6 border-l-2 border-gray-200 space-y-4">
                                {reasoningChain.map((step, idx) => (
                                    <div key={idx} className="relative">
                                        <div className="absolute -left-[25px] w-4 h-4 rounded-full bg-indigo-500 flex items-center justify-center">
                                            <span className="text-[10px] text-white font-bold">{idx + 1}</span>
                                        </div>
                                        <p className="text-sm text-gray-700 bg-gray-50 rounded-lg p-3">
                                            {step}
                                        </p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}

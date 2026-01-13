export default function AnalysisProgress({ currentStep }) {
    const steps = [
        { id: 1, label: 'Analyzing imaging data', icon: '🩻' },
        { id: 2, label: 'Extracting clinical entities', icon: '📝' },
        { id: 3, label: 'Querying knowledge graph', icon: '🔗' },
        { id: 4, label: 'Generating diagnosis', icon: '🔬' }
    ]

    return (
        <div className="bg-white rounded-2xl shadow-lg p-6">
            <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                <div className="spinner w-5 h-5"></div>
                Analysis in Progress
            </h3>

            <div className="space-y-3">
                {steps.map((step) => {
                    const isActive = step.id === currentStep
                    const isComplete = step.id < currentStep

                    return (
                        <div
                            key={step.id}
                            className={`flex items-center gap-3 p-3 rounded-xl transition-all ${isActive
                                    ? 'bg-indigo-50 border-2 border-indigo-200'
                                    : isComplete
                                        ? 'bg-green-50'
                                        : 'bg-gray-50'
                                }`}
                        >
                            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm ${isActive
                                    ? 'bg-indigo-500 text-white animate-pulse'
                                    : isComplete
                                        ? 'bg-green-500 text-white'
                                        : 'bg-gray-200 text-gray-500'
                                }`}>
                                {isComplete ? '✓' : step.id}
                            </div>
                            <div className="flex-1">
                                <p className={`text-sm font-medium ${isActive ? 'text-indigo-700' : isComplete ? 'text-green-700' : 'text-gray-500'
                                    }`}>
                                    {step.icon} {step.label}
                                </p>
                            </div>
                            {isActive && (
                                <div className="flex gap-1">
                                    <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-bounce"></span>
                                    <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></span>
                                    <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></span>
                                </div>
                            )}
                        </div>
                    )
                })}
            </div>
        </div>
    )
}

import { useState } from 'react'

export default function DiagnosisForm({ onSubmit, loading }) {
    const [symptoms, setSymptoms] = useState('')
    const [patientId, setPatientId] = useState('')
    const [age, setAge] = useState('')
    const [sex, setSex] = useState('')

    const handleSubmit = (e) => {
        e.preventDefault()
        if (symptoms.trim()) {
            onSubmit({ symptoms, patientId, age, sex })
        }
    }

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            handleSubmit(e)
        }
    }

    return (
        <div className="bg-white rounded-2xl shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Patient Assessment
            </h2>

            <form onSubmit={handleSubmit}>
                {/* Patient Info Row */}
                <div className="grid grid-cols-3 gap-3 mb-4">
                    <div>
                        <label className="block text-xs font-medium text-gray-500 mb-1">Patient ID</label>
                        <input
                            type="text"
                            value={patientId}
                            onChange={(e) => setPatientId(e.target.value)}
                            className="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition"
                            placeholder="P12345"
                        />
                    </div>
                    <div>
                        <label className="block text-xs font-medium text-gray-500 mb-1">Age</label>
                        <input
                            type="number"
                            value={age}
                            onChange={(e) => setAge(e.target.value)}
                            className="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition"
                            placeholder="45"
                            min="0"
                            max="120"
                        />
                    </div>
                    <div>
                        <label className="block text-xs font-medium text-gray-500 mb-1">Sex</label>
                        <select
                            value={sex}
                            onChange={(e) => setSex(e.target.value)}
                            className="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition bg-white"
                        >
                            <option value="">Select...</option>
                            <option value="M">Male</option>
                            <option value="F">Female</option>
                        </select>
                    </div>
                </div>

                {/* Symptoms Text Area */}
                <div className="mb-4">
                    <label className="block text-xs font-medium text-gray-500 mb-1">
                        Chief Complaint & Symptoms
                    </label>
                    <textarea
                        value={symptoms}
                        onChange={(e) => setSymptoms(e.target.value)}
                        onKeyDown={handleKeyDown}
                        className="w-full px-4 py-3 border border-gray-200 rounded-xl text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition resize-none"
                        rows="6"
                        placeholder="e.g., 45-year-old male presenting with fever x3 days, productive cough with yellow sputum, shortness of breath. Vitals: T 38.5°C, HR 95, BP 130/85, SpO2 94% on RA. Denies chest pain. History of HTN and DM."
                    />
                    <p className="text-xs text-gray-400 mt-1">Press Ctrl+Enter to submit</p>
                </div>

                {/* Submit Button */}
                <button
                    type="submit"
                    disabled={loading || !symptoms.trim()}
                    className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-medium py-3 px-6 rounded-xl hover:from-indigo-700 hover:to-purple-700 focus:ring-4 focus:ring-indigo-200 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2"
                >
                    {loading ? (
                        <>
                            <div className="spinner"></div>
                            Analyzing...
                        </>
                    ) : (
                        <>
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                            </svg>
                            Analyze Symptoms
                        </>
                    )}
                </button>
            </form>
        </div>
    )
}

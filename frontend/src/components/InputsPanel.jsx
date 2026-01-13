import { useState } from 'react'

export default function InputsPanel({ data, onChange, onAnalyze, loading }) {
    const [activeTab, setActiveTab] = useState('imaging')

    const tabs = [
        { id: 'imaging', label: 'Imaging', icon: '🩻' },
        { id: 'clinical', label: 'Clinical Text', icon: '📝' },
        { id: 'labs', label: 'Labs', icon: '🧪' },
        { id: 'context', label: 'Context', icon: '📋' }
    ]

    const handleChange = (field, value) => {
        onChange?.({ ...data, [field]: value })
    }

    const handleLabChange = (index, field, value) => {
        const labs = [...(data?.labs || [])]
        labs[index] = { ...labs[index], [field]: value }
        onChange?.({ ...data, labs })
    }

    const addLab = () => {
        const labs = [...(data?.labs || []), { test: '', value: '', range: '', flag: '' }]
        onChange?.({ ...data, labs })
    }

    return (
        <div className="bg-white rounded-2xl shadow-lg p-5 h-full flex flex-col">
            <h2 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Clinical Inputs
            </h2>

            {/* Tabs */}
            <div className="flex gap-1 mb-4 bg-gray-100 rounded-xl p-1">
                {tabs.map(tab => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`flex-1 py-2 px-2 rounded-lg text-xs font-medium transition-all ${activeTab === tab.id
                                ? 'bg-white text-indigo-600 shadow-sm'
                                : 'text-gray-500 hover:text-gray-700'
                            }`}
                    >
                        <span className="mr-1">{tab.icon}</span>
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Tab Content */}
            <div className="flex-1 overflow-y-auto">
                {/* Imaging Tab */}
                {activeTab === 'imaging' && (
                    <div className="space-y-4">
                        <div className="border-2 border-dashed border-gray-200 rounded-xl p-6 text-center hover:border-indigo-300 transition cursor-pointer">
                            <svg className="w-12 h-12 text-gray-300 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                            </svg>
                            <p className="text-sm text-gray-500 mb-2">Drop chest X-ray or CT image</p>
                            <input
                                type="file"
                                accept="image/*"
                                onChange={(e) => handleChange('image', e.target.files[0])}
                                className="hidden"
                                id="image-upload"
                            />
                            <label
                                htmlFor="image-upload"
                                className="inline-block px-4 py-2 bg-indigo-100 text-indigo-600 rounded-lg text-sm font-medium cursor-pointer hover:bg-indigo-200"
                            >
                                Browse Files
                            </label>
                        </div>
                        {data?.image && (
                            <div className="bg-gray-50 rounded-xl p-3">
                                <p className="text-xs text-gray-500 mb-2">Selected: {data.image.name}</p>
                                <button className="text-xs text-indigo-600 hover:underline">View Full Image</button>
                            </div>
                        )}
                    </div>
                )}

                {/* Clinical Text Tab */}
                {activeTab === 'clinical' && (
                    <div className="space-y-4">
                        <div>
                            <label className="block text-xs font-medium text-gray-500 mb-1">
                                History of Present Illness (HPI)
                            </label>
                            <textarea
                                value={data?.hpi || ''}
                                onChange={(e) => handleChange('hpi', e.target.value)}
                                placeholder="Detailed symptom description, onset, duration, severity..."
                                rows={4}
                                className="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 resize-none"
                            />
                        </div>
                        <div>
                            <label className="block text-xs font-medium text-gray-500 mb-1">
                                Clinical Assessment Notes
                            </label>
                            <textarea
                                value={data?.assessment || ''}
                                onChange={(e) => handleChange('assessment', e.target.value)}
                                placeholder="Physical exam findings, clinical impressions..."
                                rows={3}
                                className="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 resize-none"
                            />
                        </div>
                    </div>
                )}

                {/* Labs Tab */}
                {activeTab === 'labs' && (
                    <div className="space-y-3">
                        <div className="bg-gray-50 rounded-xl overflow-hidden">
                            <table className="w-full text-xs">
                                <thead className="bg-gray-100">
                                    <tr>
                                        <th className="px-3 py-2 text-left text-gray-600">Test</th>
                                        <th className="px-3 py-2 text-left text-gray-600">Value</th>
                                        <th className="px-3 py-2 text-left text-gray-600">Range</th>
                                        <th className="px-3 py-2 text-center text-gray-600">Flag</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {(data?.labs || []).map((lab, idx) => (
                                        <tr key={idx} className="border-t border-gray-100">
                                            <td className="px-2 py-1">
                                                <input
                                                    type="text"
                                                    value={lab.test}
                                                    onChange={(e) => handleLabChange(idx, 'test', e.target.value)}
                                                    placeholder="WBC"
                                                    className="w-full px-2 py-1 border border-gray-200 rounded text-xs"
                                                />
                                            </td>
                                            <td className="px-2 py-1">
                                                <input
                                                    type="text"
                                                    value={lab.value}
                                                    onChange={(e) => handleLabChange(idx, 'value', e.target.value)}
                                                    placeholder="15.2"
                                                    className="w-full px-2 py-1 border border-gray-200 rounded text-xs"
                                                />
                                            </td>
                                            <td className="px-2 py-1">
                                                <input
                                                    type="text"
                                                    value={lab.range}
                                                    onChange={(e) => handleLabChange(idx, 'range', e.target.value)}
                                                    placeholder="4-11"
                                                    className="w-full px-2 py-1 border border-gray-200 rounded text-xs"
                                                />
                                            </td>
                                            <td className="px-2 py-1 text-center">
                                                <select
                                                    value={lab.flag}
                                                    onChange={(e) => handleLabChange(idx, 'flag', e.target.value)}
                                                    className="px-2 py-1 border border-gray-200 rounded text-xs"
                                                >
                                                    <option value="">-</option>
                                                    <option value="high">↑</option>
                                                    <option value="low">↓</option>
                                                </select>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                        <button
                            onClick={addLab}
                            className="w-full py-2 border-2 border-dashed border-gray-200 rounded-lg text-xs text-gray-500 hover:border-indigo-300 hover:text-indigo-600"
                        >
                            + Add Lab Result
                        </button>
                    </div>
                )}

                {/* Context Tab */}
                {activeTab === 'context' && (
                    <div className="space-y-4">
                        <div>
                            <label className="block text-xs font-medium text-gray-500 mb-1">Comorbidities</label>
                            <textarea
                                value={data?.comorbidities || ''}
                                onChange={(e) => handleChange('comorbidities', e.target.value)}
                                placeholder="HTN, DM, COPD, etc."
                                rows={2}
                                className="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 resize-none"
                            />
                        </div>
                        <div>
                            <label className="block text-xs font-medium text-gray-500 mb-1">Current Medications</label>
                            <textarea
                                value={data?.medications || ''}
                                onChange={(e) => handleChange('medications', e.target.value)}
                                placeholder="Metoprolol 50mg, Lisinopril 10mg..."
                                rows={2}
                                className="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 resize-none"
                            />
                        </div>
                        <div>
                            <label className="block text-xs font-medium text-gray-500 mb-1">Allergies</label>
                            <div className="flex gap-2">
                                <input
                                    type="text"
                                    value={data?.allergies || ''}
                                    onChange={(e) => handleChange('allergies', e.target.value)}
                                    placeholder="Penicillin, Sulfa, NKDA..."
                                    className="flex-1 px-3 py-2 border border-gray-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500"
                                />
                                {data?.allergies && (
                                    <span className="px-3 py-2 bg-red-100 text-red-600 rounded-lg text-xs font-medium flex items-center">
                                        ⚠️ Alert
                                    </span>
                                )}
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Run Analysis Button */}
            <button
                onClick={onAnalyze}
                disabled={loading}
                className="mt-4 w-full py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl font-semibold text-sm hover:from-indigo-700 hover:to-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
                {loading ? (
                    <>
                        <div className="spinner w-4 h-4"></div>
                        Analyzing...
                    </>
                ) : (
                    <>
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                        </svg>
                        Run CDSS Analysis
                    </>
                )}
            </button>
        </div>
    )
}

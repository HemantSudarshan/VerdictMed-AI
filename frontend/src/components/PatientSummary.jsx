import { useState } from 'react'

export default function PatientSummary({ data, onChange }) {
    const [vitals, setVitals] = useState(data?.vitals || {
        hr: '',
        bp_systolic: '',
        bp_diastolic: '',
        spo2: '',
        temp: ''
    })

    const handleChange = (field, value) => {
        const updated = { ...data, [field]: value }
        onChange?.(updated)
    }

    const handleVitalChange = (field, value) => {
        const newVitals = { ...vitals, [field]: value }
        setVitals(newVitals)
        onChange?.({ ...data, vitals: newVitals })
    }

    return (
        <div className="bg-white rounded-2xl shadow-lg p-5 h-full">
            <h2 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
                Patient Summary
            </h2>

            {/* Patient ID */}
            <div className="mb-4">
                <label className="block text-xs font-medium text-gray-500 mb-1">Patient ID</label>
                <input
                    type="text"
                    value={data?.patientId || ''}
                    onChange={(e) => handleChange('patientId', e.target.value)}
                    placeholder="Enter patient ID"
                    className="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                />
            </div>

            {/* Age & Sex Row */}
            <div className="grid grid-cols-2 gap-3 mb-4">
                <div>
                    <label className="block text-xs font-medium text-gray-500 mb-1">Age</label>
                    <input
                        type="number"
                        value={data?.age || ''}
                        onChange={(e) => handleChange('age', e.target.value)}
                        placeholder="Years"
                        className="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500"
                    />
                </div>
                <div>
                    <label className="block text-xs font-medium text-gray-500 mb-1">Sex</label>
                    <select
                        value={data?.sex || ''}
                        onChange={(e) => handleChange('sex', e.target.value)}
                        className="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500"
                    >
                        <option value="">Select</option>
                        <option value="male">Male</option>
                        <option value="female">Female</option>
                        <option value="other">Other</option>
                    </select>
                </div>
            </div>

            {/* Vitals Section */}
            <div className="mb-4">
                <label className="block text-xs font-medium text-gray-500 mb-2">Vital Signs</label>
                <div className="bg-gray-50 rounded-xl p-3 space-y-2">
                    {/* Heart Rate */}
                    <div className="flex items-center justify-between">
                        <span className="text-xs text-gray-600 flex items-center gap-1">
                            <span className="text-red-500">❤️</span> HR
                        </span>
                        <div className="flex items-center gap-1">
                            <input
                                type="number"
                                value={vitals.hr}
                                onChange={(e) => handleVitalChange('hr', e.target.value)}
                                placeholder="--"
                                className="w-16 px-2 py-1 border border-gray-200 rounded text-sm text-center"
                            />
                            <span className="text-xs text-gray-400">bpm</span>
                        </div>
                    </div>

                    {/* Blood Pressure */}
                    <div className="flex items-center justify-between">
                        <span className="text-xs text-gray-600 flex items-center gap-1">
                            <span className="text-blue-500">🩸</span> BP
                        </span>
                        <div className="flex items-center gap-1">
                            <input
                                type="number"
                                value={vitals.bp_systolic}
                                onChange={(e) => handleVitalChange('bp_systolic', e.target.value)}
                                placeholder="--"
                                className="w-12 px-2 py-1 border border-gray-200 rounded text-sm text-center"
                            />
                            <span className="text-gray-400">/</span>
                            <input
                                type="number"
                                value={vitals.bp_diastolic}
                                onChange={(e) => handleVitalChange('bp_diastolic', e.target.value)}
                                placeholder="--"
                                className="w-12 px-2 py-1 border border-gray-200 rounded text-sm text-center"
                            />
                            <span className="text-xs text-gray-400">mmHg</span>
                        </div>
                    </div>

                    {/* SpO2 */}
                    <div className="flex items-center justify-between">
                        <span className="text-xs text-gray-600 flex items-center gap-1">
                            <span className="text-cyan-500">🫁</span> SpO₂
                        </span>
                        <div className="flex items-center gap-1">
                            <input
                                type="number"
                                value={vitals.spo2}
                                onChange={(e) => handleVitalChange('spo2', e.target.value)}
                                placeholder="--"
                                className="w-16 px-2 py-1 border border-gray-200 rounded text-sm text-center"
                            />
                            <span className="text-xs text-gray-400">%</span>
                        </div>
                    </div>

                    {/* Temperature */}
                    <div className="flex items-center justify-between">
                        <span className="text-xs text-gray-600 flex items-center gap-1">
                            <span className="text-orange-500">🌡️</span> Temp
                        </span>
                        <div className="flex items-center gap-1">
                            <input
                                type="number"
                                step="0.1"
                                value={vitals.temp}
                                onChange={(e) => handleVitalChange('temp', e.target.value)}
                                placeholder="--"
                                className="w-16 px-2 py-1 border border-gray-200 rounded text-sm text-center"
                            />
                            <span className="text-xs text-gray-400">°C</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Chief Complaint */}
            <div>
                <label className="block text-xs font-medium text-gray-500 mb-1">Chief Complaint</label>
                <textarea
                    value={data?.chiefComplaint || ''}
                    onChange={(e) => handleChange('chiefComplaint', e.target.value)}
                    placeholder="Primary reason for visit..."
                    rows={3}
                    className="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 resize-none"
                />
            </div>
        </div>
    )
}

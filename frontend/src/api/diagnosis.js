import axios from 'axios'

const API_URL = 'http://localhost:8000'
const API_KEY = 'dev-test-key'

export async function diagnose(formData) {
    const response = await axios.post(
        `${API_URL}/api/v1/diagnose`,
        {
            symptoms: formData.symptoms,
            patient_id: formData.patientId || null,
            age: formData.age ? parseInt(formData.age) : null,
            sex: formData.sex || null
        },
        {
            headers: {
                'X-API-Key': API_KEY,
                'Content-Type': 'application/json'
            }
        }
    )
    return response.data
}

export async function checkHealth() {
    const response = await axios.get(`${API_URL}/health`)
    return response.data
}

# NEXUS AI — Machine Learning Microservices

Enterprise Machine Learning microservice built with **FastAPI**, **Scikit-Learn**, and **NumPy** for employee sentiment, burnout risk prediction, IT/HR support ticket classification, and security anomaly detection.

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/lithik-raj/Unified_employee_service_application&root-directory=ml-services)

---

## ⚡ Features & Models

1. **Burnout Risk Prediction**
   - **Model:** `RandomForestClassifier` (100 estimators)
   - **Endpoint:** `POST /predict-burnout`
   - **Signals:** Working hours, meeting density, leave utilization, ticket volume, attendance rate.

2. **Automated Support Ticket Classifier**
   - **Model:** NLP Pipeline (`TfidfVectorizer` + `MultinomialNB`)
   - **Endpoint:** `POST /classify-ticket`
   - **Categories:** IT Support, HR Operations, Finance & Payroll, Facilities & Operations.

3. **Security Anomaly Detection**
   - **Model:** Unsupervised `IsolationForest`
   - **Endpoint:** `POST /detect-anomaly`
   - **Signals:** Excessive failed logins, abnormal data downloads, after-hours access patterns, unknown IP addresses.

4. **Face Embedding & Verification**
   - **Model:** Cosine descriptor matching / DeepFace Facenet fallback
   - **Endpoint:** `POST /compare-faces`, `POST /generate-embedding`

5. **Interactive Console & OpenAPI Documentation**
   - **Interactive Console:** `GET /`
   - **Swagger UI:** `GET /docs`
   - **Health Check:** `GET /health`
   - **Department Analytics:** `GET /analytics/burnout-by-dept`

---

## 🚀 Deployment on Vercel

Configured for **@vercel/python** serverless deployment via `vercel.json` and `api/index.py`:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/index.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "api/index.py"
    }
  ]
}
```

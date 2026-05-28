# Architecture — Swiss ED Predictor

## Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────┐
│                    SOURCES OPEN DATA                         │
│  SpiGes/OFSP    MétéoSuisse    opentransport    OFS         │
└────────┬────────────┬──────────────┬────────────┬───────────┘
         ▼            ▼              ▼            ▼
┌─────────────────────────────────────────────────────────────┐
│              DATA PIPELINE  (src/ingestion/ + features/)     │
│  Pandas · ETL · Feature Engineering · Normalisation         │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   ML ENGINE  (src/model/)                    │
│  XGBoost (MVP) · Scikit-learn · SHAP · Scoring J+1→J+5     │
│  ── V2: LSTM + MLflow tracking ──────────────────────────   │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              API BACKEND  (src/api/)  ✅ MVP                 │
│  FastAPI · REST endpoints · Auth scaffold · Docker          │
│  GET  /health         → system status                       │
│  POST /predict        → admission forecast                  │
│  GET  /cantons        → reference data                      │
│  GET  /model/info     → model metadata                      │
│  ── V2: JWT/RBAC · multi-tenant · FHIR HL7 ─────────────   │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   DASHBOARD  (src/dashboard/)                │
│                                                             │
│  ✅ MVP  →  Streamlit  (livrable hackathon 48h)             │
│     KPIs · Chart 72h · SHAP features · Alertes opérat.     │
│                                                             │
│  🔜 V2   →  React + D3.js                                   │
│     Interface multi-canton · temps réel · export PDF        │
└─────────────────────────────────────────────────────────────┘

Infrastructure: Docker · docker-compose · GitHub Actions CI/CD · MIT
```

## Décisions techniques MVP vs V2

| Composant | MVP (Hackathon 48h) | V2 (Post-hackathon) |
|-----------|--------------------|--------------------|
| Dashboard | **Streamlit** | React + D3.js |
| Auth | Stub / passthrough | JWT + RBAC cantonal |
| Modèle | **XGBoost** | LSTM + MLflow |
| Deploy | Docker local | Kubernetes / Renku |
| Données | SpiGes CSV local | API live + streaming |

## Endpoints API (MVP)

```
GET  /health          → status, model_loaded, version
POST /predict         → {canton, horizon_hours} → {admissions, alert, confidence}
GET  /cantons         → liste des cantons supportés
GET  /model/info      → métadonnées modèle + référence scientifique
```

## Roadmap

```
Phase 1 — Hackathon    XGBoost · FastAPI · Streamlit · Docker · GitHub
Phase 2 — Q3 2026      LSTM · MLflow · React/D3 · Canton pilote · API live
Phase 3 — Q4 2026      FHIR HL7 · Dispatch 144 · Multi-canton · Kubernetes
```

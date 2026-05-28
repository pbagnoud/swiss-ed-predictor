# 🏥 Swiss ED Predictor
### Système Prédictif d'Anticipation des Pics aux Urgences Hospitalières Suisses

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GovTech Hackathon 2026](https://img.shields.io/badge/GovTech-Hackathon%202026-blue.svg)](https://www.govtech-hackathon.ch)
[![Open Data](https://img.shields.io/badge/Data-100%25%20Swiss%20Open%20Data-red.svg)](docs/data_sources.md)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org)

> **GovTech Hackathon 2026 · Zollikofen (BE) · 28–29 Mai 2026**  
> Challenge Owner: Albert Deutou Ngodji

---

## 🎯 Le Problème

Les urgences hospitalières suisses font face à des **pics d'affluence imprévisibles** qui saturent les capacités, allongent les temps d'attente et dégradent la qualité des soins.

Aucun outil national ne permet aujourd'hui d'anticiper ces pics en croisant des données multi-sources ouvertes.

---

## 💡 Notre Solution

Un pipeline ML qui croise **4 sources de données ouvertes suisses** pour prédire les pics d'affluence à **24–72h**, reproductible par tout canton sans coût.

```
SpiGes/OFSP  ──┐
MétéoSuisse  ──┤──▶  Feature Engineering  ──▶  XGBoost  ──▶  Dashboard
opentransport──┤
OFS          ──┘
```

### Différenciation vs solutions existantes

| Critère | ProcSim 🇨🇭 | CALYPS Saniia 🇨🇭 | **Notre projet** ✅ |
|---------|------------|-----------------|-------------------|
| Données | Internes | Internes | **100% Open Data CH** |
| Accès | Commercial | Commercial | **Open Source MIT** |
| Sources | 1 (hospit.) | 1 (hospit.) | **4 sources croisées** |
| Cantons | Sur devis | Sur devis | **Gratuit / libre** |

---

## 📊 Sources de Données

| Source | Type | Usage |
|--------|------|-------|
| [SpiGes / OFSP](https://www.bag.admin.ch/spiges) | Statistiques hospitalières annuelles | Patterns saisonniers & démographiques |
| [MétéoSuisse](https://www.meteosuisse.admin.ch/services-et-publications/service/open-data.html) | API météo temps réel | Signal court terme |
| [opentransportdata.swiss](https://opentransportdata.swiss) | Mobilité & flux transport | Proxy d'affluence |
| [OFS](https://www.bfs.admin.ch) | Données démographiques cantonales | Contexte population |

---

## 🚀 Quickstart

```bash
# 1. Cloner le repo
git clone https://github.com/swatch/swiss-ed-predictor.git
cd swiss-ed-predictor

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Télécharger les données
python scripts/download_data.py

# 4. Entraîner le modèle
python src/model/train.py

# 5. Lancer le dashboard
python src/dashboard/app.py
```

---

## 🗂️ Structure du Projet

```
swiss-ed-predictor/
├── data/
│   ├── raw/              # Données brutes (non versionnées)
│   ├── processed/        # Données transformées
│   └── external/         # Données de référence
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_evaluation.ipynb
├── src/
│   ├── ingestion/        # Connecteurs API (Météo, Transport, OFS)
│   ├── features/         # Feature engineering
│   ├── model/            # Entraînement XGBoost
│   ├── dashboard/        # Interface décideurs (Streamlit)
│   └── utils/            # Helpers communs
├── tests/
├── docs/
│   ├── data_sources.md
│   └── architecture.md
├── scripts/
│   └── download_data.py
├── requirements.txt
├── Makefile
└── README.md
```

---

## 📅 Plan MVP — 48 Heures

### Jour 1 — Fondations & Modèle
- [x] Setup repo & pipeline CI
- [ ] Ingestion données SpiGes, MétéoSuisse, opentransportdata
- [ ] Nettoyage & feature engineering
- [ ] Entraînement modèle XGBoost baseline

### Jour 2 — Optimisation & Livraison
- [ ] Évaluation & optimisation du modèle
- [ ] Dashboard Streamlit (décideurs cantonaux)
- [ ] Documentation & présentation finale

---

## 🔬 Références Scientifiques

- **ProcSim (EPFL/Suisse)** — LSTM, 93.35% précision, prédiction 14 jours
- **King et al. (Nature npj Digital Medicine, 2022)** — XGBoost, AUROC 0.90, 109k visites UK
- **Weidman et al. (JAMA Network Open, 2025)** — ML triage pré-hospitalier, transport soins critiques

---

## 🤝 Équipe

| Nom | Rôle |
|-----|------|
| Albert Deutou Ngodji | Challenge Owner · Backend · ML |
| *[Open]* | Data Scientist · Séries temporelles |
| *[Open]* | Expert SpiGes · Santé publique |
| *[Open]* | Data Visualization · Dashboard |

---

## 📜 Licence

MIT License — voir [LICENSE](LICENSE)

> Ce projet est développé dans le cadre du GovTech Hackathon 2026.  
> Il est conçu pour être librement reproductible par tout canton suisse.

---

## 📬 Contact

**Albert Deutou Ngodji** · a.deutou@gmail.com  
GitHub: [@swatch](https://github.com/adeutou)

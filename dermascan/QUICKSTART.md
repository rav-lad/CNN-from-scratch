# 🚀 DermaScan - Quick Start Guide

Ce guide vous permet de démarrer rapidement avec DermaScan.

## Installation (5 minutes)

### 1. Prérequis
```bash
# Vérifier Python (3.8+ requis)
python --version

# Cloner le repo (si pas déjà fait)
git clone <repo-url>
cd CNN-from-scratch
```

### 2. Environnement Virtuel
```bash
# Créer l'environnement
python -m venv .venv

# Activer (Linux/Mac)
source .venv/bin/activate

# Activer (Windows)
.venv\Scripts\activate
```

### 3. Dépendances
```bash
# Installer les dépendances de base
pip install -e ".[dev]"

# Installer les dépendances DermaScan
pip install -r dermascan/requirements.txt
```

## Démarrage Rapide (1 minute)

### Option 1: Script Shell (Recommandé - Linux/Mac)
```bash
bash dermascan/scripts/run_server.sh
```

### Option 2: Python Direct
```bash
python -m uvicorn dermascan.api.app:app --reload --port 8000
```

### Option 3: Depuis le module API
```bash
cd dermascan/api
python app.py
```

## Utilisation

1. **Ouvrir le navigateur**: http://localhost:8000
2. **Télécharger une image**: Cliquez ou glissez-déposez
3. **Analyser**: Cliquez sur "Analyser l'image"
4. **Résultats**: Visualisez les prédictions

## Test de l'API

### Avec cURL
```bash
# Health check
curl http://localhost:8000/api/health

# Liste des conditions
curl http://localhost:8000/api/conditions

# Prédiction (remplacer path/to/image.jpg)
curl -X POST http://localhost:8000/api/predict \
  -F "file=@path/to/image.jpg"
```

### Avec Python
```python
import requests

# Upload et prédiction
url = "http://localhost:8000/api/predict"
files = {"file": open("skin_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Structure des Fichiers

```
dermascan/
├── api/
│   └── app.py              # ✅ Serveur FastAPI
├── preprocessing/
│   └── image_processor.py  # ✅ Traitement d'images
├── inference/
│   └── predictor.py        # ✅ Modèle CNN
├── database/
│   └── conditions.py       # ✅ Infos médicales
└── configs/
    └── dermascan_model.yaml # ✅ Configuration

frontend/
├── templates/
│   └── index.html          # ✅ Interface web
└── static/
    ├── css/styles.css      # ✅ Styles
    └── js/app.js           # ✅ Logique frontend
```

## Problèmes Courants

### Erreur: Module 'fastapi' not found
```bash
pip install -r dermascan/requirements.txt
```

### Erreur: Port 8000 already in use
```bash
# Changer de port
python -m uvicorn dermascan.api.app:app --port 8080
```

### Erreur: Permission denied (run_server.sh)
```bash
chmod +x dermascan/scripts/run_server.sh
```

### Warning: Model weights not found
C'est normal! Le modèle sera initialisé avec des poids aléatoires.
Pour entraîner le modèle, voir la section suivante.

## Prochaines Étapes

### 1. Télécharger les Données (Optionnel)
```bash
python -m dermascan.scripts.download_data --dataset ham10000
# Suivre les instructions affichées
```

### 2. Entraîner le Modèle (Optionnel)
```bash
# Nécessite le dataset HAM10000
python -m src.cli.train --config dermascan/configs/dermascan_model.yaml
```

### 3. Explorer la Documentation
- [README DermaScan](README.md) - Documentation complète
- [Plan Global](../DERMASCAN_PLAN.md) - Architecture détaillée
- [README Principal](../README.md) - Projet CNN from Scratch

## Endpoints API Disponibles

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Interface web |
| `/api/health` | GET | Status du serveur |
| `/api/predict` | POST | Prédiction sur image |
| `/api/conditions` | GET | Liste des conditions |
| `/api/conditions/{name}` | GET | Détails d'une condition |

## Exemples de Réponses

### Health Check
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

### Prédiction
```json
{
  "success": true,
  "predictions": [
    {
      "condition": "Melanocytic Nevus",
      "confidence": 0.87,
      "severity": "Low (Usually benign)",
      "description": "...",
      "recommendations": [...]
    }
  ],
  "warning": "This is not a medical diagnosis..."
}
```

## Développement

### Modifier le Frontend
```bash
# Éditer les fichiers
frontend/templates/index.html   # Structure HTML
frontend/static/css/styles.css  # Apparence
frontend/static/js/app.js       # Logique

# Le serveur rechargera automatiquement (--reload)
```

### Modifier le Backend
```bash
# Éditer les fichiers
dermascan/api/app.py           # Endpoints API
dermascan/inference/predictor.py  # Modèle
dermascan/preprocessing/image_processor.py  # Preprocessing

# Le serveur rechargera automatiquement (--reload)
```

### Tests
```bash
# Tests unitaires (quand implémentés)
pytest tests/dermascan/

# Test manuel de l'API
curl http://localhost:8000/api/health
```

## Arrêter le Serveur

Appuyez sur `Ctrl + C` dans le terminal.

## Aide et Support

- **Documentation**: [dermascan/README.md](README.md)
- **Plan du projet**: [DERMASCAN_PLAN.md](../DERMASCAN_PLAN.md)
- **Issues**: Ouvrir une issue sur GitHub

## Checklist de Démarrage

- [ ] Python 3.8+ installé
- [ ] Environnement virtuel créé et activé
- [ ] Dépendances installées
- [ ] Serveur démarré avec succès
- [ ] http://localhost:8000 accessible
- [ ] Test de l'API health réussi
- [ ] Upload d'une image test fonctionnel

---

**Prêt à démarrer? Exécutez:**
```bash
bash dermascan/scripts/run_server.sh
```

Puis ouvrez http://localhost:8000 dans votre navigateur! 🎉

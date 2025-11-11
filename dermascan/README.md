# 🔬 DermaScan - Diagnostic Dermatologique par IA

**DermaScan** est une application d'intelligence artificielle pour l'analyse et la classification de conditions dermatologiques à partir d'images de peau.

> ⚠️ **Disclaimer**: Cet outil est à but éducatif et ne remplace en aucun cas un diagnostic médical professionnel. Consultez toujours un dermatologue qualifié pour tout problème de peau.

## 🎯 Objectif

Permettre à un utilisateur de:
1. Télécharger une photo d'une lésion cutanée
2. Recevoir une prédiction sur la condition possible
3. Obtenir des informations détaillées et des recommandations
4. Être guidé vers une consultation médicale appropriée

## 🏗️ Architecture

```
DermaScan/
├── API Backend (FastAPI)
│   ├── Endpoints REST
│   ├── Gestion des uploads
│   └── Serveur de prédictions
│
├── Modèle IA (CNN NumPy)
│   ├── Architecture personnalisée
│   ├── Entraînement sur HAM10000
│   └── 7 classes de conditions cutanées
│
├── Preprocessing
│   ├── Redimensionnement d'images
│   ├── Normalisation
│   └── Augmentation de données
│
├── Base de Données Médicale
│   ├── Descriptions des conditions
│   ├── Symptômes associés
│   └── Recommandations
│
└── Frontend Web
    ├── Interface d'upload
    ├── Affichage des résultats
    └── Informations médicales
```

## 🧬 Conditions Détectables

Le modèle peut identifier 7 types de conditions dermatologiques:

| Condition | Abréviation | Sévérité |
|-----------|-------------|----------|
| Actinic Keratosis | AK | Faible à Modérée |
| Basal Cell Carcinoma | BCC | Modérée à Élevée |
| Benign Keratosis | BKL | Faible (Bénin) |
| Dermatofibroma | DF | Faible (Bénin) |
| **Melanoma** | MEL | **Élevée** |
| Melanocytic Nevus (Mole) | NV | Faible |
| Vascular Lesion | VASC | Faible |

## 🚀 Installation et Démarrage

### Prérequis

- Python 3.8+
- pip
- Environnement virtuel (recommandé)

### Installation

```bash
# 1. Cloner le repository
git clone <repo-url>
cd CNN-from-scratch

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Installer les dépendances de base
pip install -e ".[dev]"

# 4. Installer les dépendances DermaScan
pip install -r dermascan/requirements.txt
```

### Démarrage Rapide

```bash
# Méthode 1: Script shell (Linux/Mac)
bash dermascan/scripts/run_server.sh

# Méthode 2: Python directement
python -m uvicorn dermascan.api.app:app --reload --port 8000

# Méthode 3: Depuis l'app
cd dermascan/api
python app.py
```

Ouvrez votre navigateur à: **http://localhost:8000**

## 🐳 Déploiement Docker

DermaScan peut être déployé facilement avec Docker pour un environnement isolé et reproductible.

### Quick Start avec Docker

```bash
# Méthode 1: Docker Compose (Recommandé)
docker-compose up -d

# Méthode 2: Script automatique
bash dermascan/scripts/docker_run.sh

# Méthode 3: Docker build & run manuel
docker build -t dermascan:latest .
docker run -d -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/reports:/app/reports \
  dermascan:latest
```

### Modes de Déploiement

**Développement (avec hot-reload):**
```bash
docker-compose -f docker-compose.dev.yml up
# Code changes → Auto-reload
```

**Production (avec Nginx):**
```bash
docker-compose --profile production up -d
# API: http://localhost:8000
# Web: http://localhost (nginx avec rate limiting)
```

### Scripts Docker Disponibles

```bash
# Build l'image
bash dermascan/scripts/docker_build.sh

# Run développement
bash dermascan/scripts/docker_run.sh dev

# Run production
bash dermascan/scripts/docker_run.sh prod

# Deploy complet (build + test + push)
bash dermascan/scripts/docker_deploy.sh
```

### Configuration

**Variables d'environnement (.env):**
```bash
PORT=8000
LOG_LEVEL=info
MODEL_PATH=/app/data/dermatology/models/dermascan_best.npz
```

**Volumes persistants:**
- `./data:/app/data` - Datasets et modèles
- `./reports:/app/reports` - Logs et métriques
- `./checkpoints:/app/checkpoints` - Checkpoints d'entraînement

**Documentation complète:** [DOCKER.md](../DOCKER.md)

## 📊 Données d'Entraînement

### Dataset Recommandé: HAM10000

Le dataset **HAM10000** (Human Against Machine avec 10,000 images) est recommandé pour l'entraînement:

- 10,015 images dermatoscopiques
- 7 catégories de lésions pigmentées
- Images de haute qualité
- Métadonnées complètes

### Téléchargement des Données

```bash
# Afficher les instructions de téléchargement
python -m dermascan.scripts.download_data --dataset ham10000

# Ou manuellement:
# 1. Visitez: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
# 2. Téléchargez le dataset
# 3. Extrayez dans: data/dermatology/raw/HAM10000/
```

### Structure des Données

```
data/dermatology/
├── raw/
│   └── HAM10000/
│       ├── HAM10000_images_part_1/
│       ├── HAM10000_images_part_2/
│       └── HAM10000_metadata.csv
├── processed/
│   ├── train/
│   ├── val/
│   └── test/
└── models/
    └── dermascan_best.npz
```

## 🎓 Entraînement du Modèle

```bash
# Entraîner le modèle DermaScan
python -m src.cli.train --config dermascan/configs/dermascan_model.yaml

# Évaluer les performances
python -m src.cli.evaluate \
    --config dermascan/configs/dermascan_model.yaml \
    --weights data/dermatology/models/dermascan_best.npz
```

### Configuration du Modèle

Le fichier `dermascan/configs/dermascan_model.yaml` contient:
- Architecture du CNN
- Hyperparamètres d'entraînement
- Configuration d'augmentation de données
- Callbacks et métriques

## 🧪 API Endpoints

### `GET /`
Page d'accueil de l'application web

### `GET /api/health`
Vérification de l'état du serveur
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

### `POST /api/predict`
Prédiction sur une image uploadée

**Request:**
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -F "file=@skin_image.jpg"
```

**Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "condition": "Melanocytic Nevus",
      "confidence": 0.87,
      "description": "...",
      "severity": "Low",
      "recommendations": ["..."]
    }
  ],
  "warning": "This is not a medical diagnosis..."
}
```

### `GET /api/conditions`
Liste toutes les conditions supportées

### `GET /api/conditions/{condition_name}`
Détails d'une condition spécifique

## 📱 Utilisation Frontend

1. **Upload**: Cliquez ou glissez-déposez une image
2. **Analyse**: Le modèle analyse l'image
3. **Résultats**: Visualisez les prédictions avec:
   - Nom de la condition
   - Niveau de confiance
   - Sévérité
   - Description
   - Symptômes
   - Recommandations
   - Urgence de consultation

## 🧠 Architecture du Modèle CNN

```
Input (224 x 224 x 3)
    ↓
[Conv2D(32) → BatchNorm → ReLU → Conv2D(32) → BatchNorm → ReLU → MaxPool]
    ↓
[Conv2D(64) → BatchNorm → ReLU → Conv2D(64) → BatchNorm → ReLU → MaxPool]
    ↓
[Conv2D(128) → BatchNorm → ReLU → Conv2D(128) → BatchNorm → ReLU → MaxPool]
    ↓
[Conv2D(256) → BatchNorm → ReLU → MaxPool]
    ↓
Dense(512) → ReLU → Dropout(0.5)
    ↓
Dense(7) → Softmax
    ↓
Output (7 classes)
```

**Caractéristiques:**
- 100% NumPy (pas de frameworks)
- Backpropagation manuelle
- BatchNormalization pour stabilité
- Dropout pour régularisation
- Architecture inspirée de ResNet/VGG

## 📈 Métriques de Performance

Le modèle est évalué sur:
- **Accuracy**: Précision globale
- **Precision**: Par classe
- **Recall**: Par classe
- **F1-Score**: Moyenne harmonique
- **Confusion Matrix**: Matrice de confusion
- **Per-class Metrics**: Métriques détaillées

## 🔒 Sécurité et Confidentialité

- Les images ne sont **pas sauvegardées** sur le serveur
- Traitement en mémoire uniquement
- Pas de base de données d'utilisateurs
- HTTPS recommandé en production
- Validation stricte des fichiers uploadés

## 🚧 Limitations

1. **Éducatif uniquement**: Ne remplace pas un médecin
2. **Dataset limité**: Entraîné sur des images dermatoscopiques
3. **7 classes**: Ne couvre pas toutes les conditions cutanées
4. **Qualité d'image**: Résultats optimaux avec images claires
5. **Pas de GPU**: Inférence en CPU (NumPy)

## 🔬 Améliorations Futures

- [ ] Support de datasets supplémentaires (ISIC, PAD-UFES-20)
- [ ] Augmentation du nombre de classes
- [ ] Technique d'explicabilité (Grad-CAM)
- [ ] Version mobile (TensorFlow Lite)
- [ ] Multi-langue (EN, FR, ES)
- [ ] Historique des analyses (avec consentement)
- [ ] Intégration avec systèmes de télémédecine

## 📚 Ressources

### Datasets
- [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- [ISIC Archive](https://www.isic-archive.com/)
- [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1)

### Références Médicales
- [American Academy of Dermatology](https://www.aad.org/)
- [Skin Cancer Foundation](https://www.skincancer.org/)
- [DermNet NZ](https://dermnetnz.org/)

### Papiers de Recherche
- Esteva et al. (2017) - "Dermatologist-level classification of skin cancer"
- Tschandl et al. (2018) - "The HAM10000 dataset"
- Codella et al. (2019) - "Skin Lesion Analysis Toward Melanoma Detection"

## 👥 Contribution

Ce projet est basé sur **CNN from Scratch** et utilise son infrastructure NumPy.

### Développement Local

```bash
# Tests
pytest tests/

# Linting
black dermascan/
flake8 dermascan/

# Type checking
mypy dermascan/
```

## 📄 License

MIT License - Voir LICENSE

## ⚕️ Avertissement Médical

**IMPORTANT**: DermaScan est un outil éducatif et de recherche. Il ne doit PAS être utilisé pour:
- Autodiagnostic sans supervision médicale
- Remplacer une consultation dermatologique
- Décisions de traitement
- Cas d'urgence médicale

**En cas de doute sur une lésion cutanée, consultez immédiatement un professionnel de santé qualifié.**

---

**Fait avec ❤️ et NumPy**

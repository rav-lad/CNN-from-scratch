# 📋 Plan Global DermaScan

## Vue d'Ensemble

**DermaScan** est une application web de diagnostic dermatologique assistée par IA, construite sur l'infrastructure CNN from Scratch existante.

### Objectif Principal
Permettre aux utilisateurs de soumettre une photo de leur peau et recevoir une analyse automatique identifiant des conditions dermatologiques potentielles.

---

## 🏗️ Architecture Technique

### 1. Stack Technologique

#### Backend
- **Framework API**: FastAPI (rapide, moderne, avec validation automatique)
- **Modèle IA**: CNN implémenté en NumPy (réutilisation du code existant)
- **Preprocessing**: Pillow pour manipulation d'images
- **Serveur**: Uvicorn (ASGI)

#### Frontend
- **Interface**: HTML5 + CSS3 + JavaScript Vanilla
- **Design**: Responsive, mobile-friendly
- **Features**: Drag-and-drop, preview, résultats interactifs

#### Data
- **Dataset**: HAM10000 (10,015 images dermatoscopiques)
- **Classes**: 7 types de lésions cutanées
- **Format**: Images JPG + métadonnées CSV

---

## 📁 Structure du Projet

```
CNN-from-scratch/
│
├── src/                          # Code CNN existant (réutilisé)
│   ├── core/                     # Losses, optimizers, metrics
│   ├── layers/                   # Conv2D, Dense, BatchNorm, etc.
│   ├── models/                   # Sequential model
│   ├── data/                     # MNIST, CIFAR-10
│   └── train/                    # Training loop, callbacks
│
├── dermascan/                    # 🆕 Nouveau module DermaScan
│   │
│   ├── api/                      # API REST
│   │   ├── app.py               # Application FastAPI principale
│   │   ├── routes/              # Endpoints organisés
│   │   └── schemas/             # Modèles Pydantic
│   │
│   ├── preprocessing/            # Traitement d'images
│   │   └── image_processor.py  # Resize, normalisation, augmentation
│   │
│   ├── inference/                # Prédictions
│   │   └── predictor.py        # Chargement modèle + inférence
│   │
│   ├── database/                 # Base de données médicale
│   │   └── conditions.py       # Infos sur les conditions cutanées
│   │
│   ├── models/                   # Architectures spécifiques
│   │   └── dermascan_cnn.py    # Architecture optimisée pour dermato
│   │
│   ├── configs/                  # Configurations
│   │   └── dermascan_model.yaml # Config training/inference
│   │
│   ├── scripts/                  # Scripts utilitaires
│   │   ├── download_data.py    # Téléchargement HAM10000
│   │   ├── train_dermascan.py  # Script d'entraînement
│   │   └── run_server.sh       # Démarrage serveur
│   │
│   └── README.md                # Documentation DermaScan
│
├── frontend/                     # 🆕 Interface utilisateur
│   ├── static/
│   │   ├── css/
│   │   │   └── styles.css      # Styles modernes
│   │   └── js/
│   │       └── app.js          # Logique frontend
│   └── templates/
│       └── index.html          # Page principale
│
├── data/
│   └── dermatology/             # 🆕 Données dermatologiques
│       ├── raw/                 # Dataset brut (HAM10000)
│       ├── processed/           # Données preprocessées
│       └── models/              # Modèles entraînés
│
├── notebooks/
│   └── dermascan/               # 🆕 Notebooks d'exploration
│
└── tests/
    └── dermascan/               # 🆕 Tests spécifiques
```

---

## 🔄 Flux de Fonctionnement

### 1. Upload & Preprocessing
```
Utilisateur → Upload Image (PNG/JPG)
    ↓
Validation (format, taille < 10MB)
    ↓
ImageProcessor.process_uploaded_image()
    ↓
Resize 224x224 → Normalize → Format (1, C, H, W)
```

### 2. Inférence
```
Image preprocessée
    ↓
DermaScanPredictor.predict()
    ↓
CNN Forward Pass (100% NumPy)
    ↓
Softmax → Top-3 Predictions
```

### 3. Enrichissement des Résultats
```
Predictions (classe + confidence)
    ↓
SkinConditionDatabase.get_condition_info()
    ↓
Ajouter: description, symptômes, recommendations, urgence
```

### 4. Affichage
```
Résultats enrichis → JSON Response
    ↓
Frontend JavaScript
    ↓
Affichage cartes colorées avec badges de sévérité
```

---

## 🧠 Architecture du Modèle CNN

### Design Choices

**Basé sur VGG/ResNet adapté pour images médicales:**

```
Input: 224 x 224 x 3 (RGB)

Block 1:
  Conv2D(3→32, 3x3, pad=1) → BatchNorm → ReLU
  Conv2D(32→32, 3x3, pad=1) → BatchNorm → ReLU
  MaxPool(2x2) → 112 x 112 x 32

Block 2:
  Conv2D(32→64, 3x3, pad=1) → BatchNorm → ReLU
  Conv2D(64→64, 3x3, pad=1) → BatchNorm → ReLU
  MaxPool(2x2) → 56 x 56 x 64

Block 3:
  Conv2D(64→128, 3x3, pad=1) → BatchNorm → ReLU
  Conv2D(128→128, 3x3, pad=1) → BatchNorm → ReLU
  MaxPool(2x2) → 28 x 28 x 128

Block 4:
  Conv2D(128→256, 3x3, pad=1) → BatchNorm → ReLU
  MaxPool(2x2) → 14 x 14 x 256

Classifier:
  Flatten → 256*14*14 = 50,176
  Dense(50176 → 512) → ReLU
  Dropout(0.5)
  Dense(512 → 7) → Softmax

Output: 7 classes
```

**Paramètres:**
- Total params: ~25M
- Entraînement: Adam, LR=0.001, Batch=32
- Régularisation: Dropout(0.5), Weight Decay, BatchNorm

---

## 🎯 Classes Détectées

| # | Classe | Code | Sévérité | Fréquence Dataset |
|---|--------|------|----------|-------------------|
| 0 | Actinic Keratosis | AK | Modérée | ~3% |
| 1 | Basal Cell Carcinoma | BCC | Élevée | ~5% |
| 2 | Benign Keratosis | BKL | Faible | ~11% |
| 3 | Dermatofibroma | DF | Faible | ~1% |
| 4 | **Melanoma** | MEL | **Très Élevée** | ~11% |
| 5 | Melanocytic Nevus | NV | Faible | ~67% |
| 6 | Vascular Lesion | VASC | Faible | ~1% |

**Note:** Dataset déséquilibré → Techniques:
- Class weighting
- Augmentation ciblée sur classes minoritaires
- Focal Loss (optionnel)

---

## 📊 Pipeline d'Entraînement

### Phase 1: Préparation des Données

```bash
# 1. Télécharger HAM10000
python -m dermascan.scripts.download_data --dataset ham10000

# 2. Structure attendue
data/dermatology/raw/HAM10000/
  ├── HAM10000_images_part_1/*.jpg  (5,000 images)
  ├── HAM10000_images_part_2/*.jpg  (5,015 images)
  └── HAM10000_metadata.csv

# 3. Preprocessing (optionnel, fait à la volée)
# - Resize to 224x224
# - Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# - Augmentation: rotation, flip, brightness
```

### Phase 2: Entraînement

```bash
# Configuration dans: dermascan/configs/dermascan_model.yaml

python -m src.cli.train --config dermascan/configs/dermascan_model.yaml

# Outputs:
# - Checkpoints: data/dermatology/models/dermascan_best.npz
# - Logs: reports/dermascan_training.csv
# - Figures: reports/figures/dermascan_*.png
```

### Phase 3: Évaluation

```bash
python -m src.cli.evaluate \
  --config dermascan/configs/dermascan_model.yaml \
  --weights data/dermatology/models/dermascan_best.npz

# Métriques:
# - Accuracy globale
# - Precision/Recall/F1 par classe
# - Confusion Matrix
# - ROC curves (7 classes)
```

---

## 🚀 Déploiement

### Développement Local

```bash
# 1. Installer dépendances
pip install -r dermascan/requirements.txt

# 2. Démarrer serveur
bash dermascan/scripts/run_server.sh
# ou
python -m uvicorn dermascan.api.app:app --reload --port 8000

# 3. Ouvrir navigateur
http://localhost:8000
```

### Production (Suggestions)

```bash
# Option 1: Docker
# Créer Dockerfile avec:
# - Python 3.9+ base image
# - Install requirements
# - Copy code + models
# - CMD: uvicorn dermascan.api.app:app --host 0.0.0.0 --port 8000

# Option 2: Cloud (Heroku, AWS, GCP)
# - Utiliser gunicorn + uvicorn workers
# - Variables d'environnement pour configs
# - CDN pour static files
# - Load balancer si scaling

# Option 3: Serverless (AWS Lambda + API Gateway)
# - Fonction Lambda pour inférence
# - S3 pour stockage modèle
# - API Gateway pour endpoints
```

---

## 🔒 Considérations Importantes

### Sécurité
- ✅ Validation stricte des uploads (type, taille)
- ✅ Pas de stockage des images utilisateur
- ✅ Traitement en mémoire uniquement
- ⚠️ HTTPS obligatoire en production
- ⚠️ Rate limiting pour éviter abus

### Médical & Légal
- ⚠️ **Disclaimer visible**: Pas un diagnostic médical
- ⚠️ **Recommandations**: Toujours consulter un dermatologue
- ⚠️ **Urgence**: Guidance claire pour cas sérieux (mélanome)
- ⚠️ **Conformité**: RGPD (pas de données stockées = OK)

### Performance
- Inférence: ~500ms-2s (CPU NumPy)
- Amélioration possible: Convertir en PyTorch/TF pour GPU
- Caching: Résultats identiques (hash image)

---

## 📈 Métriques de Succès

### Techniques
- [ ] Accuracy > 80% sur test set
- [ ] Recall melanoma > 90% (critique!)
- [ ] Temps inférence < 3s
- [ ] API response time < 5s

### Utilisateur
- [ ] Interface intuitive (upload en 1 clic)
- [ ] Résultats clairs et compréhensibles
- [ ] Informations médicales utiles
- [ ] Call-to-action vers consultation

---

## 🛣️ Roadmap

### Version 0.1 (MVP) ✅
- [x] Structure du projet
- [x] API FastAPI fonctionnelle
- [x] Frontend upload + résultats
- [x] Preprocessing images
- [x] Modèle CNN architecture
- [x] Base de données conditions
- [x] Documentation

### Version 0.2 (Training)
- [ ] Data loader HAM10000
- [ ] Pipeline d'entraînement complet
- [ ] Métriques et évaluation
- [ ] Modèle entraîné et validé
- [ ] Tests unitaires

### Version 0.3 (Enhancement)
- [ ] Augmentation de données avancée
- [ ] Class balancing
- [ ] Hyperparameter tuning
- [ ] Explicabilité (heatmaps)
- [ ] Multi-langue (EN/FR)

### Version 1.0 (Production Ready)
- [ ] Docker deployment
- [ ] CI/CD pipeline
- [ ] Monitoring & logging
- [ ] A/B testing
- [ ] Documentation utilisateur complète

---

## 🧪 Tests

### Tests Unitaires
```bash
# Preprocessing
pytest tests/dermascan/test_preprocessing.py

# Predictor
pytest tests/dermascan/test_predictor.py

# API
pytest tests/dermascan/test_api.py

# Database
pytest tests/dermascan/test_database.py
```

### Tests d'Intégration
```bash
# End-to-end: upload → predict → response
pytest tests/dermascan/test_integration.py

# Performance
pytest tests/dermascan/test_performance.py --benchmark
```

---

## 📚 Références

### Datasets
- **HAM10000**: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- **ISIC**: https://www.isic-archive.com/
- **PAD-UFES-20**: https://data.mendeley.com/datasets/zr7vgbcyr2/1

### Papers
1. Esteva et al. (2017) - "Dermatologist-level classification of skin cancer with deep neural networks"
2. Tschandl et al. (2018) - "The HAM10000 dataset, a large collection of multi-source dermatoscopic images"
3. Codella et al. (2019) - "Skin Lesion Analysis Toward Melanoma Detection 2018"

### Ressources Médicales
- American Academy of Dermatology: https://www.aad.org/
- Skin Cancer Foundation: https://www.skincancer.org/
- DermNet NZ: https://dermnetnz.org/

---

## 💡 Conseils de Développement

### Pour l'Entraînement
1. **Commencer petit**: Entraîner d'abord sur subset (1000 images)
2. **Valider pipeline**: S'assurer que tout fonctionne avant full training
3. **Monitor overfitting**: Val loss vs train loss
4. **Checkpoints fréquents**: Sauvegarder tous les 5 epochs
5. **Logs détaillés**: CSV + TensorBoard-like visualizations

### Pour l'API
1. **Gestion d'erreurs**: Try-catch partout avec messages clairs
2. **Validation stricte**: Pydantic schemas pour requests
3. **Timeout**: Limiter temps de traitement
4. **Logs**: Logger tous les appels API
5. **Versioning**: /api/v1/ pour évolutions futures

### Pour le Frontend
1. **Feedback utilisateur**: Loading spinners, messages d'erreur
2. **Responsive**: Tester mobile + desktop
3. **Accessibility**: Alt texts, ARIA labels
4. **Performance**: Lazy loading, compression images
5. **Analytics**: Tracking usage (anonyme)

---

## ✅ Checklist de Lancement

Avant de déployer en production:

- [ ] Tests passent (>90% coverage)
- [ ] Modèle validé (métriques acceptables)
- [ ] Disclaimer médical visible
- [ ] HTTPS configuré
- [ ] Rate limiting activé
- [ ] Logs & monitoring en place
- [ ] Backup du modèle
- [ ] Documentation à jour
- [ ] Terms of Service / Privacy Policy
- [ ] Contact / Support visible

---

**Projet créé avec ❤️ et NumPy - Pour l'éducation et la recherche médicale**

# 🐳 DermaScan - Guide Docker

Guide complet pour déployer DermaScan avec Docker.

## 📋 Table des Matières

- [Prérequis](#prérequis)
- [Quick Start](#quick-start)
- [Docker Compose](#docker-compose)
- [Production Deployment](#production-deployment)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Prérequis

### Installation Docker

**Linux:**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

**Mac:**
- Télécharger [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop)

**Windows:**
- Télécharger [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)

### Installation Docker Compose

```bash
# Linux
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Vérification
docker --version
docker-compose --version
```

---

## Quick Start

### Option 1: Docker Compose (Recommandé)

```bash
# Cloner le repo
git clone <repo-url>
cd CNN-from-scratch

# Démarrer avec Docker Compose
docker-compose up -d

# Vérifier les logs
docker-compose logs -f dermascan-api

# Accéder à l'application
open http://localhost:8000
```

### Option 2: Docker Run

```bash
# Build l'image
docker build -t dermascan:latest .

# Run le container
docker run -d \
  --name dermascan \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/reports:/app/reports \
  dermascan:latest

# Vérifier
curl http://localhost:8000/api/health
```

### Option 3: Script Automatique

```bash
# Build
bash dermascan/scripts/docker_build.sh

# Run (développement)
bash dermascan/scripts/docker_run.sh dev

# Run (production)
bash dermascan/scripts/docker_run.sh prod
```

---

## Docker Compose

### Configurations Disponibles

#### 1. Default Mode (Simple)

```yaml
# docker-compose.yml
services:
  dermascan-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./reports:/app/reports
```

**Démarrage:**
```bash
docker-compose up -d
```

**Accès:** http://localhost:8000

#### 2. Development Mode (Hot-Reload)

```yaml
# docker-compose.dev.yml
services:
  dermascan-dev:
    build: .
    command: uvicorn dermascan.api.app:app --reload
    volumes:
      - ./src:/app/src          # Code source monté
      - ./dermascan:/app/dermascan
      - ./frontend:/app/frontend
```

**Démarrage:**
```bash
docker-compose -f docker-compose.dev.yml up
```

**Features:**
- ✅ Hot-reload automatique
- ✅ Logs en temps réel
- ✅ Modifications code → Rechargement immédiat

#### 3. Production Mode (Nginx + API)

```bash
# Démarrer avec nginx reverse proxy
docker-compose --profile production up -d
```

**Services:**
- `dermascan-api` sur port 8000 (interne)
- `nginx` sur port 80/443 (public)

**Features:**
- ✅ Rate limiting
- ✅ Gzip compression
- ✅ SSL/TLS (si configuré)
- ✅ Load balancing ready

---

## Production Deployment

### 1. Build pour Production

```bash
# Set version
export VERSION=1.0.0

# Build avec tag version
docker build -t dermascan:${VERSION} -t dermascan:latest .

# Tester l'image
docker run --rm -d --name dermascan-test -p 8001:8000 dermascan:latest
sleep 5
curl http://localhost:8001/api/health
docker stop dermascan-test
```

### 2. Push vers Registry

#### Docker Hub
```bash
# Login
docker login

# Tag
docker tag dermascan:latest yourusername/dermascan:latest
docker tag dermascan:${VERSION} yourusername/dermascan:${VERSION}

# Push
docker push yourusername/dermascan:latest
docker push yourusername/dermascan:${VERSION}
```

#### AWS ECR
```bash
# Login
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

# Tag
docker tag dermascan:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/dermascan:latest

# Push
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/dermascan:latest
```

#### Google Container Registry
```bash
# Login
gcloud auth configure-docker

# Tag
docker tag dermascan:latest gcr.io/your-project/dermascan:latest

# Push
docker push gcr.io/your-project/dermascan:latest
```

### 3. Deploy Script

```bash
# Script complet de déploiement
bash dermascan/scripts/docker_deploy.sh
```

Ce script:
1. ✅ Build l'image
2. ✅ Test health check
3. ✅ Push vers registry (optionnel)
4. ✅ Instructions deployment cloud

---

## Configuration

### Variables d'Environnement

```bash
# .env file
PORT=8000
LOG_LEVEL=info
PYTHONUNBUFFERED=1

# Optional
MODEL_PATH=/app/data/dermatology/models/dermascan_best.npz
MAX_UPLOAD_SIZE=10485760  # 10MB
ALLOWED_ORIGINS=*
```

**Utilisation:**
```bash
# Avec docker run
docker run --env-file .env -p 8000:8000 dermascan:latest

# Avec docker-compose
# Ajouter dans docker-compose.yml:
services:
  dermascan-api:
    env_file: .env
```

### Volumes

```yaml
volumes:
  # Data persistence
  - ./data:/app/data                    # Datasets & models
  - ./reports:/app/reports              # Logs & figures
  - ./checkpoints:/app/checkpoints      # Model checkpoints

  # Development (optionnel)
  - ./src:/app/src                      # Source code
  - ./dermascan:/app/dermascan
  - ./frontend:/app/frontend
```

### Ports

| Port | Service | Description |
|------|---------|-------------|
| 8000 | API | FastAPI application |
| 80 | Nginx | HTTP (production) |
| 443 | Nginx | HTTPS (production) |

---

## Architecture Docker

### Dockerfile Multi-stage

```dockerfile
# Stage 1: Builder (build dependencies)
FROM python:3.10-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime (minimal image)
FROM python:3.10-slim
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY src/ dermascan/ frontend/ ./
CMD ["uvicorn", "dermascan.api.app:app", "--host", "0.0.0.0"]
```

**Avantages:**
- ✅ Image finale plus petite (~400MB vs 1GB+)
- ✅ Build cache optimisé
- ✅ Security (pas de build tools en production)

### Health Checks

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/api/health')"
```

**Monitoring:**
```bash
# Status des containers
docker ps

# Health check status
docker inspect dermascan-api | grep Health -A 10

# Logs du health check
docker logs dermascan-api 2>&1 | grep health
```

---

## Commandes Utiles

### Gestion des Containers

```bash
# Démarrer
docker-compose up -d

# Arrêter
docker-compose down

# Redémarrer
docker-compose restart

# Voir les logs
docker-compose logs -f

# Voir les logs d'un service spécifique
docker-compose logs -f dermascan-api

# Exécuter une commande dans le container
docker-compose exec dermascan-api bash

# Rebuild après changements
docker-compose up -d --build
```

### Inspection & Debug

```bash
# Entrer dans le container
docker exec -it dermascan-api bash

# Voir les processus
docker top dermascan-api

# Statistiques en temps réel
docker stats dermascan-api

# Inspecter la configuration
docker inspect dermascan-api

# Voir les volumes
docker volume ls
docker volume inspect cnn-from-scratch_dermascan-data
```

### Nettoyage

```bash
# Arrêter et supprimer containers
docker-compose down

# Supprimer avec volumes
docker-compose down -v

# Nettoyer images inutilisées
docker image prune

# Nettoyer tout
docker system prune -a
```

---

## Deployment Cloud

### AWS ECS (Elastic Container Service)

```bash
# 1. Push image vers ECR (voir section Registry)

# 2. Créer task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-def.json

# 3. Créer ou update service
aws ecs update-service \
  --cluster dermascan-cluster \
  --service dermascan-service \
  --task-definition dermascan:1 \
  --force-new-deployment
```

### Google Cloud Run

```bash
# Deploy directement depuis source
gcloud run deploy dermascan \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

# Ou depuis image
gcloud run deploy dermascan \
  --image gcr.io/your-project/dermascan:latest \
  --platform managed \
  --region us-central1
```

### Azure Container Instances

```bash
# Créer container instance
az container create \
  --resource-group dermascan-rg \
  --name dermascan \
  --image yourusername/dermascan:latest \
  --dns-name-label dermascan \
  --ports 8000
```

### Heroku (Container Registry)

```bash
# Login
heroku container:login

# Push
heroku container:push web -a dermascan-app

# Release
heroku container:release web -a dermascan-app

# Open
heroku open -a dermascan-app
```

---

## Troubleshooting

### Container ne démarre pas

```bash
# Vérifier les logs
docker-compose logs dermascan-api

# Erreurs courantes:
# - Port déjà utilisé → Changer port dans docker-compose.yml
# - Permissions → sudo docker-compose up
# - Build failed → docker-compose build --no-cache
```

### Port déjà utilisé

```bash
# Trouver le processus
sudo lsof -i :8000

# Tuer le processus
sudo kill -9 <PID>

# Ou changer le port
# Dans docker-compose.yml: "8001:8000"
```

### Permissions denied

```bash
# Ajouter user au groupe docker
sudo usermod -aG docker $USER
newgrp docker

# Ou run avec sudo (pas recommandé)
sudo docker-compose up
```

### Out of memory

```bash
# Augmenter mémoire Docker Desktop
# Settings → Resources → Memory → 4GB+

# Ou limiter dans docker-compose.yml
services:
  dermascan-api:
    deploy:
      resources:
        limits:
          memory: 2G
```

### Image trop grande

```bash
# Utiliser .dockerignore (déjà inclus)
# Vérifier taille
docker images dermascan

# Optimiser:
# - Multi-stage build (déjà fait)
# - Utiliser alpine: FROM python:3.10-alpine
# - Nettoyer cache: RUN --mount=type=cache
```

### Hot-reload ne fonctionne pas

```bash
# S'assurer d'utiliser docker-compose.dev.yml
docker-compose -f docker-compose.dev.yml up

# Vérifier volumes montés
docker-compose -f docker-compose.dev.yml config

# Redémarrer
docker-compose -f docker-compose.dev.yml restart
```

---

## Performance Optimization

### Build Cache

```dockerfile
# Copier requirements en premier (cache layer)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Puis copier code (change plus souvent)
COPY src/ ./src/
```

### Layer Optimization

```dockerfile
# Combiner commandes RUN
RUN apt-get update && \
    apt-get install -y package1 package2 && \
    rm -rf /var/lib/apt/lists/*
```

### Production Tips

```dockerfile
# Pas de cache pip
RUN pip install --no-cache-dir -r requirements.txt

# User non-root
RUN useradd -m appuser
USER appuser

# Disable .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
```

---

## Monitoring en Production

### Logs

```bash
# Logs en temps réel
docker-compose logs -f

# Logs avec timestamps
docker-compose logs -t

# Dernières 100 lignes
docker-compose logs --tail=100

# Logs vers fichier
docker-compose logs > dermascan.log
```

### Metrics avec Prometheus

```yaml
# docker-compose.yml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
```

### Grafana Dashboard

```yaml
services:
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

---

## Security Best Practices

### 1. User non-root
```dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

### 2. Scan vulnerabilities
```bash
# Avec Trivy
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image dermascan:latest

# Avec Snyk
snyk container test dermascan:latest
```

### 3. Secrets management
```bash
# Ne JAMAIS commit .env
# Utiliser Docker secrets ou cloud secret manager

# Docker Swarm secrets
echo "my_secret" | docker secret create api_key -
```

### 4. Network isolation
```yaml
networks:
  frontend:
  backend:
    internal: true  # Pas d'accès externe
```

---

## Ressources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Production Checklist](https://docs.docker.com/engine/security/security/)

---

## Support

Pour les problèmes Docker:
1. Vérifier les logs: `docker-compose logs`
2. Consulter ce guide
3. Ouvrir une issue GitHub

**Dernière mise à jour:** 2025-11-11

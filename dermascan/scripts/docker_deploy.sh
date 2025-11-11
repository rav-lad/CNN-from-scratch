#!/bin/bash
# Deploy DermaScan to production

set -e

echo "🚀 Deploying DermaScan to Production"
echo "====================================="
echo ""

# Configuration
REGISTRY=${DOCKER_REGISTRY:-"docker.io"}
IMAGE_NAME=${IMAGE_NAME:-"dermascan"}
VERSION=${VERSION:-"0.1.0"}

# Full image name
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${VERSION}"
LATEST_IMAGE="${REGISTRY}/${IMAGE_NAME}:latest"

echo "📋 Configuration:"
echo "   Registry: ${REGISTRY}"
echo "   Image: ${IMAGE_NAME}"
echo "   Version: ${VERSION}"
echo ""

# Build
echo "1️⃣  Building image..."
docker build -t ${FULL_IMAGE} -t ${LATEST_IMAGE} .

echo ""
echo "2️⃣  Testing image..."
# Run basic health check
docker run --rm -d --name dermascan-test -p 8001:8000 ${FULL_IMAGE}
sleep 5

if curl -f http://localhost:8001/api/health > /dev/null 2>&1; then
    echo "   ✅ Health check passed"
else
    echo "   ❌ Health check failed"
    docker stop dermascan-test
    exit 1
fi

docker stop dermascan-test
echo ""

# Push to registry (optional)
read -p "3️⃣  Push to registry ${REGISTRY}? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   Pushing ${FULL_IMAGE}..."
    docker push ${FULL_IMAGE}
    echo "   Pushing ${LATEST_IMAGE}..."
    docker push ${LATEST_IMAGE}
    echo "   ✅ Images pushed successfully"
else
    echo "   ⏭️  Skipped pushing to registry"
fi

echo ""
echo "✅ Deployment preparation complete!"
echo ""
echo "To deploy with docker-compose:"
echo "  docker-compose -f docker-compose.yml --profile production up -d"
echo ""
echo "To deploy to cloud (example for AWS ECS/GCP Cloud Run):"
echo "  # AWS ECS"
echo "  aws ecs update-service --cluster my-cluster --service dermascan --force-new-deployment"
echo ""
echo "  # GCP Cloud Run"
echo "  gcloud run deploy dermascan --image ${FULL_IMAGE} --platform managed"
echo ""
echo "  # Azure Container Instances"
echo "  az container create --resource-group myResourceGroup --name dermascan --image ${FULL_IMAGE}"

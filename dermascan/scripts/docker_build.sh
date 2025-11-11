#!/bin/bash
# Build Docker image for DermaScan

set -e

echo "🐳 Building DermaScan Docker Image"
echo "===================================="
echo ""

# Get version from __init__.py or use default
VERSION=${VERSION:-"0.1.0"}

# Build image
echo "📦 Building image: dermascan:${VERSION}"
docker build -t dermascan:${VERSION} -t dermascan:latest .

echo ""
echo "✅ Build complete!"
echo ""
echo "Image tags:"
echo "  - dermascan:${VERSION}"
echo "  - dermascan:latest"
echo ""
echo "Next steps:"
echo "  - Run: docker run -p 8000:8000 dermascan:latest"
echo "  - Or use: docker-compose up"

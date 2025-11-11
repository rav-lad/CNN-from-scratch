#!/bin/bash
# Run DermaScan with Docker Compose

set -e

echo "🚀 Starting DermaScan with Docker"
echo "=================================="
echo ""

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found!"
    echo "Please install: https://docs.docker.com/compose/install/"
    exit 1
fi

# Determine environment
ENV=${1:-"dev"}

if [ "$ENV" = "prod" ] || [ "$ENV" = "production" ]; then
    echo "🏭 Starting in PRODUCTION mode"
    docker-compose -f docker-compose.yml --profile production up -d
    echo ""
    echo "✅ DermaScan is running!"
    echo "   - API: http://localhost:8000"
    echo "   - Web: http://localhost (nginx)"
    echo ""
    echo "To stop: docker-compose down"
    echo "To view logs: docker-compose logs -f"
elif [ "$ENV" = "dev" ] || [ "$ENV" = "development" ]; then
    echo "🔧 Starting in DEVELOPMENT mode (with hot-reload)"
    docker-compose -f docker-compose.dev.yml up
else
    echo "🌐 Starting in DEFAULT mode"
    docker-compose up -d
    echo ""
    echo "✅ DermaScan is running!"
    echo "   - API & Web: http://localhost:8000"
    echo ""
    echo "To stop: docker-compose down"
    echo "To view logs: docker-compose logs -f dermascan-api"
fi

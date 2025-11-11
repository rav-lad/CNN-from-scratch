#!/bin/bash
# Start DermaScan API Server

echo "🚀 Starting DermaScan Server..."
echo "================================"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "⚠️  Virtual environment not found!"
    echo "Please run: python -m venv .venv && source .venv/bin/activate && pip install -e ."
    exit 1
fi

# Activate virtual environment
if [ -d ".venv/bin" ]; then
    source .venv/bin/activate
elif [ -d ".venv/Scripts" ]; then
    source .venv/Scripts/activate
fi

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing DermaScan dependencies..."
    pip install -r dermascan/requirements.txt
fi

# Start server
echo "🌐 Server starting at http://localhost:8000"
echo "📱 Open your browser and navigate to http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")/../.." || exit
python -m uvicorn dermascan.api.app:app --reload --host 0.0.0.0 --port 8000

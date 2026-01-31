#!/bin/bash

# Truth-Lens Quick Start Script

echo "=========================================="
echo "  Truth-Lens - Starting Application"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if model exists
if [ ! -f "data/models/truth_lens_model.h5" ] && [ ! -f "data/models/best_model.h5" ]; then
    echo "⚠️  No trained model found!"
    echo ""
    echo "Options:"
    echo "  1. Train a new model: python src/train.py"
    echo "  2. Download pre-trained model (if available)"
    echo ""
    read -p "Train now? (y/n): " train_choice
    
    if [ "$train_choice" = "y" ]; then
        echo "Starting training..."
        python src/train.py
    else
        echo "Cannot start without model. Exiting."
        exit 1
    fi
fi

# Start the API server in the background
echo ""
echo "Starting API server on http://localhost:8000..."
cd src/api
python app.py &
API_PID=$!
cd ../..

# Wait for API to start
echo "Waiting for API to initialize..."
sleep 5

# Check if API is running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ API server is running"
else
    echo "❌ API server failed to start"
    kill $API_PID 2>/dev/null
    exit 1
fi

# Start frontend server
echo ""
echo "Starting frontend on http://localhost:3000..."
cd frontend
python3 -m http.server 3000 &
FRONTEND_PID=$!
cd ..

echo ""
echo "=========================================="
echo "  ✓ Truth-Lens is running!"
echo "=========================================="
echo ""
echo "  Frontend:  http://localhost:3000"
echo "  API:       http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Wait for user interrupt
trap "echo ''; echo 'Stopping servers...'; kill $API_PID $FRONTEND_PID 2>/dev/null; echo 'Done!'; exit 0" INT

# Keep script running
wait

#!/bin/bash
echo "Starting OpenRouter Document Chatbot..."
echo "========================================"

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Virtual environment not found!"
    echo "Please run: ./install.sh"
    exit 1
fi

source venv/bin/activate

echo "Starting Streamlit server..."
echo "The app will open in your browser automatically"
echo "If it doesn't open, go to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py
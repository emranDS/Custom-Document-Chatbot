#!/bin/bash
echo "Installing Final OpenRouter Chatbot..."
echo "========================================"

cd "$(dirname "$0")"

echo "Cleaning old environment..."
rm -rf venv __pycache__ */__pycache__ vector_data

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing packages..."
pip install streamlit==1.28.1
pip install streamlit-chat==0.1.1
pip install python-dotenv==1.0.0
pip install PyPDF2==3.0.1
pip install python-docx==1.1.0
pip install numpy==1.24.3
pip install tiktoken==0.5.1
pip install requests==2.31.0

echo "Creating .env file..."
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# OpenRouter API Key - Get from https://openrouter.ai
OPENROUTER_API_KEY=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

# OpenRouter Model (free model)
OPENROUTER_MODEL=openai/gpt-oss-120b:free

# Optional: Set to true for detailed logging
DEBUG=false
EOF
    echo "Created .env file. Please edit with your OpenRouter API key."
fi

echo ""
echo "Installation complete!"
echo "========================================"
echo "To run: source venv/bin/activate && streamlit run app.py"
#!/bin/bash

# Setup script for TTT-Agent

echo "🚀 Setting up TTT-Agent environment..."

# Check if Python 3.8+ is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
required_version="3.8"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "❌ Python $python_version is installed, but Python 3.8 or higher is required."
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "⚙️ Creating .env file from template..."
    cp .env.template .env
    echo "📝 Please edit .env file and add your API keys"
fi

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/logs
mkdir -p data/chroma_db

# Make scripts executable
chmod +x scripts/*.sh

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your Gemini API key"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Run the agent: python src/main.py"
echo ""
echo "Happy coding! 🎉"

#!/bin/bash

# Run script for TTT-Agent

echo "🚀 Starting TTT-Agent..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./scripts/setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Please copy .env.template to .env and configure it."
    exit 1
fi

# Run the agent
python src/main.py

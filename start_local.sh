#!/bin/bash
echo "Installing compatible version of MCP for social media search..."
./venv/bin/pip install "mcp<2.0.0"

echo "Starting Streamlit App locally..."
./venv/bin/streamlit run summary.ai/main.py

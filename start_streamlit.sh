#!/bin/bash
# Start the Streamlit Dashboard
echo "Starting Streamlit App..."
if [ -d "./venv" ]; then
    ./venv/bin/streamlit run summary.ai/streamlit_app.py
else
    streamlit run summary.ai/streamlit_app.py
fi

#!/bin/bash

# Ensure dependencies are installed
pip install -r requirements.txt

# Run the FastAPI app (which now includes the frontend)
# --host 0.0.0.0 makes it accessible externally
# --port 8000 is the standard port
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

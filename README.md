# baby-cry-classification

## Description
Web application for automatic classification of infant cry using machine learning.

## Features
- Upload audio file
- Extract audio features (MFCC, etc.)
- Classify cry type (hungry, pain, tired...)

## Tech stack
- Python (FastAPI)
- Machine Learning (SVM)
- HTML/CSS/JS

## How to run
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload

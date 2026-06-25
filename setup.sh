#!/bin/bash
apt-get update
apt-get install -y tesseract-ocr tesseract-ocr-urd poppler-utils
pip install -r requirements.txt
python app.py 

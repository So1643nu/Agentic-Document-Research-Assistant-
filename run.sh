#!/usr/bin/env bash
set -e

if [ ! -f .env ]; then
  echo "No .env found. Copying .env.example -> .env"
  echo "Remember to add your ANTHROPIC_API_KEY before chatting."
  cp .env.example .env
fi

streamlit run app.py

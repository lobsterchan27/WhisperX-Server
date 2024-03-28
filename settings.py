import os

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_DIR = os.getenv("MODEL_DIRECTORY")
OUTPUT_DIR = os.getenv("OUTPUT_DIRECTORY")
VOICES_DIRECTORY = os.getenv("VOICES_DIRECTORY")
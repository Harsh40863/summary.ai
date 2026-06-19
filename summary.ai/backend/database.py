import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

env_path = Path(__file__).resolve().parent / ".env"
root_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
load_dotenv(dotenv_path=root_env_path, override=False)

mongodb_uri = os.getenv("MONGODB_URI")
if not mongodb_uri:
    raise RuntimeError("MONGODB_URI is required")

client = MongoClient(mongodb_uri)
db = client["hackindia_auth"]
users = db["users"]

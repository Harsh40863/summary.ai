import os
from pathlib import Path
from pymongo import MongoClient
from dotenv import load_dotenv

backend_dir = Path(__file__).resolve().parent
root_dir = backend_dir.parent

load_dotenv(root_dir / ".env", override=True)
load_dotenv(backend_dir / ".env", override=True)
load_dotenv(root_dir / "think" / ".env", override=False)

MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI or "your_mongodb_connection_string_here" in MONGODB_URI:
    print("WARNING: MONGODB_URI is not configured in .env file.")

client = MongoClient(MONGODB_URI)

db = client.hackindia
users_collection = db.users
sessions_collection = db.sessions

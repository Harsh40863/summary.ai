from fastapi import APIRouter, HTTPException, Depends, status
from datetime import datetime
from models import UserSignup, UserLogin, TokenResponse
from auth import hash_password, verify_password, create_access_token, get_current_user
from database import users_collection

router = APIRouter()

@router.post("/signup")
def signup(user: UserSignup):
    if users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = hash_password(user.password)
    user_doc = {
        "name": user.name,
        "email": user.email,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow()
    }
    users_collection.insert_one(user_doc)
    return {"message": "Account created successfully"}

@router.post("/login", response_model=TokenResponse)
def login(user: UserLogin):
    user_doc = users_collection.find_one({"email": user.email})
    if not user_doc:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not verify_password(user.password, user_doc["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": user.email})
    return TokenResponse(access_token=access_token, token_type="bearer")

@router.get("/me")
def read_users_me(current_email: str = Depends(get_current_user)):
    user_doc = users_collection.find_one({"email": current_email})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")
    return {"name": user_doc["name"], "email": user_doc["email"]}

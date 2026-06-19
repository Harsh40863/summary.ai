from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status

from backend.auth import create_access_token, hash_password, verify_password
from backend.database import users
from backend.models import TokenResponse, UserLogin, UserSignup

router = APIRouter()


@router.post("/signup")
def signup(payload: UserSignup):
    existing_user = users.find_one({"email": payload.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    users.insert_one(
        {
            "name": payload.name,
            "email": payload.email,
            "hashed_password": hash_password(payload.password),
            "created_at": datetime.now(timezone.utc),
        }
    )

    return {"message": "Account created successfully"}


@router.post("/login", response_model=TokenResponse)
def login(payload: UserLogin):
    user = users.find_one({"email": payload.email})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    if not verify_password(payload.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    token = create_access_token({"sub": payload.email})
    return TokenResponse(access_token=token)

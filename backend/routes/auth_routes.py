from fastapi import APIRouter, HTTPException, Depends, status, Response, Request
from datetime import datetime, timedelta
from models import UserSignup, UserLogin, TokenResponse
from auth import hash_password, verify_password, create_access_token, get_current_user
from database import users_collection, blacklist_collection

router = APIRouter()

@router.post("/signup")
def signup(user: UserSignup, response: Response):
    try:
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
        
        # Automatically log the user in on signup with a 3-day token
        expire_delta = timedelta(days=3)
        access_token = create_access_token(data={"sub": user.email}, expires_delta=expire_delta)
        
        # Set HttpOnly cookie
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            max_age=259200, # 3 days in seconds
            samesite="lax",
            secure=False # set to True in production if using HTTPS
        )
        
        return {
            "message": "Account created successfully",
            "access_token": access_token,
            "token_type": "bearer"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

@router.post("/login", response_model=TokenResponse)
def login(user: UserLogin, response: Response):
    try:
        user_doc = users_collection.find_one({"email": user.email})
        if not user_doc:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        if not verify_password(user.password, user_doc["hashed_password"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Generate a 3-day token
        expire_delta = timedelta(days=3)
        access_token = create_access_token(data={"sub": user.email}, expires_delta=expire_delta)
        
        # Set HttpOnly cookie
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            max_age=259200, # 3 days in seconds
            samesite="lax",
            secure=False
        )
        
        return TokenResponse(access_token=access_token, token_type="bearer")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

@router.post("/logout")
def logout(request: Request, response: Response):
    try:
        # Extract token from cookie first, fallback to Auth header
        token = request.cookies.get("access_token")
        if not token:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                
        if token:
            # Blacklist token for 3 days
            expire_at = datetime.utcnow() + timedelta(days=3)
            blacklist_collection.insert_one({
                "token": token,
                "blacklisted_at": datetime.utcnow(),
                "expire_at": expire_at
            })
            
        # Delete access_token cookie
        response.delete_cookie(
            key="access_token",
            httponly=True,
            samesite="lax",
            secure=False
        )
        return {"message": "Logged out successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logout failed: {str(e)}")


@router.get("/me")
def read_users_me(current_email: str = Depends(get_current_user)):
    try:
        user_doc = users_collection.find_one({"email": current_email})
        if not user_doc:
            raise HTTPException(status_code=404, detail="User not found")
        return {"name": user_doc.get("name", ""), "email": user_doc.get("email", "")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

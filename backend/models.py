from pydantic import BaseModel

class UserSignup(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

class GenerateRequest(BaseModel):
    prompt: str

class GenerateResponse(BaseModel):
    result: str

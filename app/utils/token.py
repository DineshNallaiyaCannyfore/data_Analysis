import jwt
from datetime import datetime, timedelta, UTC
from typing import Dict

SECRET_KEY = "Din@#!*secret123"
ALGORITHM = "HS256"

def create_token(data: Dict, expires_in: int = 3600) -> str:
    to_encode = data.copy()
    expire = datetime.now(UTC) + timedelta(seconds=expires_in) 
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return token

def verify_token(token: str) -> Dict:
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return decoded
    except jwt.ExpiredSignatureError:
        return {"error": "Token has expired"}
    except jwt.InvalidTokenError:
        return {"error": "Invalid token"}

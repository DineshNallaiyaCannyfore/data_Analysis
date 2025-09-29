from fastapi import Depends, HTTPException, Header, status
from app.utils.token import verify_token
from typing import Optional

def raise_auth_error(code: str, message: str):
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"status_code": code, "message": message, "status": "error","code":401},
        headers={"WWW-Authenticate": "Bearer"},
    )

async def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise_auth_error("TOKEN_MISSING", "Authentication token is missing")

    if not authorization.lower().startswith("bearer "):
        raise_auth_error("INVALID_HEADER", "Authorization header must start with Bearer")

    token = authorization.split(" ")[1]
    try:
        payload = verify_token(token)
        user_id = payload.get("id") or payload.get("sub")
        if not user_id:
            raise_auth_error("INVALID_TOKEN", "Token payload is invalid")
        return {"id": user_id}
    except Exception:
        raise_auth_error("INVALID_TOKEN", "Token is invalid or expired")

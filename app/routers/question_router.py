from typing import List
from fastapi import APIRouter, Depends, File, Form, Query, Response,HTTPException, UploadFile
from fastapi.responses import JSONResponse
from app.service.question_service import handleFileUpload
from fastapi import status
from app.utils.security import get_current_user
query_router = APIRouter()

@query_router.post("/getData")
async def fileUpload(
    files: List[UploadFile] = File(...),
    ):
    try:
    
        result = await handleFileUpload(files)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
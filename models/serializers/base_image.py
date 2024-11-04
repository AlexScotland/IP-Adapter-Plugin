from pydantic import BaseModel, Field

from settings import SELECTED_MODEL

class BaseImage(BaseModel):
    prompt: str
    height: int = 600
    width: int = 600
    model: str = SELECTED_MODEL
    face_analysis_model: str = "buffalo_s"

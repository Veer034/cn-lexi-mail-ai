
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
import datetime

class Advice(BaseModel):
    advice: str
    url: str

class Complaint(BaseModel):
    complaint: str
    advises: List[Advice]

class EmailClassificationDto(BaseModel):
    tenantId: str
    threadId: str
    type: str
    subType: str
    queryResponse: Optional[str] = None
    complaints: Optional[List[Complaint]] = None
    suggestions: Optional[List[str]] = None
    eventTime: int = Field(default_factory=lambda: int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000))

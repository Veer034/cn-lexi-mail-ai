
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
import datetime

class Advice(BaseModel):
    query: str
    advice: str
    url: str

class Complaint(BaseModel):
    complaints: List[str]
    advises: List[Advice]

class EmailClassificationDto(BaseModel):
    tenantId: str
    threadId: str
    messageId: str
    type: str
    subType: str
    queryResponse: Optional[str] = None
    complaint: Optional[Complaint] = None
    suggestions: Optional[List[str]] = None
    eventTime: int = Field(default_factory=lambda: int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000))

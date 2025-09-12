
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

# Ticket Data Model
class TicketData(BaseModel):
    title: str
    description: str
    priority: str


class EmailClassificationDto(BaseModel):
    """
    Complete DTO for email classification results to be published on Kafka topic
    Includes all variables for query, complaint, and suggestion processing
    """
    
    # Core identification fields
    tenantId: str
    threadId: str
    senderName: str
    messageId: str
    type: str
    subType: str
    
    # Query processing fields
    queryResponse: Optional[str] = None
    
    # Complaint processing fields
    complaint: Optional[Complaint] = None
    complaintResponse: Optional[str] = None
    ticketData: Optional[TicketData] = None 
    
    # Suggestion processing fields
    suggestions: Optional[List[str]] = None
    suggestionResponse: Optional[str] = None
    
    
    # Timestamp
    eventTime: int = Field(default_factory=lambda: int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000))
    
    
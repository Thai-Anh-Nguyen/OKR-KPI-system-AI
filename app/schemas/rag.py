from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class AlertRequest(BaseModel):
    user_id: str = Field(..., description="Employee UUID")
    risk_score: float = Field(..., description="Risk score triggering the alert")
    risk_label: str = Field(..., description="Risk classification label")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for LLM")


class AlertContent(BaseModel):
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Alert description")
    recommendations: List[str] = Field(default_factory=list, description="Action recommendations")
    priority: str = Field(..., description="Priority level: low/medium/high")


class AlertResponse(BaseModel):
    user_id: str
    alert_id: Optional[str] = None
    content: AlertContent
    generated_at: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "alert_id": "alert-123",
                "content": {
                    "title": "Performance Review Needed",
                    "description": "Employee showing signs of performance decline",
                    "recommendations": [
                        "Schedule 1-on-1 meeting",
                        "Review KPI targets",
                        "Provide support resources"
                    ],
                    "priority": "high"
                },
                "generated_at": "2024-04-27T10:30:00Z"
            }
        }


class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: Optional[str] = None

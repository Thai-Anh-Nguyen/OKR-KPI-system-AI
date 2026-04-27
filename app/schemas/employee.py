from pydantic import BaseModel, Field
from typing import Optional, Literal
from uuid import UUID


class EmployeeBase(BaseModel):
    user_id: str = Field(..., description="Employee UUID")
    full_name: Optional[str] = None
    unit_id: Optional[str] = None
    role: Optional[str] = None


class RiskFeatures(BaseModel):
    kpi_completion_rate: float = Field(..., description="KPI completion percentage (0.0-1.0)")
    checkin_delay_days: float = Field(..., description="Average days between check-ins")
    feedback_sentiment_score: float = Field(..., description="Average feedback sentiment (-1.0 to 1.0)")
    objective_participation_ratio: float = Field(..., description="Ratio of objectives participated")
    
    class Config:
        json_schema_extra = {
            "example": {
                "kpi_completion_rate": 0.85,
                "checkin_delay_days": 5.2,
                "feedback_sentiment_score": 0.3,
                "objective_participation_ratio": 1.2
            }
        }


RiskLabel = Literal["low", "medium", "high"]


class RiskAnalysisRequest(BaseModel):
    user_id: str = Field(..., description="Employee UUID")
    features: RiskFeatures = Field(..., description="Risk feature vectors")


class RiskAnalysisResponse(BaseModel):
    user_id: str
    risk_label: RiskLabel
    risk_score: float = Field(..., description="Risk score (0.0-1.0)")
    features: RiskFeatures
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "risk_label": "medium",
                "risk_score": 0.5,
                "features": {
                    "kpi_completion_rate": 0.85,
                    "checkin_delay_days": 5.2,
                    "feedback_sentiment_score": 0.3,
                    "objective_participation_ratio": 1.2
                }
            }
        }


class EmployeeRiskProfile(EmployeeBase):
    risk_label: Optional[RiskLabel] = None
    risk_score: Optional[float] = None
    features: Optional[RiskFeatures] = None

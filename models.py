from typing import Optional
from pydantic import BaseModel, Field, model_validator


class BoundingBox(BaseModel):
    ymin: float = Field(..., ge=0.0, le=1.0, description="Normalized top coordinate [0, 1]")
    xmin: float = Field(..., ge=0.0, le=1.0, description="Normalized left coordinate [0, 1]")
    ymax: float = Field(..., ge=0.0, le=1.0, description="Normalized bottom coordinate [0, 1]")
    xmax: float = Field(..., ge=0.0, le=1.0, description="Normalized right coordinate [0, 1]")

    @model_validator(mode="after")
    def validate_bounds(self) -> "BoundingBox":
        if self.ymin > self.ymax:
            raise ValueError(f"ymin ({self.ymin}) cannot be greater than ymax ({self.ymax})")
        if self.xmin > self.xmax:
            raise ValueError(f"xmin ({self.xmin}) cannot be greater than xmax ({self.xmax})")
        return self

    @property
    def area(self) -> float:
        return (self.ymax - self.ymin) * (self.xmax - self.xmin)


class GroundTruthItem(BaseModel):
    id: str
    image_id: str
    prompt: str = ""
    label: str
    bbox: BoundingBox


class PredictionItem(BaseModel):
    id: str
    image_id: str
    label: str
    bbox: BoundingBox
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class EvalResult(BaseModel):
    item_id: str
    image_id: str
    label_match: bool
    iou: float
    is_correct: bool
    status: str  # "CORRECT", "LOCALIZATION_ERROR", "CLASSIFICATION_ERROR", "MISSING_PREDICTION"


class EvaluationSummary(BaseModel):
    total_samples: int
    correct_samples: int
    accuracy_at_threshold: float
    mean_iou: float
    iou_threshold: float
    results: list[EvalResult]

from pydantic import BaseModel, Field, field_validator
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml

class DataConfig(BaseModel):
    raw_wsi_path: Path
    annotations_path: Path
    processed_tiles_path: Path
    tile_size: int = 512
    target_magnification: float = 40.0
    overlap_percent: float = 20.0
    min_tissue_percent: float = 0.5
    
    @field_validator('tile_size')
    def tile_size_multiple(cls, v):
        if v % 32 != 0:
            raise ValueError('Tile size must be multiple of 32 for YOLO')
        return v

class ModelConfig(BaseModel):
    architecture: str = "yolov8m"
    num_classes: int = 5
    pretrained: bool = True
    custom_anchors: Dict[str, List[List[int]]]
    input_channels: int = 3  # or 6 for RGB+HSV

class TrainConfig(BaseModel):
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    warmup_epochs: int = 5
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: float = 0.25
    early_stopping_patience: int = 15
    
class InferenceConfig(BaseModel):
    confidence_threshold: float = 0.25
    nms_iou_threshold: float = 0.5
    min_aspect_ratio: float = 2.0
    max_aspect_ratio: float = 10.0

class Config(BaseModel):
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    inference: InferenceConfig
    
    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load and validate configuration from YAML file safely."""
        with open(path, 'r') as f:
            # VULNERABILITY REMEDIATION: Using safe_load instead of load
            yaml_dict = yaml.safe_load(f)
        return cls(**yaml_dict)

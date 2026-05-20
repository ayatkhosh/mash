"""
UrbanInsight - حزمة المرافق
============================
تحتوي على جميع الوحدات المساعدة لمشروع UrbanInsight لكشف وتقسيم المباني
من صور الأقمار الصناعية.

UrbanInsight Utilities Package
================================
Contains all helper modules for the UrbanInsight project for building
detection and segmentation from satellite imagery.

المكونات / Components:
    - data_loader: تحميل ومعالجة الصور / Image loading and preprocessing
    - dino_detector: كشف المباني بنموذج DINO / Building detection with DINO
    - sam_segmenter: تقسيم الصور بنموذج SAM / Image segmentation with SAM
    - post_processor: معالجة ما بعد التقسيم / Post-processing pipeline
    - building_id_generator: توليد معرفات المباني / Building ID generation
"""

from .data_loader import ImageLoader
from .dino_detector import DINODetector
from .sam_segmenter import SAMSegmenter
from .post_processor import PostProcessor
from .building_id_generator import BuildingIDGenerator

__all__ = [
    "ImageLoader",
    "DINODetector",
    "SAMSegmenter",
    "PostProcessor",
    "BuildingIDGenerator",
]

__version__ = "1.0.0"
__author__ = "UrbanInsight Team"

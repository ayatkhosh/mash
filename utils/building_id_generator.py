"""
utils/building_id_generator.py
================================
وحدة توليد وإدارة معرفات المباني لمشروع UrbanInsight

Building ID Generation and Management Module for UrbanInsight Project

الوظائف الرئيسية / Main Functions:
    - توليد معرف فريد لكل مبنى / Generate unique ID per building
    - حفظ معلومات المباني / Save building information
    - تحديد المركز والمساحة والخصائص / Determine center, area, and properties
    - تصدير النتائج بصيغة JSON / Export results in JSON format

المتطلبات / Requirements:
    pip install numpy
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BuildingIDGenerator:
    """
    مولّد وإدارة معرفات المباني لمشروع UrbanInsight
    Building ID Generator and Manager for UrbanInsight

    يقوم بتوليد معرّفات فريدة لكل مبنى مكتشف وحفظ خصائصه الكاملة.
    Generates unique identifiers for each detected building and saves
    its complete properties.

    المثال / Example:
        >>> generator = BuildingIDGenerator(prefix="BLD", project_id="P001")
        >>> building_id = generator.generate(mask, score=0.92, image_path="img.tif")
        >>> generator.save_to_json("/output/buildings.json")
    """

    def __init__(
        self,
        prefix: str = "BLD",
        project_id: str = "UrbanInsight",
        use_uuid: bool = True,
    ):
        """
        تهيئة مولّد المعرفات
        Initialize ID generator

        المعاملات / Parameters:
            prefix (str):
                بادئة المعرف (مثال: 'BLD' → 'BLD-001')
                ID prefix (e.g., 'BLD' → 'BLD-001'). Default: 'BLD'
            project_id (str):
                معرف المشروع لتضمينه في البيانات
                Project ID to embed in the data. Default: 'UrbanInsight'
            use_uuid (bool):
                استخدام UUID للمعرفات الفريدة عالمياً. الافتراضي: True
                Use UUID for globally unique IDs. Default: True
        """
        self.prefix = prefix
        self.project_id = project_id
        self.use_uuid = use_uuid

        self._buildings: List[Dict] = []
        self._counter: int = 0
        self._session_id = str(uuid.uuid4())[:8].upper()

        logger.info(
            f"✅ BuildingIDGenerator: prefix='{prefix}', "
            f"session='{self._session_id}'"
        )

    def generate(
        self,
        mask: np.ndarray,
        score: float = 1.0,
        image_path: Optional[str] = None,
        image_name: Optional[str] = None,
        extra_properties: Optional[Dict] = None,
    ) -> str:
        """
        توليد معرف فريد لمبنى وحفظ خصائصه
        Generate a unique ID for a building and save its properties

        المعاملات / Parameters:
            mask (np.ndarray):
                القناع الثنائي للمبنى (H, W)
                Binary mask of the building (H, W)
            score (float):
                درجة ثقة الكشف (0-1). الافتراضي: 1.0
                Detection confidence score (0-1). Default: 1.0
            image_path (str, optional):
                المسار الكامل لصورة المصدر
                Full path to source image
            image_name (str, optional):
                اسم الصورة المصدر (إذا لم يُحدَّد image_path)
                Source image name (if image_path not provided)
            extra_properties (dict, optional):
                خصائص إضافية مخصصة
                Additional custom properties

        المُخرجات / Returns:
            str: المعرف الفريد للمبنى / Unique building ID

        المثال / Example:
            >>> bid = gen.generate(mask, score=0.91, image_name="paris_001.tif")
            >>> print(bid)  # 'BLD-A3F2-001'
        """
        self._counter += 1

        # توليد المعرف / Generate ID
        building_id = self._create_id()

        # حساب الخصائص / Compute properties
        props = self._compute_properties(mask)

        # معلومات المصدر / Source info
        src_name = (
            Path(image_path).name if image_path else (image_name or "unknown")
        )

        # بناء سجل المبنى / Build building record
        record = {
            "id": building_id,
            "sequential_number": self._counter,
            "project_id": self.project_id,
            "session_id": self._session_id,
            "source_image": src_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "confidence_score": round(float(score), 4),
            "geometry": {
                "area_px": props["area_px"],
                "centroid_x": props["centroid_x"],
                "centroid_y": props["centroid_y"],
                "bbox_x1": props["bbox"][0],
                "bbox_y1": props["bbox"][1],
                "bbox_x2": props["bbox"][2],
                "bbox_y2": props["bbox"][3],
                "bbox_width": props["bbox"][2] - props["bbox"][0],
                "bbox_height": props["bbox"][3] - props["bbox"][1],
                "aspect_ratio": props["aspect_ratio"],
                "compactness": props["compactness"],
            },
        }

        # إضافة خصائص اضافية / Add extra properties
        if extra_properties:
            record["extra"] = extra_properties

        self._buildings.append(record)
        logger.debug(f"🏢 مبنى جديد: {building_id} | مساحة: {props['area_px']} px")

        return building_id

    def generate_batch(
        self,
        masks: List[Dict],
        image_path: Optional[str] = None,
    ) -> List[str]:
        """
        توليد معرفات لمجموعة من الأقنعة دفعة واحدة
        Generate IDs for a batch of masks at once

        المعاملات / Parameters:
            masks (List[Dict]): قائمة الأقنعة من PostProcessor / Masks from PostProcessor
            image_path (str, optional): مسار الصورة المصدر / Source image path

        المُخرجات / Returns:
            List[str]: قائمة المعرفات / List of IDs
        """
        ids = []
        for mask_data in masks:
            bid = self.generate(
                mask=mask_data["mask"],
                score=mask_data.get("score", 1.0),
                image_path=image_path,
                extra_properties={
                    "iou_score": mask_data.get("iou_score"),
                    "box": mask_data.get("box", []),
                },
            )
            ids.append(bid)

        logger.info(f"📦 تم توليد {len(ids)} معرف")
        return ids

    def get_all_buildings(self) -> List[Dict]:
        """
        الحصول على قائمة جميع المباني المسجّلة
        Get list of all registered buildings

        المُخرجات / Returns:
            List[Dict]: قائمة سجلات المباني / List of building records
        """
        return self._buildings.copy()

    def get_building_by_id(self, building_id: str) -> Optional[Dict]:
        """
        البحث عن مبنى بمعرّفه
        Find a building by its ID

        المعاملات / Parameters:
            building_id (str): المعرف الفريد / Unique ID

        المُخرجات / Returns:
            dict or None: سجل المبنى أو None / Building record or None
        """
        for building in self._buildings:
            if building["id"] == building_id:
                return building
        return None

    def save_to_json(
        self, output_path: str, indent: int = 2
    ) -> str:
        """
        حفظ جميع بيانات المباني في ملف JSON
        Save all building data to a JSON file

        المعاملات / Parameters:
            output_path (str): مسار ملف الإخراج / Output file path
            indent (int): مسافة بادئة JSON. الافتراضي: 2 / JSON indent. Default: 2

        المُخرجات / Returns:
            str: مسار الملف المحفوظ / Path to saved file
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": {
                "project_id": self.project_id,
                "session_id": self._session_id,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "total_buildings": len(self._buildings),
                "version": "1.0.0",
            },
            "buildings": self._buildings,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)

        logger.info(
            f"💾 تم حفظ {len(self._buildings)} مبنى في: {output_path}"
        )
        return str(output_path)

    def get_summary(self) -> Dict:
        """
        الحصول على ملخص إحصائي للمباني المسجّلة
        Get statistical summary of registered buildings

        المُخرجات / Returns:
            dict: ملخص الإحصائيات / Summary statistics
        """
        if not self._buildings:
            return {"total": 0}

        areas = [b["geometry"]["area_px"] for b in self._buildings]
        scores = [b["confidence_score"] for b in self._buildings]
        images = list({b["source_image"] for b in self._buildings})

        return {
            "total_buildings": len(self._buildings),
            "processed_images": len(images),
            "image_list": images,
            "area_stats": {
                "total_px": int(sum(areas)),
                "mean_px": float(np.mean(areas)),
                "median_px": float(np.median(areas)),
                "min_px": int(min(areas)),
                "max_px": int(max(areas)),
                "std_px": float(np.std(areas)),
            },
            "confidence_stats": {
                "mean": float(np.mean(scores)),
                "min": float(min(scores)),
                "max": float(max(scores)),
            },
            "session_id": self._session_id,
            "project_id": self.project_id,
        }

    def reset(self):
        """
        إعادة تعيين جميع البيانات المحفوظة
        Reset all stored data
        """
        self._buildings = []
        self._counter = 0
        logger.info("🔄 تم إعادة تعيين BuildingIDGenerator")

    # ─── دوال مساعدة خاصة / Private helper methods ───────────────────

    def _create_id(self) -> str:
        """
        إنشاء معرف فريد
        Create a unique ID
        """
        if self.use_uuid:
            unique_part = str(uuid.uuid4())[:8].upper()
            return f"{self.prefix}-{unique_part}-{self._counter:04d}"
        else:
            return f"{self.prefix}-{self._session_id}-{self._counter:04d}"

    @staticmethod
    def _compute_properties(mask: np.ndarray) -> Dict:
        """
        حساب خصائص المبنى من القناع
        Compute building properties from mask
        """
        import cv2

        mask_uint8 = mask.astype(np.uint8)
        area_px = int(mask.sum())

        # المركز / Centroid
        moments = cv2.moments(mask_uint8)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            h, w = mask.shape
            cx, cy = w / 2.0, h / 2.0

        # الصندوق المحيط / Bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if rows.any() and cols.any():
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
        else:
            x1, y1, x2, y2 = 0, 0, 0, 0

        # نسبة العرض للارتفاع / Aspect ratio
        bbox_w = max(x2 - x1, 1)
        bbox_h = max(y2 - y1, 1)
        aspect_ratio = round(float(bbox_w / bbox_h), 3)

        # الإحكام (مدى دائرية الشكل) / Compactness (circularity measure)
        contours = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        perimeter = cv2.arcLength(
            contours[0] if contours else np.array([[[0, 0]]]),
            True,
        )
        compactness = (
            round(float(4 * np.pi * area_px / (perimeter ** 2)), 4)
            if perimeter > 0
            else 0.0
        )

        return {
            "area_px": area_px,
            "centroid_x": round(float(cx), 2),
            "centroid_y": round(float(cy), 2),
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "aspect_ratio": aspect_ratio,
            "compactness": compactness,
        }

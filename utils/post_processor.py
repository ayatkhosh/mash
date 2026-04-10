"""
utils/post_processor.py
=======================
وحدة معالجة ما بعد التقسيم لمشروع UrbanInsight

Post-Processing Module for UrbanInsight Project

الوظائف الرئيسية / Main Functions:
    - تنظيف الأقنعة والنتائج / Clean masks and results
    - معالجة الأخطاء الصغيرة / Handle small errors/noise
    - دمج الأقنعة المتقطعة / Merge fragmented masks
    - معالجة المكونات المتصلة / Handle connected components

المتطلبات / Requirements:
    pip install opencv-python-headless numpy scipy
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


class PostProcessor:
    """
    مُعالج ما بعد التقسيم لتنظيف وتحسين نتائج SAM+DINO
    Post-processor to clean and improve SAM+DINO results

    يطبّق سلسلة من العمليات لتنظيف الأقنعة وإزالة الضوضاء
    وإصلاح حدود المباني.

    Applies a pipeline of operations to clean masks, remove noise,
    and fix building boundaries.

    المثال / Example:
        >>> processor = PostProcessor(min_area=100, smoothing_kernel=5)
        >>> cleaned = processor.process(masks)
        >>> stats = processor.compute_statistics(cleaned)
    """

    def __init__(
        self,
        min_area: int = 100,
        max_area: Optional[int] = None,
        smoothing_kernel: int = 3,
        closing_iterations: int = 2,
        opening_iterations: int = 1,
        fill_holes: bool = True,
        remove_border_masks: bool = False,
    ):
        """
        تهيئة مُعالج ما بعد التقسيم
        Initialize post-processor

        المعاملات / Parameters:
            min_area (int):
                المساحة الدنيا للقناع بالبكسل (لإزالة الضوضاء الصغيرة)
                Minimum mask area in pixels (to remove small noise). Default: 100
            max_area (int, optional):
                المساحة القصوى للقناع (لإزالة الكشوفات الكبيرة جداً)
                Maximum mask area (to remove overly large detections)
            smoothing_kernel (int):
                حجم نواة التنعيم (يجب أن تكون فردية)
                Smoothing kernel size (must be odd). Default: 3
            closing_iterations (int):
                عدد تكرارات عملية الإغلاق (سد الفجوات)
                Number of morphological closing iterations. Default: 2
            opening_iterations (int):
                عدد تكرارات عملية الفتح (إزالة الضوضاء)
                Number of morphological opening iterations. Default: 1
            fill_holes (bool):
                ملء الثقوب الداخلية في الأقنعة. الافتراضي: True
                Fill internal holes in masks. Default: True
            remove_border_masks (bool):
                إزالة الأقنعة الملامسة لحواف الصورة. الافتراضي: False
                Remove masks touching image borders. Default: False
        """
        self.min_area = min_area
        self.max_area = max_area
        self.smoothing_kernel = smoothing_kernel if smoothing_kernel % 2 == 1 else smoothing_kernel + 1
        self.closing_iterations = closing_iterations
        self.opening_iterations = opening_iterations
        self.fill_holes = fill_holes
        self.remove_border_masks = remove_border_masks

        # نوافذ العمليات المورفولوجية / Morphological operation kernels
        self._kernel_ellipse = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (smoothing_kernel, smoothing_kernel)
        )
        self._kernel_rect = cv2.getStructuringElement(
            cv2.MORPH_RECT, (3, 3)
        )

        logger.info(
            f"✅ PostProcessor: min_area={min_area}, kernel={smoothing_kernel}, "
            f"closing={closing_iterations}, opening={opening_iterations}"
        )

    def process(self, masks: List[Dict]) -> List[Dict]:
        """
        تطبيق سلسلة المعالجة الكاملة على الأقنعة
        Apply the full processing pipeline to masks

        المعاملات / Parameters:
            masks (List[Dict]): قائمة الأقنعة من SAM / List of masks from SAM

        المُخرجات / Returns:
            List[Dict]: الأقنعة المعالجة / Processed masks
        """
        if not masks:
            logger.warning("⚠️ لا توجد أقنعة للمعالجة / No masks to process")
            return []

        processed = []
        removed_count = 0

        for i, mask_data in enumerate(masks):
            try:
                mask = mask_data["mask"].astype(np.uint8)

                # 1. تنظيف مورفولوجي / Morphological cleaning
                mask = self._morphological_clean(mask)

                # 2. ملء الثقوب / Fill holes
                if self.fill_holes:
                    mask = self._fill_internal_holes(mask)

                # 3. تحسين الحواف / Smooth boundaries
                mask = self._smooth_boundaries(mask)

                # 4. تحليل المكونات / Analyze components
                components = self._get_connected_components(mask)

                for comp in components:
                    area = int(comp.sum())

                    # فلترة المساحة / Area filtering
                    if area < self.min_area:
                        removed_count += 1
                        continue
                    if self.max_area and area > self.max_area:
                        removed_count += 1
                        continue

                    # إزالة الأقنعة على الحواف / Remove border masks
                    if self.remove_border_masks and self._touches_border(comp):
                        removed_count += 1
                        continue

                    new_mask = {
                        **mask_data,
                        "mask": comp.astype(bool),
                        "area": area,
                        "bbox": self._get_bounding_box(comp),
                        "centroid": self._get_centroid(comp),
                    }
                    processed.append(new_mask)

            except Exception as e:
                logger.warning(f"⚠️ فشل معالجة القناع {i}: {e}")
                continue

        logger.info(
            f"✅ المعالجة: {len(masks)} → {len(processed)} قناع "
            f"(أُزيل {removed_count})"
        )
        return processed

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        تنظيف قناع واحد
        Clean a single mask

        المعاملات / Parameters:
            mask (np.ndarray): القناع الثنائي / Binary mask

        المُخرجات / Returns:
            np.ndarray: القناع المنظّف / Cleaned mask
        """
        mask_uint8 = mask.astype(np.uint8)
        mask_uint8 = self._morphological_clean(mask_uint8)
        if self.fill_holes:
            mask_uint8 = self._fill_internal_holes(mask_uint8)
        return mask_uint8.astype(bool)

    def compute_statistics(self, masks: List[Dict]) -> Dict:
        """
        حساب إحصائيات الأقنعة المعالجة
        Compute statistics for processed masks

        المعاملات / Parameters:
            masks (List[Dict]): قائمة الأقنعة / List of masks

        المُخرجات / Returns:
            dict: إحصائيات المعالجة / Processing statistics
        """
        if not masks:
            return {
                "total_buildings": 0,
                "total_area_px": 0,
                "avg_area_px": 0,
                "min_area_px": 0,
                "max_area_px": 0,
            }

        areas = [m.get("area", int(m["mask"].sum())) for m in masks]
        scores = [m.get("score", 0) for m in masks]

        return {
            "total_buildings": len(masks),
            "total_area_px": int(sum(areas)),
            "avg_area_px": float(np.mean(areas)),
            "median_area_px": float(np.median(areas)),
            "min_area_px": int(min(areas)),
            "max_area_px": int(max(areas)),
            "std_area_px": float(np.std(areas)),
            "avg_confidence": float(np.mean(scores)) if scores else 0.0,
        }

    def create_segmentation_map(
        self, masks: List[Dict], image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        إنشاء خريطة تقسيم ملوّنة لجميع المباني
        Create a colored segmentation map for all buildings

        المعاملات / Parameters:
            masks (List[Dict]): قائمة الأقنعة / List of masks
            image_shape (tuple): (ارتفاع، عرض) الصورة / (H, W) of image

        المُخرجات / Returns:
            np.ndarray: خريطة ملوّنة (H, W, 3) / Colored map (H, W, 3)
        """
        seg_map = np.zeros((*image_shape, 3), dtype=np.uint8)

        # ألوان ثابتة لكل مبنى / Fixed colors per building
        rng = np.random.default_rng(seed=42)
        colors = rng.integers(50, 255, size=(len(masks), 3))

        for i, mask_data in enumerate(masks):
            mask = mask_data["mask"]
            seg_map[mask] = colors[i]

        return seg_map

    # ─── دوال مساعدة خاصة / Private helper methods ───────────────────

    def _morphological_clean(self, mask: np.ndarray) -> np.ndarray:
        """
        تنظيف مورفولوجي: فتح (إزالة ضوضاء) + إغلاق (سد فجوات)
        Morphological cleaning: opening (noise removal) + closing (gap filling)
        """
        # فتح مورفولوجي لإزالة الضوضاء الصغيرة / Opening to remove small noise
        if self.opening_iterations > 0:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                self._kernel_ellipse,
                iterations=self.opening_iterations,
            )

        # إغلاق مورفولوجي لسد الفجوات الصغيرة / Closing to fill small gaps
        if self.closing_iterations > 0:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                self._kernel_ellipse,
                iterations=self.closing_iterations,
            )

        return mask

    def _fill_internal_holes(self, mask: np.ndarray) -> np.ndarray:
        """
        ملء الثقوب الداخلية في القناع
        Fill internal holes in the mask
        """
        filled = ndimage.binary_fill_holes(mask).astype(np.uint8)
        return filled

    def _smooth_boundaries(self, mask: np.ndarray) -> np.ndarray:
        """
        تنعيم حدود القناع باستخدام Gaussian blur + threshold
        Smooth mask boundaries using Gaussian blur + threshold
        """
        blurred = cv2.GaussianBlur(
            mask.astype(np.float32),
            (self.smoothing_kernel, self.smoothing_kernel),
            0,
        )
        _, smoothed = cv2.threshold(
            (blurred * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY
        )
        return smoothed

    def _get_connected_components(
        self, mask: np.ndarray
    ) -> List[np.ndarray]:
        """
        تحليل المكونات المتصلة وإرجاع كل مكوّن كقناع منفصل
        Analyze connected components and return each as a separate mask
        """
        num_labels, labels = cv2.connectedComponents(mask)
        components = []

        for label_id in range(1, num_labels):  # تجاوز الخلفية (0)
            component = (labels == label_id).astype(np.uint8)
            if component.sum() >= self.min_area:
                components.append(component)

        return components if components else [mask]

    @staticmethod
    def _touches_border(mask: np.ndarray) -> bool:
        """التحقق من ملامسة القناع لحواف الصورة / Check if mask touches borders"""
        return bool(
            mask[0, :].any()
            or mask[-1, :].any()
            or mask[:, 0].any()
            or mask[:, -1].any()
        )

    @staticmethod
    def _get_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
        """
        الحصول على الصندوق المحيط بالقناع
        Get bounding box of mask (x1, y1, x2, y2)
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return int(x1), int(y1), int(x2), int(y2)

    @staticmethod
    def _get_centroid(mask: np.ndarray) -> Tuple[float, float]:
        """
        حساب مركز الثقل للقناع
        Compute centroid of mask (x, y)
        """
        moments = cv2.moments(mask.astype(np.uint8))
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            return float(cx), float(cy)
        h, w = mask.shape
        return float(w / 2), float(h / 2)

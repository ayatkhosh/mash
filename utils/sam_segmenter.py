"""
utils/sam_segmenter.py
======================
وحدة التقسيم الدلالي باستخدام نموذج SAM لمشروع UrbanInsight

Semantic Segmentation Module using SAM for UrbanInsight Project

الوظائف الرئيسية / Main Functions:
    - تحميل نموذج SAM / Load SAM model
    - تحويل كشوفات DINO إلى أقنعة دقيقة / Convert DINO detections to precise masks
    - معالجة المباني المتلاصقة والمتداخلة / Handle adjacent and overlapping buildings

المتطلبات / Requirements:
    pip install git+https://github.com/facebookresearch/segment-anything.git
    تحميل أوزان SAM / Download SAM weights:
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class SAMSegmenter:
    """
    مُقسِّم الصور باستخدام نموذج SAM (Segment Anything Model)
    Image Segmenter using SAM (Segment Anything Model)

    يستقبل صناديق الكشف من DINO ويُنشئ أقنعة دقيقة لكل مبنى.
    Receives detection boxes from DINO and creates precise masks per building.

    المثال / Example:
        >>> segmenter = SAMSegmenter(
        ...     checkpoint_path="/content/weights/sam_vit_h_4b8939.pth",
        ...     model_type="vit_h"
        ... )
        >>> masks = segmenter.segment_from_boxes(image, detections["boxes"])
    """

    # أنواع نماذج SAM المدعومة / Supported SAM model types
    SUPPORTED_MODELS = {
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_b": "sam_vit_b_01ec64.pth",
    }

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "vit_h",
        device: Optional[str] = None,
        multimask_output: bool = False,
    ):
        """
        تهيئة مُقسِّم SAM
        Initialize SAM Segmenter

        المعاملات / Parameters:
            checkpoint_path (str):
                مسار ملف أوزان SAM (.pth)
                Path to SAM weights file (.pth)
            model_type (str):
                نوع النموذج: 'vit_h' (الأقوى), 'vit_l', 'vit_b' (الأخف)
                Model type: 'vit_h' (strongest), 'vit_l', 'vit_b' (lightest)
            device (str, optional):
                الجهاز المستخدم. الافتراضي: تلقائي
                Device to use. Default: auto-detect
            multimask_output (bool):
                إرجاع عدة أقنعة لكل صندوق. الافتراضي: False
                Return multiple masks per box. Default: False
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.model_type = model_type
        self.multimask_output = multimask_output

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"نوع نموذج غير مدعوم / Unsupported model type: {model_type}. "
                f"الأنواع المدعومة / Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.predictor = None
        self.auto_generator = None
        self._load_model()

        logger.info(
            f"✅ SAMSegmenter تم تهيئته: {model_type} على {self.device}"
        )

    def _load_model(self):
        """
        تحميل نموذج SAM
        Load SAM model
        """
        if not self.checkpoint_path.exists():
            download_url = (
                "https://dl.fbaipublicfiles.com/segment_anything/"
                f"{self.SUPPORTED_MODELS[self.model_type]}"
            )
            raise FileNotFoundError(
                f"ملف أوزان SAM غير موجود / SAM weights not found: "
                f"{self.checkpoint_path}\n"
                f"قم بتحميله / Download it:\n"
                f"  wget {download_url}"
            )

        try:
            from segment_anything import sam_model_registry, SamPredictor
            from segment_anything import SamAutomaticMaskGenerator

            sam = sam_model_registry[self.model_type](
                checkpoint=str(self.checkpoint_path)
            )
            sam.to(self.device)
            sam.eval()

            self.predictor = SamPredictor(sam)
            self.auto_generator = SamAutomaticMaskGenerator(
                sam,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
            )

            logger.info(f"✅ تم تحميل نموذج SAM ({self.model_type}) بنجاح")

        except ImportError as e:
            logger.error(
                f"❌ تعذّر استيراد segment_anything: {e}\n"
                "قم بتثبيته / Install it:\n"
                "  pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
            raise

    def segment_from_boxes(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        scores: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        إنشاء أقنعة دقيقة من صناديق الكشف
        Generate precise masks from detection boxes

        المعاملات / Parameters:
            image (np.ndarray):
                مصفوفة الصورة (H, W, 3)
                Image array (H, W, 3)
            boxes (np.ndarray):
                صناديق الكشف بتنسيق xyxy (N, 4)
                Detection boxes in xyxy format (N, 4)
            scores (np.ndarray, optional):
                درجات الثقة لكل صندوق
                Confidence scores per box

        المُخرجات / Returns:
            List[Dict]: قائمة أقنعة لكل مبنى / List of masks per building:
                [
                    {
                        'mask': np.ndarray (H, W) - القناع الثنائي,
                        'box': np.ndarray (4,) - الصندوق,
                        'score': float - درجة الثقة,
                        'area': int - مساحة القناع بالبكسل
                    },
                    ...
                ]
        """
        if self.predictor is None:
            raise RuntimeError("النموذج غير محمّل / Model not loaded")

        if len(boxes) == 0:
            logger.info("⚠️ لا توجد صناديق كشف / No detection boxes provided")
            return []

        # تحويل الصورة للنطاق 0-255 إذا لزم / Convert image to 0-255 if needed
        img_uint8 = self._ensure_uint8(image)

        # تهيئة الصورة في النموذج / Set image in predictor
        self.predictor.set_image(img_uint8)

        results = []
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).to(self.device)

        try:
            # تحويل جميع الصناديق دفعة واحدة / Transform all boxes at once
            transformed_boxes = self.predictor.transform.apply_boxes_torch(
                boxes_tensor, img_uint8.shape[:2]
            )

            # تشغيل SAM / Run SAM
            with torch.no_grad():
                masks_batch, iou_scores, _ = self.predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=self.multimask_output,
                )

            # معالجة كل قناع / Process each mask
            for i, (mask_tensor, iou) in enumerate(
                zip(masks_batch, iou_scores)
            ):
                # اختيار أفضل قناع / Select best mask
                if self.multimask_output:
                    best_idx = iou.argmax().item()
                    mask = mask_tensor[best_idx].cpu().numpy()
                    mask_score = iou[best_idx].item()
                else:
                    mask = mask_tensor[0].cpu().numpy()
                    mask_score = iou[0].item()

                box_score = float(scores[i]) if scores is not None and i < len(scores) else 1.0

                results.append(
                    {
                        "mask": mask.astype(bool),
                        "box": boxes[i],
                        "score": float(box_score * mask_score),
                        "iou_score": float(mask_score),
                        "area": int(mask.sum()),
                    }
                )

            logger.info(
                f"✅ تم إنشاء {len(results)} قناع من {len(boxes)} صندوق"
            )

        except Exception as e:
            logger.error(f"❌ فشل إنشاء الأقنعة: {e}")
            raise

        return results

    def segment_automatic(self, image: np.ndarray) -> List[Dict]:
        """
        تقسيم تلقائي للصورة بدون مدخلات (وضع SAM التلقائي)
        Automatic segmentation without inputs (SAM automatic mode)

        المعاملات / Parameters:
            image (np.ndarray): مصفوفة الصورة / Image array

        المُخرجات / Returns:
            List[Dict]: قائمة الأقنعة المُكتشفة / List of detected masks
        """
        if self.auto_generator is None:
            raise RuntimeError("المولّد التلقائي غير متاح / Auto generator not available")

        img_uint8 = self._ensure_uint8(image)

        try:
            with torch.no_grad():
                masks = self.auto_generator.generate(img_uint8)
            logger.info(f"✅ التقسيم التلقائي: {len(masks)} قناع")
            return masks
        except Exception as e:
            logger.error(f"❌ فشل التقسيم التلقائي: {e}")
            raise

    def merge_overlapping_masks(
        self, masks: List[Dict], iou_threshold: float = 0.5
    ) -> List[Dict]:
        """
        دمج الأقنعة المتداخلة للمباني المتلاصقة
        Merge overlapping masks for adjacent buildings

        المعاملات / Parameters:
            masks (List[Dict]): قائمة الأقنعة / List of masks
            iou_threshold (float): حد التداخل للدمج / IoU threshold for merging

        المُخرجات / Returns:
            List[Dict]: الأقنعة بعد الدمج / Merged masks
        """
        if len(masks) < 2:
            return masks

        merged = []
        used = [False] * len(masks)

        for i in range(len(masks)):
            if used[i]:
                continue

            current = masks[i].copy()
            for j in range(i + 1, len(masks)):
                if used[j]:
                    continue

                iou = self._compute_mask_iou(
                    masks[i]["mask"], masks[j]["mask"]
                )
                if iou > iou_threshold:
                    # دمج القناعين / Merge the two masks
                    current["mask"] = current["mask"] | masks[j]["mask"]
                    current["area"] = int(current["mask"].sum())
                    current["score"] = max(current["score"], masks[j]["score"])
                    used[j] = True

            merged.append(current)
            used[i] = True

        logger.info(
            f"🔗 دمج الأقنعة: {len(masks)} → {len(merged)}"
        )
        return merged

    def split_adjacent_buildings(
        self, mask: np.ndarray, min_area: int = 200
    ) -> List[np.ndarray]:
        """
        فصل المباني المتلاصقة باستخدام تحويل Watershed
        Separate adjacent buildings using Watershed transform

        المعاملات / Parameters:
            mask (np.ndarray): القناع الثنائي / Binary mask
            min_area (int): المساحة الدنيا لكل مبنى بالبكسل / Min area per building

        المُخرجات / Returns:
            List[np.ndarray]: قائمة أقنعة المباني المنفصلة / List of separated masks
        """
        mask_uint8 = mask.astype(np.uint8) * 255

        # إيجاد تحولات المسافة / Compute distance transform
        dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        dist_normalized = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )

        # تعيين النقاط الأصيلة / Find sure foreground regions
        _, sure_fg = cv2.threshold(dist_normalized, 0.5 * dist_normalized.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)

        # العثور على المكونات المتصلة / Find connected components
        _, markers = cv2.connectedComponents(sure_fg)
        markers += 1
        markers[mask_uint8 == 0] = 0

        # تطبيق Watershed / Apply watershed
        img_bgr = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
        markers_ws = cv2.watershed(img_bgr, markers.copy())

        # استخراج أقنعة منفصلة / Extract separate masks
        separated = []
        for label in np.unique(markers_ws):
            if label <= 1:  # تجاوز الخلفية والحدود / Skip background and borders
                continue
            component_mask = (markers_ws == label)
            if component_mask.sum() >= min_area:
                separated.append(component_mask)

        return separated if separated else [mask.astype(bool)]

    # ─── دوال مساعدة خاصة / Private helper methods ───────────────────

    @staticmethod
    def _ensure_uint8(image: np.ndarray) -> np.ndarray:
        """تحويل الصورة لنوع uint8 / Convert image to uint8 type"""
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                return (image * 255).astype(np.uint8)
            return image.astype(np.uint8)
        return image

    @staticmethod
    def _compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        حساب نسبة التقاطع / الاتحاد بين قناعين
        Compute Intersection over Union between two masks
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return float(intersection / union) if union > 0 else 0.0

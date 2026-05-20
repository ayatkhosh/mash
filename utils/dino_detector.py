"""
utils/dino_detector.py
======================
وحدة كشف المباني باستخدام نموذج Grounding DINO لمشروع UrbanInsight

Building Detection Module using Grounding DINO for UrbanInsight Project

الوظائف الرئيسية / Main Functions:
    - تحميل نموذج Grounding DINO / Load Grounding DINO model
    - كشف المباني والنباتات والفراغات / Detect buildings, vegetation, open spaces
    - إدارة النصوص (Prompts) المختلفة / Manage various detection prompts
    - معالجة الأخطاء والاستثناءات / Handle errors and exceptions

المتطلبات / Requirements:
    pip install torch transformers supervision
    git clone https://github.com/IDEA-Research/GroundingDINO
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class DINODetector:
    """
    كاشف المباني باستخدام نموذج Grounding DINO
    Building Detector using Grounding DINO Model

    يستخدم نموذج Grounding DINO للكشف عن المباني والعناصر الحضرية
    في صور الأقمار الصناعية بناءً على نصوص وصفية (text prompts).

    Uses Grounding DINO model to detect buildings and urban elements
    in satellite imagery based on text prompts.

    المثال / Example:
        >>> detector = DINODetector(
        ...     weights_path="/content/weights/gdino.pth",
        ...     config_path="/content/GroundingDINO/groundingdino/config/..."
        ... )
        >>> detections = detector.detect(image, prompt="buildings")
    """

    # النصوص الافتراضية للكشف / Default detection prompts
    DEFAULT_PROMPTS = {
        "buildings": "building . rooftop . house . structure . construction",
        "vegetation": "tree . vegetation . garden . grass . plant . park",
        "open_spaces": "road . parking . empty lot . open area . courtyard",
        "all_urban": (
            "building . rooftop . house . tree . vegetation . road . "
            "parking . open area"
        ),
    }

    def __init__(
        self,
        weights_path: str,
        config_path: str,
        grounding_dino_path: str = "/content/GroundingDINO",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        device: Optional[str] = None,
    ):
        """
        تهيئة كاشف DINO
        Initialize DINO Detector

        المعاملات / Parameters:
            weights_path (str):
                مسار ملف أوزان النموذج (.pth)
                Path to model weights file (.pth)
            config_path (str):
                مسار ملف إعدادات النموذج (.py)
                Path to model config file (.py)
            grounding_dino_path (str):
                مسار مجلد GroundingDINO المثبّت
                Path to installed GroundingDINO directory
            box_threshold (float):
                حد الثقة لصناديق الكشف (0-1). الافتراضي: 0.30
                Confidence threshold for detection boxes (0-1). Default: 0.30
            text_threshold (float):
                حد الثقة للتطابق النصي (0-1). الافتراضي: 0.25
                Confidence threshold for text matching (0-1). Default: 0.25
            device (str, optional):
                الجهاز المستخدم ('cuda' أو 'cpu'). الافتراضي: تلقائي
                Device to use ('cuda' or 'cpu'). Default: auto-detect
        """
        self.weights_path = Path(weights_path)
        self.config_path = Path(config_path)
        self.grounding_dino_path = Path(grounding_dino_path)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # تحديد الجهاز / Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self._gdino_available = False

        # تحميل النموذج / Load model
        self._setup_gdino_path()
        self._load_model()

        logger.info(
            f"✅ DINODetector تم تهيئته على {self.device} | "
            f"box_thresh={box_threshold}, text_thresh={text_threshold}"
        )

    def _setup_gdino_path(self):
        """
        إضافة مسار GroundingDINO لمسارات Python
        Add GroundingDINO path to Python paths
        """
        gdino_str = str(self.grounding_dino_path)
        if gdino_str not in sys.path:
            sys.path.insert(0, gdino_str)
            logger.debug(f"📂 تم إضافة {gdino_str} لـ sys.path")

    def _load_model(self):
        """
        تحميل نموذج Grounding DINO
        Load the Grounding DINO model
        """
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"ملف الأوزان غير موجود / Weights not found: {self.weights_path}\n"
                "قم بتحميله / Download it:\n"
                "wget https://github.com/IDEA-Research/GroundingDINO/releases/"
                "download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
            )

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"ملف الإعدادات غير موجود / Config not found: {self.config_path}"
            )

        try:
            from groundingdino.util.inference import load_model

            self.model = load_model(
                str(self.config_path), str(self.weights_path)
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            self._gdino_available = True
            logger.info("✅ تم تحميل نموذج Grounding DINO بنجاح")
        except ImportError as e:
            logger.error(
                f"❌ تعذّر استيراد GroundingDINO: {e}\n"
                "تأكد من تثبيته / Make sure it is installed:\n"
                "  git clone https://github.com/IDEA-Research/GroundingDINO"
            )
            raise
        except Exception as e:
            logger.error(f"❌ فشل تحميل النموذج: {e}")
            raise

    def detect(
        self,
        image: np.ndarray,
        prompt: str = "buildings",
        custom_prompt: Optional[str] = None,
    ) -> Dict:
        """
        كشف العناصر في الصورة باستخدام نص توصيفي
        Detect elements in image using a text prompt

        المعاملات / Parameters:
            image (np.ndarray):
                مصفوفة الصورة بشكل (H, W, 3)
                Image array with shape (H, W, 3)
            prompt (str):
                اسم النص من القاموس الافتراضي:
                Prompt name from the default dictionary:
                    'buildings', 'vegetation', 'open_spaces', 'all_urban'
            custom_prompt (str, optional):
                نص مخصص للكشف (يتجاوز prompt إذا محدد)
                Custom detection text (overrides prompt if provided)

        المُخرجات / Returns:
            dict: نتائج الكشف / Detection results:
                {
                    'boxes': np.ndarray (N, 4) - صناديق بتنسيق xyxy,
                    'scores': np.ndarray (N,) - درجات الثقة,
                    'labels': list[str] - التسميات المكتشفة,
                    'prompt_used': str - النص المستخدم,
                    'count': int - عدد الكشوفات
                }
        """
        if not self._gdino_available or self.model is None:
            raise RuntimeError("النموذج غير محمّل / Model not loaded")

        # تحديد النص المستخدم / Determine prompt text
        text = custom_prompt if custom_prompt else self.DEFAULT_PROMPTS.get(
            prompt, prompt
        )

        try:
            from groundingdino.util.inference import load_image, predict

            # تحويل الصورة للصيغة المطلوبة / Convert image to required format
            pil_image = Image.fromarray(
                (image * 255).astype(np.uint8) if image.max() <= 1.0 else image
            ).convert("RGB")

            # ─── تشغيل الكشف / Run detection ─────────────────────────
            boxes, logits, phrases = predict(
                model=self.model,
                image=self._preprocess_for_dino(pil_image),
                caption=text,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )

            # تحويل الصناديق للإحداثيات المطلقة / Convert boxes to absolute coords
            h, w = image.shape[:2]
            boxes_xyxy = self._convert_boxes_to_xyxy(boxes, w, h)

            results = {
                "boxes": boxes_xyxy,
                "scores": logits.cpu().numpy() if hasattr(logits, "cpu") else np.array(logits),
                "labels": phrases,
                "prompt_used": text,
                "count": len(boxes_xyxy),
            }

            logger.info(
                f"🔍 تم الكشف عن {results['count']} عنصر "
                f"باستخدام النص: '{prompt}'"
            )
            return results

        except Exception as e:
            logger.error(f"❌ فشل الكشف: {e}")
            raise

    def detect_buildings(self, image: np.ndarray) -> Dict:
        """
        كشف المباني فقط (اختصار لـ detect مع prompt='buildings')
        Detect buildings only (shortcut for detect with prompt='buildings')

        المعاملات / Parameters:
            image (np.ndarray): مصفوفة الصورة / Image array

        المُخرجات / Returns:
            dict: نتائج الكشف / Detection results
        """
        return self.detect(image, prompt="buildings")

    def detect_all(self, image: np.ndarray) -> Dict[str, Dict]:
        """
        كشف جميع العناصر (مباني + نباتات + فراغات)
        Detect all elements (buildings + vegetation + open spaces)

        المعاملات / Parameters:
            image (np.ndarray): مصفوفة الصورة / Image array

        المُخرجات / Returns:
            dict: نتائج الكشف لكل فئة / Detection results per category
        """
        results = {}
        for category in ["buildings", "vegetation", "open_spaces"]:
            try:
                results[category] = self.detect(image, prompt=category)
            except Exception as e:
                logger.warning(f"⚠️ فشل كشف '{category}': {e}")
                results[category] = {
                    "boxes": np.array([]),
                    "scores": np.array([]),
                    "labels": [],
                    "count": 0,
                    "error": str(e),
                }
        return results

    def filter_by_score(
        self, detections: Dict, min_score: float = 0.35
    ) -> Dict:
        """
        تصفية الكشوفات حسب درجة الثقة
        Filter detections by confidence score

        المعاملات / Parameters:
            detections (dict): نتائج الكشف / Detection results
            min_score (float): الحد الأدنى لدرجة الثقة / Min confidence score

        المُخرجات / Returns:
            dict: الكشوفات المصفّاة / Filtered detections
        """
        if len(detections.get("scores", [])) == 0:
            return detections

        mask = detections["scores"] >= min_score
        return {
            "boxes": detections["boxes"][mask],
            "scores": detections["scores"][mask],
            "labels": [
                l for l, m in zip(detections["labels"], mask) if m
            ],
            "prompt_used": detections.get("prompt_used", ""),
            "count": int(mask.sum()),
        }

    def visualize_detections(
        self, image: np.ndarray, detections: Dict
    ) -> np.ndarray:
        """
        رسم صناديق الكشف على الصورة
        Draw detection boxes on image

        المعاملات / Parameters:
            image (np.ndarray): الصورة الأصلية / Original image
            detections (dict): نتائج الكشف / Detection results

        المُخرجات / Returns:
            np.ndarray: الصورة مع الصناديق المرسومة / Image with drawn boxes
        """
        vis = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.copy()
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        for i, (box, score) in enumerate(
            zip(detections.get("boxes", []), detections.get("scores", []))
        ):
            x1, y1, x2, y2 = box.astype(int)
            label = detections["labels"][i] if i < len(detections["labels"]) else ""
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"{label} {score:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    # ─── دوال مساعدة خاصة / Private helper methods ───────────────────

    @staticmethod
    def _preprocess_for_dino(pil_image: "Image.Image") -> "torch.Tensor":
        """
        معالجة الصورة مسبقاً للإدخال لنموذج DINO
        Preprocess image for DINO model input
        """
        import torchvision.transforms as T

        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return transform(pil_image)

    @staticmethod
    def _convert_boxes_to_xyxy(
        boxes: "torch.Tensor", img_w: int, img_h: int
    ) -> np.ndarray:
        """
        تحويل الصناديق من تنسيق cx,cy,w,h (مُطبَّع) إلى x1,y1,x2,y2 (مطلق)
        Convert boxes from normalized cx,cy,w,h to absolute x1,y1,x2,y2

        المعاملات / Parameters:
            boxes: صناديق بتنسيق (N, 4) cx,cy,w,h مُطبَّعة
                   Boxes in format (N, 4) normalized cx,cy,w,h
            img_w: عرض الصورة / Image width
            img_h: ارتفاع الصورة / Image height
        """
        if len(boxes) == 0:
            return np.array([])

        boxes_np = boxes.cpu().numpy() if hasattr(boxes, "cpu") else np.array(boxes)

        cx, cy, bw, bh = (
            boxes_np[:, 0],
            boxes_np[:, 1],
            boxes_np[:, 2],
            boxes_np[:, 3],
        )
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h

        # ضمان القيم ضمن حدود الصورة / Clamp to image bounds
        x1 = np.clip(x1, 0, img_w).astype(int)
        y1 = np.clip(y1, 0, img_h).astype(int)
        x2 = np.clip(x2, 0, img_w).astype(int)
        y2 = np.clip(y2, 0, img_h).astype(int)

        return np.stack([x1, y1, x2, y2], axis=1)

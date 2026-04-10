"""
utils/data_loader.py
====================
وحدة تحميل ومعالجة الصور لمشروع UrbanInsight

Image Loading and Preprocessing Module for UrbanInsight Project

الوظائف الرئيسية / Main Functions:
    - تحميل الصور من مجلد / Load images from a directory
    - معالجة الدقة المختلفة (50-150 بكسل) / Handle varying resolutions (50-150 px)
    - معالجة الصور الكبيرة الحجم / Handle large images
    - دعم صيغ TIF و PNG و JPG / Support TIF, PNG, JPG formats

المتطلبات / Requirements:
    pip install opencv-python-headless Pillow rasterio numpy
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Generator

import cv2
import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.enums import Resampling
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    logging.warning("rasterio غير متوفر - لن يتم دعم ملفات TIF المتعددة النطاقات")

logger = logging.getLogger(__name__)


class ImageLoader:
    """
    محمّل الصور لمشروع UrbanInsight
    UrbanInsight Image Loader

    يدعم تحميل ومعالجة الصور الفضائية بدقات مختلفة وصيغ متعددة.
    Supports loading and preprocessing satellite images with varying
    resolutions and multiple formats.

    المثال / Example:
        >>> loader = ImageLoader(image_dir="/data/images", target_size=(512, 512))
        >>> for img_array, img_path in loader.load_all():
        ...     print(f"Loaded: {img_path}, Shape: {img_array.shape}")
    """

    # الصيغ المدعومة / Supported formats
    SUPPORTED_FORMATS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        image_dir: str,
        target_size: Optional[Tuple[int, int]] = None,
        max_size: int = 2048,
        normalize: bool = True,
    ):
        """
        تهيئة محمّل الصور
        Initialize image loader

        المعاملات / Parameters:
            image_dir (str):
                مسار المجلد الذي يحتوي على الصور
                Path to directory containing images
            target_size (tuple, optional):
                الحجم المستهدف (عرض، ارتفاع) للتغيير الحجم
                Target size (width, height) for resizing. None = no resize
            max_size (int):
                الحجم الأقصى لأي بُعد في الصورة (للصور الكبيرة)
                Maximum size for any dimension (for large images). Default: 2048
            normalize (bool):
                تطبيع قيم البكسل إلى [0, 1]. الافتراضي: True
                Normalize pixel values to [0, 1]. Default: True
        """
        self.image_dir = Path(image_dir)
        self.target_size = target_size
        self.max_size = max_size
        self.normalize = normalize

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"مجلد الصور غير موجود / Image directory not found: {image_dir}"
            )

        logger.info(f"✅ تم تهيئة ImageLoader للمجلد: {image_dir}")

    def get_image_paths(self) -> List[Path]:
        """
        الحصول على قائمة مسارات الصور في المجلد
        Get list of image paths in the directory

        المُخرجات / Returns:
            List[Path]: قائمة مسارات الصور / List of image file paths
        """
        paths = []
        for fmt in self.SUPPORTED_FORMATS:
            paths.extend(self.image_dir.glob(f"*{fmt}"))
            paths.extend(self.image_dir.glob(f"*{fmt.upper()}"))

        # إزالة المكررات وترتيب القائمة
        paths = sorted(set(paths))
        logger.info(f"📁 تم العثور على {len(paths)} صورة في المجلد")
        return paths

    def load_image(self, image_path: str) -> np.ndarray:
        """
        تحميل صورة واحدة وإرجاعها كمصفوفة numpy
        Load a single image and return as numpy array

        المعاملات / Parameters:
            image_path (str): مسار الصورة / Path to the image file

        المُخرجات / Returns:
            np.ndarray: مصفوفة الصورة بشكل (H, W, 3) في نطاق [0,255] أو [0,1]
                        Image array with shape (H, W, 3) in range [0,255] or [0,1]

        الاستثناءات / Raises:
            ValueError: إذا كانت صيغة الملف غير مدعومة
            IOError: إذا فشل تحميل الصورة
        """
        path = Path(image_path)

        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"صيغة غير مدعومة / Unsupported format: {path.suffix}. "
                f"الصيغ المدعومة / Supported: {self.SUPPORTED_FORMATS}"
            )

        # تحميل بناءً على الصيغة / Load based on format
        if path.suffix.lower() in {".tif", ".tiff"} and RASTERIO_AVAILABLE:
            image = self._load_tif(path)
        else:
            image = self._load_standard(path)

        # معالجة الصور الكبيرة / Handle large images
        image = self._handle_large_image(image)

        # تغيير الحجم إذا طُلب / Resize if requested
        if self.target_size is not None:
            image = self._resize_image(image, self.target_size)

        # التطبيع / Normalize
        if self.normalize:
            image = image.astype(np.float32) / 255.0

        logger.debug(f"✅ تم تحميل: {path.name}, الشكل: {image.shape}")
        return image

    def load_all(self) -> Generator[Tuple[np.ndarray, Path], None, None]:
        """
        تحميل جميع الصور من المجلد (مولّد)
        Load all images from directory (generator)

        المُخرجات / Yields:
            Tuple[np.ndarray, Path]: (مصفوفة الصورة، مسار الصورة)
                                     (image array, image path)

        المثال / Example:
            >>> for img, path in loader.load_all():
            ...     process(img)
        """
        paths = self.get_image_paths()
        loaded = 0
        failed = 0

        for path in paths:
            try:
                image = self.load_image(str(path))
                loaded += 1
                yield image, path
            except Exception as e:
                failed += 1
                logger.error(f"❌ فشل تحميل {path.name}: {e}")
                continue

        logger.info(
            f"📊 تم تحميل {loaded} صورة، فشل {failed} صورة"
        )

    def load_batch(
        self, batch_size: int = 4
    ) -> Generator[List[Tuple[np.ndarray, Path]], None, None]:
        """
        تحميل الصور على دفعات لتوفير الذاكرة
        Load images in batches for memory efficiency

        المعاملات / Parameters:
            batch_size (int): حجم الدفعة / Batch size. Default: 4

        المُخرجات / Yields:
            List[Tuple[np.ndarray, Path]]: دفعة من الصور / Batch of images
        """
        batch = []
        for image, path in self.load_all():
            batch.append((image, path))
            if len(batch) >= batch_size:
                yield batch
                batch = []

        # إرجاع الدفعة الأخيرة إذا كانت غير فارغة / Return last batch if not empty
        if batch:
            yield batch

    # ─── دوال مساعدة خاصة / Private helper methods ───────────────────

    def _load_tif(self, path: Path) -> np.ndarray:
        """
        تحميل ملفات TIF باستخدام rasterio (يدعم متعدد النطاقات)
        Load TIF files using rasterio (supports multi-band)
        """
        with rasterio.open(str(path)) as src:
            # قراءة نطاقات RGB (الأولى 3 نطاقات)
            # Read RGB bands (first 3 bands)
            num_bands = src.count
            if num_bands >= 3:
                data = src.read([1, 2, 3])  # (3, H, W)
            else:
                data = src.read(1)  # (H, W) grayscale
                data = np.stack([data, data, data], axis=0)  # → (3, H, W)

            # تحويل من (3, H, W) إلى (H, W, 3)
            image = np.transpose(data, (1, 2, 0))

            # التطبيع لنطاق 0-255 إذا كانت القيم خارج هذا النطاق
            # Normalize to 0-255 if values are outside this range
            if image.dtype != np.uint8:
                min_val = image.min()
                max_val = image.max()
                if max_val > min_val:
                    image = ((image - min_val) / (max_val - min_val) * 255).astype(
                        np.uint8
                    )
                else:
                    image = np.zeros_like(image, dtype=np.uint8)

        return image

    def _load_standard(self, path: Path) -> np.ndarray:
        """
        تحميل الصور القياسية (PNG, JPG) باستخدام PIL
        Load standard images (PNG, JPG) using PIL
        """
        img = Image.open(str(path)).convert("RGB")
        return np.array(img)

    def _handle_large_image(self, image: np.ndarray) -> np.ndarray:
        """
        معالجة الصور الكبيرة بتقليص حجمها للحد الأقصى المسموح
        Handle large images by downscaling to max allowed size
        """
        h, w = image.shape[:2]
        if max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(
                f"📏 تم تقليص الصورة من ({w}x{h}) إلى ({new_w}x{new_h})"
            )
        return image

    def _resize_image(
        self, image: np.ndarray, target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        تغيير حجم الصورة للأبعاد المحددة
        Resize image to specified dimensions

        المعاملات / Parameters:
            image: مصفوفة الصورة / Image array
            target_size: (عرض، ارتفاع) / (width, height)
        """
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    def get_image_info(self, image_path: str) -> dict:
        """
        الحصول على معلومات الصورة دون تحميلها كاملاً
        Get image metadata without fully loading it

        المعاملات / Parameters:
            image_path (str): مسار الصورة / Path to image

        المُخرجات / Returns:
            dict: معلومات الصورة / Image metadata
        """
        path = Path(image_path)
        info = {"path": str(path), "name": path.name, "format": path.suffix}

        try:
            if path.suffix.lower() in {".tif", ".tiff"} and RASTERIO_AVAILABLE:
                with rasterio.open(str(path)) as src:
                    info.update(
                        {
                            "width": src.width,
                            "height": src.height,
                            "bands": src.count,
                            "dtype": str(src.dtypes[0]),
                            "crs": str(src.crs) if src.crs else None,
                        }
                    )
            else:
                with Image.open(str(path)) as img:
                    info.update(
                        {
                            "width": img.width,
                            "height": img.height,
                            "bands": len(img.getbands()),
                            "mode": img.mode,
                        }
                    )
        except Exception as e:
            info["error"] = str(e)

        return info

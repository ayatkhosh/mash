# UrbanInsight 🏢🛰️
## نظام كشف المباني وتقسيمها من صور الأقمار الصناعية
### Building Detection and Segmentation System from Satellite Imagery

---

## 📋 نبذة عن المشروع / Project Overview

**UrbanInsight** هو نظام متكامل واحترافي لكشف وتقسيم المباني من صور الأقمار الصناعية عالية الدقة، يستخدم:

**UrbanInsight** is a comprehensive, professional system for detecting and segmenting buildings from high-resolution satellite imagery, using:

| المكوّن | الوصف |
|---------|-------|
| **Grounding DINO** | كشف المباني والنباتات والفراغات بالنصوص التوصيفية |
| **SAM** | إنتاج أقنعة دقيقة للمناطق المكتشفة |
| **Post-Processing** | تنظيف وتحسين جودة الأقنعة |
| **Building ID Generator** | توليد معرّفات فريدة لكل مبنى |

---

## 🗂️ هيكل المشروع / Project Structure

```
mash/
├── 📓 UrbanInsight_Main.ipynb          # Notebook الرئيسي (8 خلايا)
├── 🧪 UrbanInsight_Test.ipynb          # Notebook الاختبار
├── 📖 README_AR.md                     # هذا الملف
├── 📦 requirements.txt                 # المكتبات المطلوبة
│
├── utils/                              # الوحدات المساعدة
│   ├── __init__.py
│   ├── data_loader.py                  # تحميل ومعالجة الصور
│   ├── dino_detector.py                # كشف المباني بـ DINO
│   ├── sam_segmenter.py                # تقسيم الصور بـ SAM
│   ├── post_processor.py               # معالجة ما بعد التقسيم
│   └── building_id_generator.py        # توليد معرّفات المباني
│
└── configs/                            # ملفات الإعدادات
    ├── model_config.yaml               # إعدادات النماذج
    ├── prompt_config.yaml              # نصوص الكشف (Prompts)
    └── processing_config.yaml          # إعدادات المعالجة
```

---

## ⚡ المتطلبات / Requirements

### البيئة الموصى بها / Recommended Environment
- **Google Colab** مع GPU (T4 أو أعلى) / with GPU (T4 or higher)
- **Python** 3.9+
- **CUDA** 11.8+ (لتسريع GPU) / for GPU acceleration

### المكتبات / Libraries
```bash
pip install -r requirements.txt
```

---

## 🚀 طريقة التشغيل / How to Run

### على Google Colab / On Google Colab

#### الخطوة 1: رفع ملفات المشروع
```python
# في Colab، ارفع المجلد بأكمله إلى /content/mash
# أو استخدم Google Drive
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/mash /content/mash
```

#### الخطوة 2: تشغيل Notebook الرئيسي
افتح `UrbanInsight_Main.ipynb` وشغّل الخلايا بالترتيب:

| الخلية | الوظيفة | الوقت التقديري |
|--------|---------|--------------|
| 1️⃣ | إعداد البيئة والمكتبات | 3-5 دقائق |
| 2️⃣ | تحميل الإعدادات | < 1 ثانية |
| 3️⃣ | تحميل الصور | 1-3 دقائق |
| 4️⃣ | كشف المباني بـ DINO | 2-5 ثوانٍ/صورة |
| 5️⃣ | تقسيم الصور بـ SAM | 3-10 ثوانٍ/صورة |
| 6️⃣ | المعالجة البعدية | < 1 ثانية/صورة |
| 7️⃣ | توليد المعرّفات | < 1 ثانية |
| 8️⃣ | عرض النتائج | < 5 ثوانٍ |

---

## 🔧 تخصيص النماذج / Model Customization

### تعديل نصوص الكشف (Prompts)
عدّل ملف `configs/prompt_config.yaml`:
```yaml
prompts:
  buildings:
    primary: "building . rooftop . house . structure"
    # أضف أو عدّل حسب منطقتك الجغرافية
```

### تعديل معاملات الكشف
عدّل ملف `configs/model_config.yaml`:
```yaml
grounding_dino:
  box_threshold: 0.30    # ارفعه لتقليل الكشوفات الخاطئة
  text_threshold: 0.25   # ارفعه للكشف الأدق
```

### تعديل معاملات المعالجة
عدّل ملف `configs/processing_config.yaml`:
```yaml
post_processing:
  min_area: 100          # حجم أدنى للمبنى بالبكسل
  fill_holes: true       # ملء الثقوب الداخلية
```

---

## 🏗️ استخدام الوحدات مباشرة / Direct Module Usage

### تحميل الصور / Image Loading
```python
from utils.data_loader import ImageLoader

loader = ImageLoader(image_dir="/data/images", normalize=True)
for img, path in loader.load_all():
    print(f"Loaded: {path.name}, Shape: {img.shape}")
```

### كشف المباني / Building Detection
```python
from utils.dino_detector import DINODetector

detector = DINODetector(
    weights_path="/weights/gdino.pth",
    config_path="/GroundingDINO/config/...",
)
detections = detector.detect_buildings(image)
print(f"Buildings found: {detections['count']}")
```

### تقسيم الصور / Image Segmentation
```python
from utils.sam_segmenter import SAMSegmenter

segmenter = SAMSegmenter(checkpoint_path="/weights/sam_vit_h.pth")
masks = segmenter.segment_from_boxes(image, detections['boxes'])
print(f"Masks created: {len(masks)}")
```

### معالجة النتائج / Processing Results
```python
from utils.post_processor import PostProcessor

processor = PostProcessor(min_area=100, fill_holes=True)
cleaned = processor.process(masks)
stats = processor.compute_statistics(cleaned)
print(f"Buildings after cleaning: {stats['total_buildings']}")
```

### توليد المعرّفات / ID Generation
```python
from utils.building_id_generator import BuildingIDGenerator

gen = BuildingIDGenerator(prefix="BLD", project_id="UrbanInsight")
ids = gen.generate_batch(cleaned, image_path="/data/img.tif")
gen.save_to_json("/output/buildings.json")
```

---

## 📊 مثال على النتائج / Sample Results

بعد تشغيل النموذج، ستجد في مجلد `/content/output`:

```
output/
├── sample_images.png           # عينة من الصور المحملة
├── dino_detection_*.png        # نتائج كشف DINO
├── sam_masks_*.png             # نتائج تقسيم SAM
├── processed_*.png             # النتائج بعد المعالجة
├── statistics_report.png       # تقرير الإحصائيات
└── buildings_data.json         # بيانات جميع المباني
```

### صيغة ملف JSON الناتج:
```json
{
  "metadata": {
    "project_id": "UrbanInsight",
    "total_buildings": 42,
    "generated_at": "2025-01-15T10:30:00Z"
  },
  "buildings": [
    {
      "id": "BLD-A3F2B1C4-0001",
      "confidence_score": 0.9234,
      "geometry": {
        "area_px": 1250,
        "centroid_x": 245.3,
        "centroid_y": 178.6,
        "bbox_x1": 200, "bbox_y1": 150,
        "bbox_x2": 290, "bbox_y2": 210
      }
    }
  ]
}
```

---

## 🐛 حل المشاكل الشائعة / Troubleshooting

| المشكلة | الحل |
|---------|------|
| `_C ImportError` في GroundingDINO | راجع الخلية 1 - الـ Patch مطبّق تلقائياً |
| نفاد ذاكرة GPU | قلّل `batch_size` في `processing_config.yaml` |
| لا توجد كشوفات | قلّل `box_threshold` أو عدّل النص في `prompt_config.yaml` |
| صور TIF لا تُحمل | تأكد من تثبيت `rasterio` |

---

## 📄 الترخيص / License

هذا المشروع مفتوح المصدر ضمن مشروع UrbanInsight.  
This project is open source within the UrbanInsight initiative.

---

## 👥 الفريق / Team

مشروع **UrbanInsight** - وحدة كشف وتقسيم المباني  
**UrbanInsight** Project - Building Detection & Segmentation Unit

> **ملاحظة:** هذا المشروع يتطلب GPU للتشغيل الفعلي. يُوصى باستخدام Google Colab مع T4 GPU أو أعلى.
>
> **Note:** This project requires GPU for actual inference. Google Colab with T4 GPU or higher is recommended.

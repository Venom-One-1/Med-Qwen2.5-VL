"""Shared constants for the ophthalmology multi-label classification task."""

from __future__ import annotations

DEFAULT_MODEL_PATH = "Qwen2.5-VL-3B-EyeCoRE-CoTTwoStage"
DEFAULT_IMAGE_ROOT = "/home/sqw/VisualSearch/mm_linyan/imgdata"
DEFAULT_NUM_LABELS = 20
DEFAULT_MAX_OCT_IMAGES = 5

DEFAULT_SPLIT_TO_ANNO = {
    "train": "VisualSearch/shaanxi_train1216_add_0825/anno.txt",
    "val": "VisualSearch/shaanxi_val1216_add_0825/anno.txt",
    "test": "VisualSearch/shaanxi_test1216_add_0825/anno.txt",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

DEFAULT_PROMPT_TEMPLATE = (
    "你将接收一组眼科图像。"
    "第1张图像是眼底彩照（CFP），其余图像是光学相干断层扫描（OCT）。"
    "请综合所有图像提取用于多标签判别的视觉特征。"
)

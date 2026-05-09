"""
PPT内嵌图片提取器 — 简易版（无argparse子命令，双击即跑）
"""

import cv2
import numpy as np
from pathlib import Path
import json

SLIDES_DIR = Path("test_output_v10/slides")
OUTPUT_DIR = Path("test_output_v10/extracted_v2")
CROP_TEACHER = False  # 不裁老师，确保图形完整

# 逐字稿分析：老师提到"如图"的幻灯片（有几何图），降低过滤阈值
DIAGRAM_SLIDES = {"slide_005", "slide_007", "slide_009", "slide_011"}

COLOR_RANGES = [
    ("red",  (np.array([0, 50, 50]),   np.array([10, 255, 255]))),
    ("red2", (np.array([170, 50, 50]), np.array([180, 255, 255]))),
    ("blue", (np.array([100, 50, 50]), np.array([130, 255, 255]))),
    ("green",(np.array([40, 50, 50]),  np.array([80, 255, 255]))),
]


def crop_ppt(img, ratio=0.28):
    h, w = img.shape[:2]
    return img[:, :int(w * (1 - ratio))]


def extract_contour(img, gray, name, out_dir, has_diagram=False):
    """
    V3：内容分块 + 文字/图片分类
    has_diagram=True -> 降低饱和度阈值，捕获无彩色弧线的线段图

    流程：
    1. 边缘检测 → 闭运算 → 找到所有内容块（文字+图片）
    2. 用色彩饱和度方差区分"图片"和"文字"
       - 文字块：饱和度低（黑字白底）
       - 图片/几何图：饱和度高（彩色弧线、线条）
    """
    h, w = img.shape[:2]
    total = h * w
    results = []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)  # S通道

    # 边缘检测 + 闭运算合并内容块（小核，避免合并到旁边的文字）
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 20, 80)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=2)

    for i, cnt in enumerate(cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]):
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        ratio_area = area / total

        if ratio_area < 0.002 or ratio_area > 0.50:
            continue

        aspect = cw / ch if ch else 0
        if aspect < 0.2 or aspect > 5.0:
            continue

        # ⭐ 文字 vs 图片分类：计算饱和度方差
        roi_s = saturation[y:y+ch, x:x+cw]
        if roi_s.size == 0:
            continue
        sat_std = float(cv2.meanStdDev(roi_s)[1][0][0])

        # 根据是否"如图"幻灯片调整饱和度阈值
        min_sat = 5 if has_diagram else 15
        if sat_std < min_sat:
            continue

        # 边距15%
        pad = int(max(cw, ch) * 0.15)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + cw + pad)
        y2 = min(h, y + ch + pad)

        crop = img[y1:y2, x1:x2]
        p = str(out_dir / f"{name}_cnt_{i:02d}.jpg")
        cv2.imwrite(p, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        results.append({
            "src": "contour",
            "area_pct": round(ratio_area*100, 1),
            "sat_std": round(sat_std, 1),
            "path": p
        })

    return results


def extract_color(img, name, out_dir):
    """方案D：色彩分割"""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros((h, w), dtype=np.uint8)
    for _, (lo, hi) in COLOR_RANGES:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=3)

    results = []
    total = h * w
    for i, cnt in enumerate(cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]):
        x, y, cw, ch = cv2.boundingRect(cnt)
        area_r = cv2.contourArea(cnt) / total
        if area_r < 0.005 or area_r > 0.50:
            continue
        pad = 15
        crop = img[max(0, y-pad):min(h, y+ch+pad), max(0, x-pad):min(w, x+cw+pad)]
        p = str(out_dir / f"{name}_color_{i:02d}.jpg")
        cv2.imwrite(p, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        results.append({"src": "color", "area_pct": round(area_r*100, 1), "path": p})
    return results


def iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if x1 >= x2 or y1 >= y2: return 0
    inter = (x2-x1)*(y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / min(a1, a2)


def main():
    print("=" * 55)
    print("  🔍 PPT内嵌图片提取器 — 批量模式")
    print("=" * 55)

    slides = sorted(SLIDES_DIR.glob("slide_*.jpg"))
    if not slides:
        print(f"  ❌ 未找到幻灯片: {SLIDES_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  输入: {SLIDES_DIR} ({len(slides)} 张)")
    print(f"  输出: {OUTPUT_DIR}")
    print(f"  裁剪老师: {'是' if CROP_TEACHER else '否'}")

    # 清理旧输出文件
    handout_dir = OUTPUT_DIR.parent / "handout_images"
    handout_dir.mkdir(exist_ok=True)
    for f in handout_dir.glob("*.jpg"):
        f.unlink()
    for f in OUTPUT_DIR.glob("*.jpg"):
        f.unlink()

    report = {}

    for sp in slides:
        name = sp.stem
        img = cv2.imread(str(sp))
        if img is None:
            print(f"\n  ❌ {name}: 无法读取")
            continue

        if CROP_TEACHER:
            img = crop_ppt(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        has_diagram = name in DIAGRAM_SLIDES
        a = extract_contour(img, gray, name, OUTPUT_DIR, has_diagram=has_diagram)
        b = extract_color(img, name, OUTPUT_DIR)

        # 精选规则：排除过大(>10%)和过小(<0.3%)的，选面积最小的轮廓
        diagram_candidates = [r for r in a if 0.3 <= r["area_pct"] <= 10.0]
        if diagram_candidates:
            # 几何图通常占1%-2%，选最小的正好是几何图
            diagram_candidates.sort(key=lambda r: r["area_pct"])
            best = diagram_candidates[0]
            handout_list = [best]
        else:
            handout_list = []
        merged = handout_list + b

        # 精选图片输出：每张幻灯片只出1张图
        for r in handout_list:
            src = Path(r["path"])
            src_name = Path(r["path"]).stem
            dst = handout_dir / f"{src_name}.jpg"
            cv2.imwrite(str(dst), cv2.imread(str(src)), [cv2.IMWRITE_JPEG_QUALITY, 95])

        report[name] = {"cnt": len(a), "color": len(b), "total": len(merged)}

        if merged:
            print(f"\n  ✅ {name}: 提取 {len(merged)} 个区域 (轮廓{len(a)}+色彩{len(b)})")
            for r in merged:
                print(f"     [{r['src']}] {Path(r['path']).name}  ({r['area_pct']}%)")
        else:
            print(f"\n  ⏭️  {name}: 未发现图片区域")

    # 汇总
    total = sum(v["total"] for v in report.values())
    with_slides = sum(1 for v in report.values() if v["total"] > 0)

    print(f"\n{'=' * 55}")
    print(f"  📊 汇总")
    print(f"  处理幻灯片: {len(slides)} 张")
    print(f"  有提取内容: {with_slides} 张")
    print(f"  提取图片数: {total} 个")
    print(f"{'=' * 55}")

    # 展示精选图
    handout_dir = OUTPUT_DIR.parent / "handout_images"
    if handout_dir.exists():
        handout_files = sorted(handout_dir.glob("*.jpg"))
        print(f"\n  📋 精选图 ({len(handout_files)} 张，输出到 handout_images/):")
        for f in handout_files:
            print(f"    {f.name}")

    # 保存报告
    with open(OUTPUT_DIR / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    input("\n按 Enter 键退出...")


if __name__ == "__main__":
    main()

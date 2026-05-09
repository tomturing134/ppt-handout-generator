# PPT Handout Generator

从课堂录播视频自动提取 PPT 关键帧 + 逐字稿，融合生成结构化讲义。

## 项目流水线

```
V1：视频 → PPT关键帧提取（Masked SSIM + YOLO 人形检测）
V2：音频 → 逐字稿（faster-whisper）
V3：关键帧 + 逐字稿 → 结构化讲义（Markdown + Word）
  └─ 内嵌图片提取（轮廓检测 + 饱和度分类）
```

## 项目结构

```
├── simple_extractor.py      # PPT内嵌几何图提取器
├── run_extractor.bat        # Windows 双击运行
├── ppt_keyframe_extractor/  # V1 关键帧提取模块
├── audio/                   # V2 音频 + 逐字稿
├── test_output_v10/
│   ├── slides/              # 关键帧原图 (11张)
│   ├── cropped_slides/      # 裁剪后幻灯片
│   └── handout_images/      # 精选几何图输出
├── 讲义_证明.md              # Markdown 讲义
├── 讲义_证明.docx            # Word 讲义
└── pyproject.toml           # 项目配置
```

## 使用方法

1. 将课堂录播视频放在项目目录下
2. 运行 `ppt_keyframe_extractor/` 提取 PPT 关键帧
3. 运行 `audio/` 提取音频并生成逐字稿
4. 双击 `run_extractor.bat` 提取几何图
5. 生成讲义：`node generate_docx.js`

## 技术亮点

- **Masked SSIM**：将老师区域置零后计算 SSIM，避免走动误触发切换
- **饱和度分类**：通过 HSV 饱和度方差区分文字块和几何图
- **逐字稿辅助**：利用 "如图" 关键词降低检测阈值，捕获黑白线段图
- **间距聚类**：自动合并图形区域和附近文字标签（A/B/C/D/O）

## 示例

| 例题 | 提取的几何图 |
|------|-------------|
| OA⊥OC, OB⊥OD → ∠AOB=∠COD | ✅ 角度关系图 |
| BD平分∠ABC, ∠1=∠C → ∠2=∠C | ✅ 角平分线图 |

## 环境

- Python 3.13.12 + OpenCV 4.13.0
- Node.js 22.12.0 + docx 库
- Windows 环境

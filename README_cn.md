# T2AV-Compass: 面向文本到音视频生成的统一评测基准

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://nju-link.github.io/T2AV-Compass/)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/NJU-LINK/T2AV-Compass)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/)

> English version: [README.md](README.md)

## 📖 摘要

**T2AV-Compass** 是一个面向 **Text-to-Audio-Video (T2AV)** 生成的统一评测基准，旨在同时覆盖：单模态质量（视频/音频）、跨模态对齐与同步、复杂指令跟随，以及感知真实感与物理合理性。

文本到音视频（T2AV）生成旨在从自然语言合成时间连贯的视频和语义同步的音频，但其评估仍然碎片化，通常依赖于单模态指标或范围狭窄的基准，无法捕捉复杂提示下的跨模态对齐、指令跟随和感知真实性。

本仓库包含：**500 条 taxonomy-driven 构建的复杂提示词**，以及用于 **MLLM-as-a-Judge** 的细粒度 checklist。

## 🌟 主要特点

- **Taxonomy-Driven 高复杂度基准**：500 条语义密集的提示词，通过分类驱动的策划和视频反演的混合管道合成。针对现有评估中经常被忽视的细粒度视听约束，如画外音和物理因果关系。

- **统一双层评估框架**：
  - **客观评估**：视频质量（VT, VA）、音频质量（PQ, CU）、跨模态对齐（T-A, T-V, A-V, DeSync, LatentSync）
  - **主观评估（MLLM-as-a-Judge）**：基于 checklist 的可解释评估，覆盖 **指令跟随** 和 **感知真实性**

- **广泛基准测试**：系统评估了 11 个最先进的 T2AV 系统，包括 Veo-3.1、Sora-2、Kling-2.6、Wan-2.5/2.6、Seedance-1.5、PixVerse-V5.5、Ovi-1.1、JavisDiT 以及组合管道。

## 📊 评估指标

### 客观指标

| 类别 | 指标 | 描述 |
|------|------|------|
| **视频质量** | VT (Video Technological) | 通过 DOVER++ 评估底层视觉完整性 |
| | VA (Video Aesthetic) | 通过 LAION-Aesthetic V2.5 评估高层感知属性 |
| **音频质量** | PQ (Perceptual Quality) | 信号保真度和声学真实性 |
| | CU (Content Usefulness) | 语义有效性和信息密度 |
| **跨模态对齐** | T-A | 通过 CLAP 的文本-音频对齐 |
| | T-V | 通过 VideoCLIP-XL-V2 的文本-视频对齐 |
| | A-V | 通过 ImageBind 的音频-视频对齐 |
| | DS (DeSync) | 时间同步误差（越低越好） |
| | LS (LatentSync) | 说话人脸场景的唇形同步质量 |

### 主观指标（MLLM-as-a-Judge）

**指令跟随 (IF)** - 7 个维度，17 个子维度：
- **Attribute（属性）**：外观、数量
- **Dynamics（动态）**：运动、交互、变换、镜头运动
- **Cinematography（摄影）**：光照、构图、调色
- **Aesthetics（美学）**：风格、氛围
- **Relations（关系）**：空间、逻辑
- **World Knowledge（世界知识）**：事实知识
- **Sound（声音）**：音效、语音、音乐

**真实性** - 5 个指标：
- **视频**：MSS（运动平滑度）、OIS（物体完整性）、TCS（时间连贯性）
- **音频**：AAS（声学伪影）、MTC（材质-音色一致性）

## 📦 数据文件

- `prompts_with_checklist.json`：核心 benchmark 数据（500 条提示词 + checklist）

## 🧩 数据格式

每个样本是一个 JSON object，核心字段如下：

| 字段 | 类型 | 说明 |
|------|------|------|
| `index` | int | 样本编号（1~500） |
| `source` | str | 来源标识（如 `LMArena`、`RealVideo`、`VidProM`、`Kling`、`Shot2Story`） |
| `subject_matter` | str | 主题/题材 |
| `core_subject` | list[str] | 核心主体类别（People/Objects/Animals…） |
| `event_scenario` | list[str] | 场景类别（Urban/Living/Natural/Virtual…） |
| `sound_type` | list[str] | 声音类别（Ambient/Musical/Speech…） |
| `camera_movement` | list[str] | 镜头运动（Static/Translation/Zoom…） |
| `prompt` | str | **整合提示词**（视觉+声音+语音等混合描述） |
| `video_prompt` | str | 仅视觉描述（便于视频端模型输入） |
| `audio_prompt` | str | 非语音音频描述（可为空字符串） |
| `speech_prompt` | list[object] | 结构化语音，元素含 `speaker`/`description`/`text` |
| `video` | str | 参考视频路径（若有；无则为空字符串） |
| `checklist_info` | object | MLLM-as-a-Judge 用的 checklist |

## 🧠 模型适配

- **端到端 T2AV 模型**（如 Veo、Kling）：直接用 `prompt`
- **两阶段/分模块系统**：
  - 视频模型：`video_prompt`
  - 音频模型：`audio_prompt`
  - TTS/语音：`speech_prompt`

## 🚀 快速开始

```python
import json

with open("prompts_with_checklist.json", "r", encoding="utf-8") as f:
    data = json.load(f)

item = data[0]
print(f"提示词: {item['prompt'][:200]}...")
print(f"视频提示词: {item['video_prompt'][:200]}...")
print(f"音频提示词: {item['audio_prompt']}")
print(f"语音提示词: {item['speech_prompt']}")
print(f"Checklist 维度: {list(item['checklist_info'].keys())}")
```

## 📈 引用

如果该工作对你的研究有帮助，欢迎引用：

```bibtex
@misc{cao2025t2avcompass,
  title        = {T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation},
  author       = {Cao, Zhe and Wang, Tao and Wang, Jiaming and Wang, Yanghai and Zhang, Yuanxing and Chen, Jialu and Deng, Miao and Wang, Jiahao and Guo, Yubin and Liao, Chenxi and Zhang, Yize and Zhang, Zhaoxiang and Liu, Jiaheng},
  year         = {2025},
  note         = {Preprint},
}
```

## 🔗 链接

- **项目主页**: [github.com/NJU-LINK/T2AV-Compass](https://github.com/NJU-LINK/T2AV-Compass)
- **数据集**: [huggingface.co/datasets/NJU-LINK/T2AV-Compass](https://huggingface.co/datasets/NJU-LINK/T2AV-Compass)

## 📧 联系方式

- `zhecao@smail.nju.edu.cn`
- `liujiaheng@nju.edu.cn`

---

**NJU-LINK Team, 南京大学** · **Kling Team, 快手科技** · **中国科学院自动化研究所**

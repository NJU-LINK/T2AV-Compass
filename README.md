# T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation

[**Project Page**](https://github.com/NJU-LINK/T2AV-Compass)

**T2AV-Compass** 是一个面向 **Text-to-Audio-Video (T2AV)** 生成的统一评测基准，旨在同时覆盖：单模态质量（视频/音频）、跨模态对齐与同步、复杂指令跟随，以及感知真实感与物理合理性。

本仓库/数据包包含：500 条 taxonomy-driven 构建的复杂提示词，以及用于 MLLM-as-a-Judge 的细粒度 checklist。

> English version: `README_EN.md`

## 🌟 主要特点（对应论文）

- **Taxonomy-driven Prompt Curation**：500 条多样、长文本、可诊断的 T2AV prompts，覆盖主体、场景、声音类型、摄影/镜头运动等维度。
- **Dual-level Evaluation Framework**：
  - **Objective evaluation**：视频质量、音频质量、以及跨模态一致性/同步等信号级指标。
  - **Subjective evaluation (MLLM-as-a-Judge)**：围绕 checklist 的 **instruction following** 与 **perceptual realism** 评估，更可解释、更便于定位失败模式。

## 📦 数据文件

- `prompts_with_checklist.json`：核心 benchmark 数据（500 条提示词 + checklist）。

## 🧩 `prompts_with_checklist.json` 字段说明（简版）

每个样本是一个 JSON object，核心字段如下：

| 字段 | 类型 | 说明 |
|---|---|---|
| `index` | int | 样本编号（1~500） |
| `source` | str | 来源标识（如 `LMArena` / `RealVideo`） |
| `subject_matter` | str | 主题/题材 |
| `core_subject` | list[str] | 核心主体类别（People/Objects/Animals…） |
| `event_scenario` | list[str] | 场景类别（Urban/Living/Natural/Virtual…） |
| `sound_type` | list[str] | 声音类别（Ambient/Musical/Speech…） |
| `camera_movement` | list[str] | 镜头运动（Static/Translation/Zoom…） |
| `prompt` | str | **整合提示词**（视觉+声音+语音等混合描述） |
| `video_prompt` | str | 仅视觉描述（便于视频端模型输入） |
| `audio_prompt` | str | 非语音音频描述（可为空字符串） |
| `speech_prompt` | list[object] | 结构化语音（可为空数组），元素含 `speaker`/`description`/`text` |
| `video` | str | 参考视频路径（若有；无则为空字符串） |
| `checklist_info` | object | MLLM-as-a-Judge 用的 checklist（见下） |

### `checklist_info` 评估维度（7 类）

`checklist_info` 将可解释的评估点组织为 7 类（每个子项是自然语言 yes/no 问句；不适用则为 `null`）：

- **Aesthetics**：风格/氛围
- **Attribute**：外观属性/数量
- **Cinematography**：光照/构图/调色
- **Dynamics**：运动/交互/变化/镜头运动
- **Relations**：空间关系/逻辑关系
- **Sound**：音效/语音/音乐/非语音约束
- **WorldKnowledge**：事实/常识/物理合理性

## 🧠 如何做“模型适配”（prompt 组织方式）

- **端到端 T2AV 模型**：直接用 `prompt`
- **两阶段/分模块系统**：
  - 视频模型：`video_prompt`
  - 音频模型：`audio_prompt`
  - TTS/语音：`speech_prompt`

## 🚀 Quick Start

```python
import json

with open("prompts_with_checklist.json", "r", encoding="utf-8") as f:
    data = json.load(f)

item = data[0]
print(item["prompt"])
print(item["video_prompt"])
print(item["audio_prompt"])
print(item["speech_prompt"])
print(item["checklist_info"].keys())
```

## 📈 引用

如果该工作对你的研究有帮助，欢迎引用（请以论文最终版本为准）：

```bibtex
@article{t2av_compass2025,
  title   = {T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation},
  author  = {NJU-LINK Team},
  year    = {2025}
}
```


# T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation

[**Project Page**](https://github.com/NJU-LINK/T2AV-Compass)

**T2AV-Compass** is a unified benchmark for evaluating **Text-to-Audio-Video (T2AV)** generation, targeting not only unimodal quality (video/audio) but also cross-modal alignment & synchronization, complex instruction following, and perceptual realism grounded in physical/common-sense constraints.

This package includes 500 taxonomy-driven prompts and fine-grained checklists for an MLLM-as-a-Judge protocol.

> 中文版：`README.md`

## 🌟 Key Features (as described in the paper)

- **Taxonomy-driven prompt curation**: 500 diverse, long-horizon, semantically rich prompts with constraints over cinematography, physical causality, and acoustic environments.
- **Dual-level evaluation framework**:
  - **Objective evaluation**: signal-level metrics for video quality, audio quality, and cross-modal alignment/synchronization.
  - **Subjective evaluation (MLLM-as-a-Judge)**: interpretable checklist-based assessment for **instruction following** and **perceptual realism**.

## 📦 Files

- `prompts_with_checklist.json`: the core benchmark data (500 prompts + checklists).

## 🧩 Schema of `prompts_with_checklist.json` (brief)

Each sample is a JSON object with the key fields below:

| Field | Type | Notes |
|---|---|---|
| `index` | int | sample id (1–500) |
| `source` | str | source tag (e.g., `LMArena`, `RealVideo`) |
| `subject_matter` | str | theme/genre |
| `core_subject` | list[str] | subject taxonomy (People/Objects/Animals/…) |
| `event_scenario` | list[str] | scenario taxonomy (Urban/Living/Natural/Virtual/…) |
| `sound_type` | list[str] | sound taxonomy (Ambient/Musical/Speech/…) |
| `camera_movement` | list[str] | camera motion taxonomy (Static/Translation/Zoom/…) |
| `prompt` | str | **integrated prompt** (visual + audio + speech) |
| `video_prompt` | str | video-only prompt |
| `audio_prompt` | str | non-speech audio prompt (can be empty) |
| `speech_prompt` | list[object] | structured speech (can be empty); items contain `speaker`/`description`/`text` |
| `video` | str | reference video path (optional; empty if none) |
| `checklist_info` | object | checklist used by MLLM-as-a-Judge (see below) |

### `checklist_info`: 7 diagnostic dimensions

Each sub-item is a natural-language yes/no question; `null` means not applicable:

- **Aesthetics** (style/mood)
- **Attribute** (appearance/quantity)
- **Cinematography** (lighting/framing/color grading)
- **Dynamics** (motion/interaction/transformation/camera motion)
- **Relations** (spatial/logical coherence)
- **Sound** (SFX/speech/music/non-speech constraints)
- **WorldKnowledge** (commonsense/physical plausibility/facts)

## 🧠 Prompt formatting for different model designs

- **Unified end-to-end T2AV models**: use `prompt`
- **Two-stage / modular pipelines**:
  - video model: `video_prompt`
  - audio model: `audio_prompt`
  - TTS / speech: `speech_prompt`

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

## 📈 Citation

If you find this work useful, please cite (please follow the final paper version for exact metadata):

```bibtex
@article{t2av_compass2025,
  title   = {T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation},
  author  = {NJU-LINK Team},
  year    = {2025}
}
```


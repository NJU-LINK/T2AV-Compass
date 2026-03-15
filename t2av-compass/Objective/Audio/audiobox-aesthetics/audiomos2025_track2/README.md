# AudioMOS Challenge 2025 - Track 2
### Audiobox-aesthetics-style prediction for text-to-speech, text-to-audio and text-to-music systems

[AudioMOS Challenge 2025](https://sites.google.com/view/voicemos-challenge/audiomos-challenge-2025) focuses on automatic evaluation of audio generation systems. Track-2 adopts the 4 axes of audiobox aesthetic scores as the targets to evaluate the automatic evaluation systems of audio samples from text-to-speech (TTS), text-to-audio (TTA), and text-to-music (TTM) systems. 

## News
- 06/05/25: release the ground-truth human annotations for the evaluation set.
- 04/01/25: release the training and dev splits.

## Training splits
We divide the AES-natural dataset into training (2700 samples) and development (250 samples) sets.
In the [train](https://github.com/facebookresearch/audiobox-aesthetics/blob/main/audiomos2025_track2/audiomos2025-track2-train_list.csv) and [dev](https://github.com/facebookresearch/audiobox-aesthetics/blob/main/audiomos2025_track2/audiomos2025-track2-dev_list.csv) files, you can find the **sample_id**, **data_path**, and the average scores of **4 audiobox aesthetic axes**. You also can refer to our previous [jsonl](https://github.com/facebookresearch/audiobox-aesthetics/tree/main/evaluation_data) files to access the full 10 scores of each axe and sample. 


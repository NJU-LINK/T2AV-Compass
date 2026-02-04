import os
import json
import torch
import laion_clap
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def main():
    prompts_file = "prompts.json"
    videos_root = "videos"
    ckpt_path = "./music_speech_audioset_epoch_15_esc_89.98.pt" 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    print(f"Loading CLAP model from {ckpt_path}...")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}. Please download it first.")

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base', device=device)
    model.load_ckpt(ckpt_path)
    model.eval()

    print(f"Loading prompts from {prompts_file}...")
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts_data = json.load(f)
    
    prompt_map = {item['index']: item['audio_description'] for item in prompts_data}

    sub_dirs = [d for d in os.listdir(videos_root) if os.path.isdir(os.path.join(videos_root, d))]
    sub_dirs.sort()

    print(f"Found sub-directories: {sub_dirs}")

    for sub_dir in sub_dirs:
        current_dir_path = os.path.join(videos_root, sub_dir)
        output_json_path = os.path.join(current_dir_path, "clap_scores.json")
        
        print(f"\nProcessing folder: {sub_dir} -> Saving to {output_json_path}")
        
        folder_results = []
        
        for idx in tqdm(prompt_map.keys(), desc=f"Scanning {sub_dir}"):
            video_filename = f"{idx}.mp4"
            video_path = os.path.join(current_dir_path, video_filename)
            text_desc = prompt_map[idx]
            
            score = None

            if os.path.exists(video_path):
                try:
                    with torch.no_grad():
                        audio_embed = model.get_audio_embedding_from_filelist(
                            x=[video_path], 
                            use_tensor=True
                        )
                        
                        text_embed = model.get_text_embedding(
                            [text_desc], 
                            use_tensor=True
                        )

                        audio_embed = torch.nn.functional.normalize(audio_embed, dim=-1)
                        text_embed = torch.nn.functional.normalize(text_embed, dim=-1)

                        sim_tensor = audio_embed @ text_embed.t()
                        score = sim_tensor.item()
                        
                        score = max(-1.0, min(1.0, score))

                except Exception as e:
                    print(f"Warning: Could not process audio for {video_path}. Error: {e}")
            
            folder_results.append({
                "index": idx,
                "clap": round(score, 4) if score is not None else None
            })
        
        folder_results.sort(key=lambda x: x['index'])
        
        with open(output_json_path, 'w', encoding='utf-8') as out_f:
            json.dump(folder_results, out_f, indent=4, ensure_ascii=False)

    print("\nAll done!")

if __name__ == "__main__":
    main()

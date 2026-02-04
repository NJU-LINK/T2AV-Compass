import os
import json
import cv2
import numpy as np
import torch
import glob
from tqdm import tqdm

from modeling import VideoCLIP_XL
from utils.text_encoder import text_encoder

def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break

v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)

def normalize(data):
    return (data / 255.0 - v_mean) / v_std

def video_preprocessing(video_path, fnum=8):
    if not os.path.exists(video_path):
        return None
    
    try:
        video = cv2.VideoCapture(video_path)
        frames = [x for x in _frame_from_video(video)]
        video.release()
        
        if len(frames) == 0:
            return None

        step = max(1, len(frames) // fnum)
        frames = frames[::step][:fnum]
        
        if len(frames) < fnum:
            frames = frames + [frames[-1]] * (fnum - len(frames))

        vid_tube = []
        for fr in frames:
            fr = fr[:,:,::-1]
            fr = cv2.resize(fr, (224, 224))
            fr = np.expand_dims(normalize(fr), axis=(0, 1))
            vid_tube.append(fr)
            
        vid_tube = np.concatenate(vid_tube, axis=1)
        vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
        vid_tube = torch.from_numpy(vid_tube)
        return vid_tube
        
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

def main():
    prompts_file = "prompts.json"
    videos_root = "videos"
    
    print(f"Loading prompts from {prompts_file}...")
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts_data = json.load(f)
    
    prompt_map = { item['index']: item['video_description'] for item in prompts_data }

    print("Loading VideoCLIP-XL model...")
    videoclip_xl = VideoCLIP_XL()
    state_dict = torch.load("./VideoCLIP-XL-v2.bin", map_location="cpu") 
    videoclip_xl.load_state_dict(state_dict)
    videoclip_xl.cuda().eval()
    
    sub_dirs = [d for d in os.listdir(videos_root) if os.path.isdir(os.path.join(videos_root, d))]
    sub_dirs.sort()

    print(f"Found sub-directories: {sub_dirs}")

    for sub_dir in sub_dirs:
        current_dir_path = os.path.join(videos_root, sub_dir)
        output_json_path = os.path.join(current_dir_path, "clip_scores.json")
        
        print(f"\nProcessing folder: {sub_dir} -> Saving to {output_json_path}")
        
        folder_results = []
        
        for idx in tqdm(prompt_map.keys(), desc=f"Scanning {sub_dir}"):
            video_filename = f"{idx}.mp4"
            video_path = os.path.join(current_dir_path, video_filename)
            text_desc = prompt_map[idx]
            
            video_input = video_preprocessing(video_path)
            
            if video_input is None:
                folder_results.append({
                    "index": idx,
                    "clip": None
                })
                continue
            
            with torch.no_grad():
                video_input = video_input.float().cuda()
                
                video_feature = videoclip_xl.vision_model.get_vid_features(video_input).float()
                video_feature = video_feature / video_feature.norm(dim=-1, keepdim=True)
                
                text_input = text_encoder.tokenize([text_desc], truncate=True).cuda()
                text_feature = videoclip_xl.text_model.encode_text(text_input).float()
                text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
                
                similarity = (text_feature @ video_feature.T).item() * 100.0
                
                folder_results.append({
                    "index": idx,
                    "clip": round(similarity, 4)
                })
        
        folder_results.sort(key=lambda x: x['index'])
        
        with open(output_json_path, 'w', encoding='utf-8') as out_f:
            json.dump(folder_results, out_f, indent=4, ensure_ascii=False)
            
    print("\nAll done!")

if __name__ == "__main__":
    main()

import json
import os
import re
import time
import base64
import tempfile
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# ================= Configuration =================
API_KEY = os.getenv("T2AV_API_KEY", "")
BASE_URL = os.getenv("T2AV_BASE_URL", "")
EVAL_MODEL = 'gemini-2.5-pro'

VIDEO_ROOT_PATH = './eval_videos/ltx2/ltx2/'
EVAL_JSON_PATH = './prompts_with_checklist.json'
OUTPUT_PATH = './eval_results/ltx2/eval_results_checklist_1.json'

MAX_WORKERS = 16  
API_TEMPERATURE = 0.0
TIMEOUT_SECONDS = 1800
# ===========================================

if not API_KEY:
    raise ValueError("Missing API key. Set T2AV_API_KEY in your environment.")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL or None)

EVAL_PROMPT_TEMPLATE = (
    "Evaluate the model-generated video content based on the following specific criterion.\n"
    "Criterion: {}\n\n"
    "Please rate the completion quality of the video on a 5-point Likert scale:\n"
    "- 1: Strongly incomplete (Completely failed / Not present).\n"
    "- 2: Somewhat incomplete (Poor / Major discrepancies).\n"
    "- 3: Neutral (Fair / Acceptable but still has flaws).\n"
    "- 4: Somewhat complete (Good / Mostly accurate).\n"
    "- 5: Fully complete (Excellent / Perfectly meets the standard).\n\n"
    "You must respond ONLY with a valid JSON object using the following structure:\n"
    "{{\n"
    " \"reason\": \"A detailed explanation for your rating.\",\n"
    " \"score\": An integer between 1 and 5\n"
    "}}\n"
)

def encode_video(path):
    """Read video and convert to Base64."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except:
        return None

def robust_api_call(prompt, video_path):
    """Execute API call with retry logic."""
    video_b64 = encode_video(video_path)
    if not video_b64:
        return {"score": -1, "reason": "Video file missing or unreadable"}

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=EVAL_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:video/mp4;base64,{video_b64}"}}
                    ]
                }],
                response_format={"type": "json_object"},
                timeout=TIMEOUT_SECONDS,
                temperature=API_TEMPERATURE
            )
            content = response.choices[0].message.content
            return parse_json(content)
        except Exception as e:
            print(f"Retry {attempt+1}/3 due to: {e}")
            time.sleep(2 * (attempt + 1))
    
    return {"score": -1, "reason": "API Timeout or Error after retries"}

def parse_json(text):
    """Parse non-standard JSON response."""
    try:
        return json.loads(text)
    except:
        match = re.search(r'\{.*\}', text.replace('\n', ''), re.DOTALL)
        if match:
            try: return json.loads(match.group())
            except: pass
        
        s_match = re.search(r'"score":\s*(\d+)', text)
        r_match = re.search(r'"reason":\s*"(.*?)"', text)
        return {
            "score": int(s_match.group(1)) if s_match else -1,
            "reason": r_match.group(1) if r_match else text[:100]
        }

def save_results(data):
    """Atomic write to prevent file corruption."""
    dir_name = os.path.dirname(OUTPUT_PATH)
    if not os.path.exists(dir_name): os.makedirs(dir_name)
    
    with tempfile.NamedTemporaryFile(mode='w', dir=dir_name, delete=False, encoding='utf-8') as tf:
        json.dump(data, tf, indent=2, ensure_ascii=False)
        temp_name = tf.name
    shutil.move(temp_name, OUTPUT_PATH)

def process_task(task_args):
    """Worker function for a single task."""
    vid_idx, vid_path, cat, dim, question = task_args
    
    prompt = EVAL_PROMPT_TEMPLATE.format(question)
    result = robust_api_call(prompt, vid_path)
    
    return vid_idx, cat, dim, result

def compute_and_report_metrics(results_data):
    """Compute and print evaluation metrics report."""
    print("\n" + "="*60)
    print(f"{'EVALUATION REPORT':^60}")
    print("="*60)

    buckets = {}
    
    for vid_id, categories in results_data.items():
        for major, sub_dims in categories.items():
            if major not in buckets:
                buckets[major] = {}
            
            for sub, res in sub_dims.items():
                raw_score = res.get('score', -1)
                if raw_score == -1: 
                    continue
                
                norm_score = (raw_score - 1) / 4.0
                
                if sub not in buckets[major]:
                    buckets[major][sub] = []
                buckets[major][sub].append(norm_score)

    avg_scores = {}
    for major, subs in buckets.items():
        avg_scores[major] = {}
        for sub, score_list in subs.items():
            if score_list:
                avg_scores[major][sub] = sum(score_list) / len(score_list)

    print(f"\n{'[1] Detailed Breakdown':<60}")
    print("-" * 60)
    
    video_major_scores = {}
    
    sorted_majors = sorted(avg_scores.keys())
    
    for major in sorted_majors:
        subs = avg_scores[major]
        if not subs: continue
        
        major_avg = sum(subs.values()) / len(subs)
        
        if major != 'Sound':
            video_major_scores[major] = major_avg
            prefix = "[Video]"
        else:
            prefix = "[Audio]"
            
        print(f"{prefix} {major:<20} : {major_avg:.4f}")
        for sub, val in sorted(subs.items()):
            print(f"    - {sub:<18} : {val:.4f}")
        print("-" * 60)

    print(f"\n{'[2] IF-Video Calculation':<60}")
    if video_major_scores:
        if_video = sum(video_major_scores.values()) / len(video_major_scores)
        
        components = [f"{k}({v:.4f})" for k, v in video_major_scores.items()]
        print(f"Formula    : ({' + '.join(components)}) / {len(components)}")
        print(f"Result     : {if_video:.4f} -> {if_video*100:.2f}/100")
    else:
        if_video = 0.0
        print("Result     : N/A (No video metrics found)")

    print(f"\n{'[3] IF-Audio Calculation':<60}")
    if 'Sound' in avg_scores:
        sound_subs = avg_scores['Sound']
        
        s_speech = sound_subs.get('Speech', 0.0)
        s_sfx = sound_subs.get('SoundEffects', 0.0)
        s_music = sound_subs.get('Music', 0.0)
        s_nonspeech = sound_subs.get('NonSpeech', 1.0) 
        
        penalty = 1.0 - s_nonspeech
        
        if_audio = ((s_speech - penalty) + s_sfx + s_music) / 3.0
        
        print(f"Formula    : ((Speech({s_speech:.4f}) - (1 - NonSpeech({s_nonspeech:.4f}))) + SoundEffects({s_sfx:.4f}) + Music({s_music:.4f})) / 3")
        print(f"Result     : {if_audio:.4f} -> {if_audio*100:.2f}/100")
    else:
        if_audio = 0.0
        print("Result     : N/A (No audio metrics found)")

    print("="*60 + "\n")
    
    return if_video, if_audio

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='Video Evaluation with Checklist')
    parser.add_argument('--recalculate', action='store_true', 
                        help='Only recalculate metrics from existing results without re-evaluating')
    args = parser.parse_args()
    
    # 如果只是重新计算分数
    if args.recalculate:
        if not os.path.exists(OUTPUT_PATH):
            print(f"Error: Results file not found at {OUTPUT_PATH}")
            return
        
        try:
            final_results = json.load(open(OUTPUT_PATH, "r", encoding='utf-8'))
            print(f"Loaded {len(final_results)} evaluated videos from {OUTPUT_PATH}")
            print("Recalculating metrics...")
            compute_and_report_metrics(final_results)
        except Exception as e:
            print(f"Error loading results file: {e}")
        return
    
    prompts_data = json.load(open(EVAL_JSON_PATH, "r", encoding='utf-8'))
    
    final_results = {}
    if os.path.exists(OUTPUT_PATH):
        try:
            final_results = json.load(open(OUTPUT_PATH, "r", encoding='utf-8'))
            print(f"Resuming from {len(final_results)} evaluated videos.")
        except:
            print("History file corrupted, starting fresh.")

    all_tasks = []
    
    for item in prompts_data:
        idx_str = str(item['index'])
        video_path = os.path.join(VIDEO_ROOT_PATH, f"{item['index']}.mp4")
        
        if idx_str not in final_results:
            final_results[idx_str] = {}
            
        checklist = item.get('checklist_info', {})
        
        for category, dimensions in checklist.items():
            for dim_name, question in dimensions.items():
                if not question: continue
                
                is_done = (
                    idx_str in final_results and 
                    category in final_results[idx_str] and 
                    dim_name in final_results[idx_str][category]
                )
                
                if not is_done:
                    all_tasks.append((idx_str, video_path, category, dim_name, question))

    print(f"Total tasks pending: {len(all_tasks)}")
    if not all_tasks:
        print("All tasks completed.")
        return

    batch_save_counter = 0 
    SAVE_INTERVAL = 20 

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(process_task, t): t for t in all_tasks}
        
        pbar = tqdm(as_completed(future_map), total=len(all_tasks), desc="Global Eval")
        
        for future in pbar:
            try:
                vid_idx, cat, dim, res = future.result()
                
                if cat not in final_results[vid_idx]:
                    final_results[vid_idx][cat] = {}
                final_results[vid_idx][cat][dim] = res
                
                batch_save_counter += 1
                if batch_save_counter >= SAVE_INTERVAL:
                    save_results(final_results)
                    batch_save_counter = 0
                    
            except Exception as e:
                print(f"Critical Worker Error: {e}")

    save_results(final_results)
    print(f"Evaluation Complete. Results saved to {OUTPUT_PATH}")

    if final_results:
        compute_and_report_metrics(final_results)
    else:
        print("No results to compute.")

if __name__ == '__main__':
    main()

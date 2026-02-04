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
PROMPT_FILES = {
    "AAS": "./eval_prompts/AAS.md",
    "MSS": "./eval_prompts/MSS.md",
    "MTC": "./eval_prompts/MTC.md",
    "OIS": "./eval_prompts/OIS.md",
    "TCS": "./eval_prompts/TCS.md"
}

OUTPUT_PATH = './eval_results/ltx2/eval_results_realism_1.json'

MAX_WORKERS = 16  
API_TEMPERATURE = 0.0
TIMEOUT_SECONDS = 1800
# ===========================================

if not API_KEY:
    raise ValueError("Missing API key. Set T2AV_API_KEY in your environment.")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL or None)

def load_all_prompts(prompt_map):
    loaded_prompts = {}
    for key, path in prompt_map.items():
        if not os.path.exists(path):
            print(f"Warning: Prompt file not found: {path}")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            loaded_prompts[key] = f.read()
    return loaded_prompts

def encode_video(path):
    """Read video and convert to Base64."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except:
        return None

def robust_api_call(prompt, video_path, criteria_name):
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
            return parse_json(content, criteria_name)
        except Exception as e:
            print(f"Retry {attempt+1}/3 due to: {e}")
            time.sleep(2 * (attempt + 1))
    
    return {"score": -1, "reason": "API Timeout or Error after retries"}

def parse_json(text, criteria_name):
    """Parse non-standard JSON response."""
    try:
        data = json.loads(text)
        score = data.get(criteria_name)
        reason = data.get("reason")
        return {"score": score, "reason": reason}
    except:
        match = re.search(r'\{.*\}', text.replace('\n', ''), re.DOTALL)
        if match:
            try: 
                data = json.loads(match.group())
                score = data.get(criteria_name)
                return {"score": score, "reason": data.get("reason")}
            except: pass
        
        s_match = re.search(rf'"{criteria_name}":\s*(\d+)', text)
        r_match = re.search(r'"reason":\s*"(.*?)"', text)
        return {
            "score": int(s_match.group(1)) if s_match else -1,
            "reason": r_match.group(1) if r_match else text[:100]
        }

def save_results(data):
    """Atomic write to prevent file corruption."""
    dir_name = os.path.dirname(OUTPUT_PATH)
    if not os.path.exists(dir_name): os.makedirs(dir_name)
    
    sorted_keys = sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else x)
    sorted_data = {k: data[k] for k in sorted_keys}

    with tempfile.NamedTemporaryFile(mode='w', dir=dir_name, delete=False, encoding='utf-8') as tf:
        json.dump(sorted_data, tf, indent=2, ensure_ascii=False)
        temp_name = tf.name
    shutil.move(temp_name, OUTPUT_PATH)

def process_task(task_args):
    """Worker function for a single task."""
    vid_idx, vid_path, criteria_name, prompt_text = task_args
    
    result = robust_api_call(prompt_text, vid_path, criteria_name)
    
    return vid_idx, criteria_name, result

def compute_and_report_metrics(results_data):
    """Compute and print evaluation metrics report."""
    print("\n" + "="*60)
    print(f"{'EVALUATION REPORT':^60}")
    print("="*60)

    buckets = {}
    
    for vid_id, criteria in results_data.items():
        for criteria_dim, res in criteria.items():
            if criteria_dim not in buckets:
                buckets[criteria_dim] = []
            raw_score = res.get('score', -1)
            if raw_score == -1: 
                continue
            
            norm_score = (raw_score - 1) / 4.0
            buckets[criteria_dim].append(norm_score)

    avg_scores = {}
    for criteria_dim, score_list in buckets.items():
        if score_list:
            avg_scores[criteria_dim] = sum(score_list) / len(score_list)

    print(f"\n{'[1] Detailed Breakdown':<60}")
    print("-" * 60)
    
    sorted_keys = sorted(avg_scores.keys())
    
    for key in sorted_keys:
        val = avg_scores.get(key, 0.0)
        print(f"    - {key:<18} : {val:.4f}")
    print("-" * 60)

    print(f"\n{'[2] Video-Realism Calculation':<60}")
    
    mss = avg_scores.get('MSS', 0.0)
    ois = avg_scores.get('OIS', 0.0)
    tcs = avg_scores.get('TCS', 0.0)
    
    video_realism = (mss + ois + tcs) / 3.0
    
    print(f"Formula    : (MSS({mss:.4f}) + OIS({ois:.4f}) + TCS({tcs:.4f})) / 3")
    print(f"Result     : {video_realism:.4f} -> {video_realism*100:.2f}/100")

    print(f"\n{'[3] Audio-Realism Calculation':<60}")
    
    aas = avg_scores.get('AAS', 0.0)
    mtc = avg_scores.get('MTC', 0.0)
    
    audio_realism = (aas + mtc) / 2.0
    
    print(f"Formula    : (AAS({aas:.4f}) + MTC({mtc:.4f})) / 2")
    print(f"Result     : {audio_realism:.4f} -> {audio_realism*100:.2f}/100")

    print("="*60 + "\n")
    
    return video_realism, audio_realism

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='Video Realism Evaluation')
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
    
    try:
        prompts_map = load_all_prompts(PROMPT_FILES)
        print(f"Loaded prompts: {list(prompts_map.keys())}")
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return

    final_results = {}
    if os.path.exists(OUTPUT_PATH):
        try:
            final_results = json.load(open(OUTPUT_PATH, "r", encoding='utf-8'))
            print(f"Resuming from {len(final_results)} evaluated videos.")
            # 如果有现有结果，立即输出分数计算结果
            if final_results:
                print("\n" + "="*60)
                print("Current Results Summary (before processing new tasks):")
                print("="*60)
                compute_and_report_metrics(final_results)
        except:
            print("History file corrupted, starting fresh.")

    all_tasks = []
    
    for i in range(1, 501):
        idx_str = str(i)
        video_path = os.path.join(VIDEO_ROOT_PATH, f"{i}.mp4")
        
        if idx_str not in final_results:
            final_results[idx_str] = {}
            
        if not os.path.exists(video_path):
            continue

        for criteria_name, prompt_text in prompts_map.items():
            is_done = (
                idx_str in final_results and 
                criteria_name in final_results[idx_str] and
                final_results[idx_str][criteria_name].get('score') != -1
            )
            
            if not is_done:
                all_tasks.append((idx_str, video_path, criteria_name, prompt_text))

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
                vid_idx, criteria_name, res = future.result()
                
                if vid_idx not in final_results:
                    final_results[vid_idx] = {}
                
                final_results[vid_idx][criteria_name] = res
                
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
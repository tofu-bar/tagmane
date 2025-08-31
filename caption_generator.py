#!/usr/bin/env python3
"""
GLM-4.1V Caption Generator Script
C#アプリケーションから呼び出されるPythonスクリプト
"""

import sys
import os
import json
import argparse
from PIL import Image
import threading

# PyTorch CUDA メモリ最適化の環境変数設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# Transformers関連のインポート
try:
    from transformers import AutoProcessor, Glm4vForConditionalGeneration
    import torch
    TRANSFORMERS_AVAILABLE = True
    print("Transformers imported successfully", file=sys.stderr)
except Exception as e:
    print(f"Warning: Transformers is not available: {e}", file=sys.stderr)
    TRANSFORMERS_AVAILABLE = False
    AutoProcessor = None
    Glm4vForConditionalGeneration = None

# GLM-4.1V-9B-Thinkingモデルのパス
MODEL_PATH = "THUDM/GLM-4.1V-9B-Thinking"

# デフォルトのキャプションプロンプト
DEFAULT_CAPTION_PROMPT = """Please describe this image in 2-3 concise sentences. Focus on the main subject and key visual elements.

Character: {character}
Copyright: {copyright}

Provide your answer wrapped in <answer></answer> tags:"""

# GLM-4V関連の変数
glm_model = None
glm_processor = None
model_lock = threading.Lock()

def initialize_glm_model():
    """GLM-4Vモデルを初期化する"""
    global glm_model, glm_processor
    
    if not TRANSFORMERS_AVAILABLE:
        print("Transformers is not available. Skipping model initialization.", file=sys.stderr)
        return False
    
    try:
        with model_lock:
            if glm_model is None:
                print("Initializing GLM-4V model...", file=sys.stderr)
                
                # プロセッサーの初期化
                glm_processor = AutoProcessor.from_pretrained(
                    MODEL_PATH, 
                    use_fast=True
                )
                
                # モデルの初期化
                if torch.cuda.is_available():
                    print(f"Using GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr)
                    print(f"GPU memory available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB", file=sys.stderr)
                    
                    glm_model = Glm4vForConditionalGeneration.from_pretrained(
                        pretrained_model_name_or_path=MODEL_PATH,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        attn_implementation="eager",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        max_memory={0: "40GB"},
                    )
                    
                    if hasattr(torch.backends.cuda, 'max_split_size_mb'):
                        torch.backends.cuda.max_split_size_mb = 512
                    
                    torch.cuda.empty_cache()
                else:
                    print("Using CPU", file=sys.stderr)
                    glm_model = Glm4vForConditionalGeneration.from_pretrained(
                        pretrained_model_name_or_path=MODEL_PATH,
                        torch_dtype=torch.float16,
                        attn_implementation="eager",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                
                print("GLM-4V model initialized successfully!", file=sys.stderr)
                return True
                
    except Exception as e:
        print(f"Error initializing GLM-4V model: {e}", file=sys.stderr)
        glm_model = None
        glm_processor = None
        return False

def categorize_tags(tags):
    """タグをカテゴリ別に分類する（簡易版）"""
    categorized = {
        'character': [],
        'copyright': [],
        'artist': [],
        'general': [],
        'rating': [],
        'quality': [],
        'meta': [],
        'model': []
    }
    
    # 基本的なカテゴリ分類ルール
    rating_tags = ['general', 'sensitive', 'questionable', 'explicit']
    quality_tags = ['best quality', 'normal quality', 'bad quality', 'worst quality']
    
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower in rating_tags:
            categorized['rating'].append(tag)
        elif tag_lower in quality_tags:
            categorized['quality'].append(tag)
        elif any(keyword in tag_lower for keyword in ['artist', 'creator', 'by ']):
            categorized['artist'].append(tag)
        elif any(keyword in tag_lower for keyword in ['series', 'game', 'anime', 'manga']):
            categorized['copyright'].append(tag)
        else:
            categorized['general'].append(tag)
    
    return categorized

def generate_caption_streaming(image_path, prompt, tags):
    """GLM-4Vを使用してキャプションを生成する（ストリーミング対応）"""
    global glm_model, glm_processor
    
    if not TRANSFORMERS_AVAILABLE:
        # テスト用ダミー出力
        dummy_text = f"This is a test caption for {os.path.basename(image_path)} with tags: {', '.join(tags[:3])}..."
        for i, char in enumerate(dummy_text):
            print(f"STREAM:{char}", flush=True)
            if i % 10 == 0:  # 10文字ごとに少し待機
                import time
                time.sleep(0.05)
        return dummy_text
    
    # モデルが初期化されていない場合は初期化
    if glm_model is None or glm_processor is None:
        if not initialize_glm_model():
            return "Error: Failed to initialize GLM-4V model"
    
    try:
        # 画像の読み込み
        image = Image.open(image_path).convert('RGB')
        
        # 大きな画像をリサイズ
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            print(f"Image resized to {new_size} for memory efficiency", file=sys.stderr)
        
        # タグをカテゴリ別に分類
        categorized_tags = categorize_tags(tags)
        
        # プロンプト用の辞書を作成
        prompt_replacements = {
            'tags': ", ".join(tags) if tags else "No tags",
            'character': ", ".join(categorized_tags['character']) if categorized_tags['character'] else "No character tags",
            'copyright': ", ".join(categorized_tags['copyright']) if categorized_tags['copyright'] else "No copyright tags",
            'artist': ", ".join(categorized_tags['artist']) if categorized_tags['artist'] else "No artist tags",
            'general': ", ".join(categorized_tags['general'][:20]) if categorized_tags['general'] else "No general tags",
            'rating': ", ".join(categorized_tags['rating']) if categorized_tags['rating'] else "No rating tags",
            'quality': ", ".join(categorized_tags['quality']) if categorized_tags['quality'] else "No quality tags",
            'meta': ", ".join(categorized_tags['meta']) if categorized_tags['meta'] else "No meta tags",
            'model': ", ".join(categorized_tags['model']) if categorized_tags['model'] else "No model tags"
        }
        
        # プロンプトをフォーマット
        formatted_prompt = prompt.format(**prompt_replacements)
        print(f"Using prompt: {formatted_prompt[:100]}...", file=sys.stderr)
        
        with model_lock:
            # トークン化
            inputs = glm_processor.apply_chat_template(
                [{"role": "user", "image": image, "content": formatted_prompt}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True
            ).to(glm_model.device)
            
            # ストリーミング生成設定
            generation_config = {
                'max_new_tokens': 512,
                'do_sample': True,
                'temperature': 0.7,
                'top_p': 0.9,
                'pad_token_id': glm_processor.tokenizer.eos_token_id,
            }
            
            # ストリーミング生成
            generated_text = ""
            with torch.no_grad():
                output = glm_model.generate(**inputs, **generation_config)
                
                # 生成されたテキストをデコード
                response_text = glm_processor.batch_decode(
                    output[:, inputs['input_ids'].size(1):],
                    skip_special_tokens=True
                )[0]
                
                # <answer></answer>タグから内容を抽出
                import re
                answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
                if answer_match:
                    generated_text = answer_match.group(1).strip()
                else:
                    generated_text = response_text.strip()
                
                # ストリーミング風に文字を出力
                for char in generated_text:
                    print(f"STREAM:{char}", flush=True)
            
            # GPU メモリクリーンアップ
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return generated_text
            
    except Exception as e:
        error_msg = f"Error generating caption: {e}"
        print(error_msg, file=sys.stderr)
        return f"Error: {e}"

def save_caption_to_json(image_path, caption):
    """JSONファイルにキャプションを保存する"""
    base_path = os.path.splitext(image_path)[0]
    json_path = base_path + ".json"
    
    try:
        # 既存のJSONファイルがあれば読み込み、なければ新規作成
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {}
        
        # キャプションを追加/更新
        data['caption'] = caption
        
        # ファイルに保存
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return True
        
    except Exception as e:
        print(f"Error saving caption to JSON: {e}", file=sys.stderr)
        return False

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='GLM-4.1V Caption Generator')
    parser.add_argument('--image', required=True, help='Path to image file')
    parser.add_argument('--tags', help='Comma-separated list of tags')
    parser.add_argument('--prompt', help='Custom prompt template')
    parser.add_argument('--save', action='store_true', help='Save caption to JSON file')
    parser.add_argument('--init-only', action='store_true', help='Only initialize model and exit')
    
    args = parser.parse_args()
    
    # モデル初期化のみの場合
    if args.init_only:
        success = initialize_glm_model()
        if success:
            print("SUCCESS", flush=True)
        else:
            print("FAILED", flush=True)
        return
    
    # 画像ファイルの存在確認
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}", file=sys.stderr)
        return
    
    # タグの解析
    tags = []
    if args.tags:
        tags = [tag.strip() for tag in args.tags.split(',') if tag.strip()]
    
    # プロンプトの設定
    prompt = args.prompt if args.prompt else DEFAULT_CAPTION_PROMPT
    
    print(f"Generating caption for: {args.image}", file=sys.stderr)
    print(f"Tags: {tags}", file=sys.stderr)
    
    # キャプション生成
    caption = generate_caption_streaming(args.image, prompt, tags)
    
    if caption and not caption.startswith("Error:"):
        print("FINAL:" + caption, flush=True)
        
        # JSONファイルに保存
        if args.save:
            success = save_caption_to_json(args.image, caption)
            if success:
                print("Caption saved to JSON file", file=sys.stderr)
            else:
                print("Failed to save caption to JSON file", file=sys.stderr)
    else:
        print(f"Failed to generate caption: {caption}", file=sys.stderr)

if __name__ == "__main__":
    main()
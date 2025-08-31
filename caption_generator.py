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

# Windows環境でのエンコーディング問題を回避
if os.name == 'nt':  # Windows
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

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

IMPORTANT: Start your response immediately with <answer>your description here</answer>. Do not use <think> tags. Provide a direct, concise description."""

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
                        max_memory={0: "32GB"},
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

# タグカテゴリ関連の辞書読み込み機能は削除（C#側から分類済みデータを受け取る）

def generate_caption_streaming(image_path, prompt, tags, categorized_tags=None):
    """GLM-4Vを使用してキャプションを生成する（ストリーミング対応）"""
    global glm_model, glm_processor
    
    if not TRANSFORMERS_AVAILABLE:
        # テスト用ダミー出力
        if isinstance(tags, list):
            dummy_text = f"This is a test caption for {os.path.basename(image_path)} with tags: {', '.join(tags[:3])}..."
        else:
            dummy_text = f"This is a test caption for {os.path.basename(image_path)}"
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
        
        # カテゴリ別タグを使用（C#から提供されるか、デフォルトを使用）
        if categorized_tags is None:
            # フォールバック：すべてgeneralとして扱う
            categorized_tags = {
                'character': [],
                'copyright': [],
                'artist': [],
                'general': tags if isinstance(tags, list) else [],
                'rating': [],
                'quality': [],
                'meta': [],
                'model': []
            }
            print(f"Using fallback categorization (all tags as general)", file=sys.stderr)
        
        # プロンプト用の辞書を作成
        print(f"Creating prompt replacements...", file=sys.stderr)
        prompt_replacements = {
            'tags': ", ".join(tags) if isinstance(tags, list) and tags else "No tags",
            'character': ", ".join(categorized_tags['character']) if categorized_tags['character'] else "No character tags",
            'copyright': ", ".join(categorized_tags['copyright']) if categorized_tags['copyright'] else "No copyright tags",
            'artist': ", ".join(categorized_tags['artist']) if categorized_tags['artist'] else "No artist tags",
            'general': ", ".join(categorized_tags['general'][:20]) if categorized_tags['general'] else "No general tags",
            'rating': ", ".join(categorized_tags['rating']) if categorized_tags['rating'] else "No rating tags",
            'quality': ", ".join(categorized_tags['quality']) if categorized_tags['quality'] else "No quality tags",
            'meta': ", ".join(categorized_tags['meta']) if categorized_tags['meta'] else "No meta tags",
            'model': ", ".join(categorized_tags['model']) if categorized_tags['model'] else "No model tags"
        }
        print(f"Prompt replacements created successfully", file=sys.stderr)
        
        # プロンプトをフォーマット
        print(f"Formatting prompt...", file=sys.stderr)
        formatted_prompt = prompt.format(**prompt_replacements)
        print(f"Using prompt: {formatted_prompt}", file=sys.stderr)
        
        with model_lock:
            # トークン化
            print(f"Starting tokenization...", file=sys.stderr)
            try:
                # GLM-4V用の正しいチャットテンプレート形式
                chat_template_input = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": formatted_prompt}
                        ]
                    }
                ]
                print(f"Chat template input created with correct format", file=sys.stderr)
                
                inputs = glm_processor.apply_chat_template(
                    chat_template_input,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True
                )
                print(f"Chat template applied successfully", file=sys.stderr)
                
                inputs = inputs.to(glm_model.device)
                print(f"Inputs moved to device successfully", file=sys.stderr)
            except Exception as e:
                print(f"Error during tokenization: {e}", file=sys.stderr)
                print(f"Error type: {type(e)}", file=sys.stderr)
                import traceback
                print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
                raise e
            
            # ストリーミング生成設定
            print(f"Setting up generation config...", file=sys.stderr)
            generation_config = {
                'max_new_tokens': 512,
                'do_sample': True,
                'temperature': 0.7,
                'top_p': 0.9,
                'pad_token_id': glm_processor.tokenizer.eos_token_id,
            }
            print(f"Generation config created", file=sys.stderr)
            
            # ストリーミング生成
            generated_text = ""
            print(f"Starting generation...", file=sys.stderr)
            try:
                with torch.no_grad():
                    print(f"Calling model.generate...", file=sys.stderr)
                    output = glm_model.generate(**inputs, **generation_config)
                    print(f"Model generation completed", file=sys.stderr)
                
                # 生成されたテキストをデコード
                print(f"Starting text decoding...", file=sys.stderr)
                response_text = glm_processor.batch_decode(
                    output[:, inputs['input_ids'].size(1):],
                    skip_special_tokens=True
                )[0]
                print(f"Raw response: {repr(response_text[:200])}", file=sys.stderr)
            except Exception as e:
                print(f"Error during generation: {e}", file=sys.stderr)
                print(f"Error type: {type(e)}", file=sys.stderr)
                import traceback
                print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
                raise e
                
            # <answer></answer>タグから内容を抽出（リトライ機能付き）
            import re
            generated_text = ""
            max_retries = 2
            
            for retry in range(max_retries + 1):
                answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
                if answer_match:
                    generated_text = answer_match.group(1).strip()
                    print(f"Extracted from <answer> tags: {repr(generated_text[:100])}", file=sys.stderr)
                    break
                elif retry < max_retries:
                    # リトライ: より短いプロンプトで再生成
                    print(f"No <answer> tags found. Retrying with simplified prompt... (attempt {retry + 2}/{max_retries + 1})", file=sys.stderr)
                    simplified_prompt = f"Describe this {categorized_tags['character'][0] if categorized_tags and categorized_tags['character'] else 'character'} cosplay in one sentence. Start with: <answer>"
                    
                    # 再生成（短いトークン制限）
                    retry_config = generation_config.copy()
                    retry_config['max_new_tokens'] = 256
                    
                    with torch.no_grad():
                        retry_inputs = glm_processor.apply_chat_template(
                            [{"role": "user", "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": simplified_prompt}
                            ]}],
                            add_generation_prompt=True,
                            tokenize=True,
                            return_tensors="pt",
                            return_dict=True
                        ).to(glm_model.device)
                        
                        retry_output = glm_model.generate(**retry_inputs, **retry_config)
                        response_text = glm_processor.batch_decode(
                            retry_output[:, retry_inputs['input_ids'].size(1):],
                            skip_special_tokens=True
                        )[0]
                        print(f"Retry response: {repr(response_text[:200])}", file=sys.stderr)
                else:
                    # 最後の手段：<think>タグから有用な情報を抽出
                    print(f"All retries failed. Extracting from <think> content...", file=sys.stderr)
                    think_patterns = [
                        r'cosplay[a-zA-Z\s]*(?:of|as)\s+([^.]+?)\s+from\s+([^.]+?)[.\s]',  # "cosplay of X from Y"
                        r'The image shows.*?cosplay.*?([^.]{20,80})[.\s]',  # "The image shows...cosplay..."
                        r'person.*?cosplaying.*?([^.]{20,80})[.\s]',  # "person cosplaying..."
                        r'([^.]{30,100}(?:cosplay|character|costume)[^.]{0,50})[.\s]'  # 一般的なコスプレ記述
                    ]
                    
                    for pattern in think_patterns:
                        think_match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                        if think_match:
                            generated_text = think_match.group(0).strip()
                            if len(generated_text) > 20:
                                print(f"Extracted from <think>: {repr(generated_text[:100])}", file=sys.stderr)
                                break
                    
                    if not generated_text:
                        # 最終フォールバック
                        character_name = ", ".join(categorized_tags['character']) if categorized_tags and categorized_tags['character'] else "character"
                        copyright_name = ", ".join(categorized_tags['copyright']) if categorized_tags and categorized_tags['copyright'] else "anime series"
                        generated_text = f"A cosplay photo featuring {character_name} from {copyright_name}."
                        print(f"Using fallback description: {generated_text}", file=sys.stderr)
            
            # ストリーミング風に文字を出力
            print(f"Starting streaming output...", file=sys.stderr)
            for char in generated_text:
                try:
                    print(f"STREAM:{char}", flush=True)
                except UnicodeEncodeError:
                    # エンコーディングエラーを回避
                    print(f"STREAM:{char.encode('utf-8', errors='replace').decode('utf-8')}", flush=True)
            
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

def interactive_mode():
    """インタラクティブモード - C#からの連続コマンドを処理"""
    print("READY", flush=True)  # C#に準備完了を通知
    
    while True:
        try:
            # C#からのコマンドを読み取り
            line = input().strip()
            
            if line == "EXIT":
                print("EXITING", flush=True)
                break
            
            # PROCESS_JSON|imagePath|jsonData形式と旧形式の両方をサポート
            if line.startswith("PROCESS_JSON|"):
                parts = line.split("|", 2)
                if len(parts) >= 3:
                    image_path = parts[1]
                    json_str = parts[2]
                    
                    # JSONデータをパース
                    try:
                        tag_data = json.loads(json_str)
                        tags = tag_data.get('all_tags', [])
                        categorized_tags = tag_data.get('categorized', None)
                        
                        print(f"Processing: {image_path} with categorized tags", file=sys.stderr)
                        print(f"Total tags received: {len(tags)}", file=sys.stderr)
                        if categorized_tags:
                            for category, cat_tags in categorized_tags.items():
                                if cat_tags:
                                    print(f"  {category}: {len(cat_tags)} tags - {cat_tags[:3]}", file=sys.stderr)
                                else:
                                    print(f"  {category}: 0 tags", file=sys.stderr)
                    except json.JSONDecodeError as e:
                        print(f"ERROR:Invalid JSON data: {e}", flush=True)
                        continue
                    
                    # 画像ファイルの存在確認
                    if not os.path.exists(image_path):
                        print(f"ERROR:Image file not found: {image_path}", flush=True)
                        continue
                    
                    # キャプション生成（カテゴリ情報付き）
                    caption = generate_caption_streaming(image_path, DEFAULT_CAPTION_PROMPT, tags, categorized_tags)
                    
                    if caption and not caption.startswith("Error:"):
                        print("FINAL:" + caption, flush=True)
                        
                        # JSONファイルに保存
                        success = save_caption_to_json(image_path, caption)
                        if success:
                            print("SAVED", flush=True)
                        else:
                            print("SAVE_FAILED", flush=True)
                    else:
                        print(f"ERROR:{caption}", flush=True)
                    
            elif line.startswith("PROCESS|"):
                # 旧形式のサポート（後方互換性）
                parts = line.split("|", 2)
                if len(parts) >= 2:
                    image_path = parts[1]
                    tags_str = parts[2] if len(parts) > 2 else ""
                    
                    # タグの解析
                    tags = []
                    if tags_str:
                        tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
                    
                    print(f"Processing (legacy format): {image_path} with tags: {tags}", file=sys.stderr)
                    
                    # 画像ファイルの存在確認
                    if not os.path.exists(image_path):
                        print(f"ERROR:Image file not found: {image_path}", flush=True)
                        continue
                    
                    # キャプション生成（カテゴリ情報なし）
                    caption = generate_caption_streaming(image_path, DEFAULT_CAPTION_PROMPT, tags)
                    
                    if caption and not caption.startswith("Error:"):
                        print("FINAL:" + caption, flush=True)
                        
                        # JSONファイルに保存
                        success = save_caption_to_json(image_path, caption)
                        if success:
                            print("SAVED", flush=True)
                        else:
                            print("SAVE_FAILED", flush=True)
                    else:
                        print(f"ERROR:{caption}", flush=True)
                else:
                    print("ERROR:Invalid command format", flush=True)
            else:
                print("ERROR:Unknown command", flush=True)
                
        except EOFError:
            print("EXITING", flush=True)
            break
        except Exception as e:
            print(f"ERROR:{e}", flush=True)

def main():
    """メイン関数"""
    print("DEBUG: Updated caption_generator.py with --categorized-json support", file=sys.stderr)
    parser = argparse.ArgumentParser(description='GLM-4.1V Caption Generator')
    parser.add_argument('--image', help='Path to image file')
    parser.add_argument('--tags', help='Comma-separated list of tags')
    parser.add_argument('--categorized-json', help='JSON string with categorized tags')
    parser.add_argument('--categorized-json-base64', help='Base64 encoded JSON string with categorized tags')
    parser.add_argument('--prompt', help='Custom prompt template')
    parser.add_argument('--save', action='store_true', help='Save caption to JSON file')
    parser.add_argument('--init-only', action='store_true', help='Only initialize model and exit')
    parser.add_argument('--interactive', action='store_true', help='Start interactive mode for persistent session')
    
    args = parser.parse_args()
    
    # インタラクティブモードの場合
    if args.interactive:
        # モデルを事前に初期化
        success = initialize_glm_model()
        if not success:
            print("INIT_FAILED", flush=True)
            return
        
        # インタラクティブモードに入る
        interactive_mode()
        return
    
    # モデル初期化のみの場合
    if args.init_only:
        success = initialize_glm_model()
        if success:
            print("SUCCESS", flush=True)
        else:
            print("FAILED", flush=True)
        return
    
    # 通常の単発処理モード
    if not args.image:
        print("Error: --image is required for single-shot mode", file=sys.stderr)
        return
    
    # 画像ファイルの存在確認
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}", file=sys.stderr)
        return
    
    # タグの解析とカテゴライズ処理
    tags = []
    categorized_tags = None
    
    if args.categorized_json_base64:
        # Base64エンコードされたカテゴライズ済みJSONデータがある場合
        try:
            import base64
            decoded_bytes = base64.b64decode(args.categorized_json_base64)
            json_str = decoded_bytes.decode('utf-8')
            tag_data = json.loads(json_str)
            tags = tag_data.get('all_tags', [])
            categorized_tags = tag_data.get('categorized', None)
            
            print(f"Single-shot mode with categorized tags (Base64)", file=sys.stderr)
            print(f"Total tags received: {len(tags)}", file=sys.stderr)
            if categorized_tags:
                for category, cat_tags in categorized_tags.items():
                    if cat_tags:
                        print(f"  {category}: {len(cat_tags)} tags - {cat_tags[:3]}", file=sys.stderr)
                    else:
                        print(f"  {category}: 0 tags", file=sys.stderr)
        except Exception as e:
            print(f"Error parsing Base64 categorized JSON: {e}", file=sys.stderr)
            # フォールバック：従来の方式
            if args.tags:
                print(f"Fallback: Processing tags: {repr(args.tags)}", file=sys.stderr)
                tags = [tag.strip() for tag in args.tags.split(',') if tag.strip()]
                print(f"Fallback: Parsed tags: {tags}", file=sys.stderr)
    elif args.categorized_json:
        # カテゴライズ済みJSONデータがある場合（直接JSON）
        try:
            tag_data = json.loads(args.categorized_json)
            tags = tag_data.get('all_tags', [])
            categorized_tags = tag_data.get('categorized', None)
            
            print(f"Single-shot mode with categorized tags (Direct JSON)", file=sys.stderr)
            print(f"Total tags received: {len(tags)}", file=sys.stderr)
            if categorized_tags:
                for category, cat_tags in categorized_tags.items():
                    if cat_tags:
                        print(f"  {category}: {len(cat_tags)} tags - {cat_tags[:3]}", file=sys.stderr)
                    else:
                        print(f"  {category}: 0 tags", file=sys.stderr)
        except json.JSONDecodeError as e:
            print(f"Error parsing categorized JSON: {e}", file=sys.stderr)
            # フォールバック：従来の方式
            if args.tags:
                print(f"Fallback: Processing tags: {repr(args.tags)}", file=sys.stderr)
                tags = [tag.strip() for tag in args.tags.split(',') if tag.strip()]
                print(f"Fallback: Parsed tags: {tags}", file=sys.stderr)
    elif args.tags:
        # 従来の方式
        print(f"Processing tags: {repr(args.tags)}", file=sys.stderr)
        tags = [tag.strip() for tag in args.tags.split(',') if tag.strip()]
        print(f"Parsed tags: {tags}", file=sys.stderr)
    
    # プロンプトの設定
    prompt = args.prompt if args.prompt else DEFAULT_CAPTION_PROMPT
    
    print(f"Generating caption for: {args.image}", file=sys.stderr)
    print(f"Tags: {len(tags)} tags total", file=sys.stderr)
    
    # キャプション生成（カテゴライズ情報付き）
    caption = generate_caption_streaming(args.image, prompt, tags, categorized_tags)
    
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
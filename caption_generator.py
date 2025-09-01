#!/usr/bin/env python3
"""
GLM-4.1V Caption Generator Script
C#アプリケーションから呼び出されるPythonスクリプト
"""

import sys
import os
import json
import argparse
import base64
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

{tag_info}

Provide your answer in <answer></answer> tags."""

# GLM-4V関連の変数
glm_model = None
glm_processor = None
model_lock = threading.Lock()

def initialize_glm_model(gpu_id=None):
    """GLM-4Vモデルを初期化する"""
    global glm_model, glm_processor
    
    if not TRANSFORMERS_AVAILABLE:
        print("Transformers is not available. Skipping model initialization.", file=sys.stderr)
        return False
    
    try:
        with model_lock:
            if glm_model is None:
                print("Initializing GLM-4V model...", file=sys.stderr)
                
                # GPU選択の処理
                selected_device = "cpu"
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    print(f"Available GPUs: {gpu_count}", file=sys.stderr)
                    
                    for i in range(gpu_count):
                        gpu_name = torch.cuda.get_device_name(i)
                        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                        print(f"GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)", file=sys.stderr)
                    
                    # GPU IDの選択
                    if gpu_id is not None:
                        if 0 <= gpu_id < gpu_count:
                            selected_device = f"cuda:{gpu_id}"
                            print(f"Using specified GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}", file=sys.stderr)
                        else:
                            print(f"Warning: GPU {gpu_id} not available, using GPU 0", file=sys.stderr)
                            selected_device = "cuda:0"
                    else:
                        selected_device = "cuda:0"  # デフォルトは最初のGPU
                        print(f"Using default GPU 0: {torch.cuda.get_device_name(0)}", file=sys.stderr)
                    
                    gpu_memory = torch.cuda.get_device_properties(selected_device).total_memory / 1024**3
                    print(f"Selected GPU memory: {gpu_memory:.2f} GB", file=sys.stderr)
                else:
                    print("No CUDA GPUs available, using CPU", file=sys.stderr)
                
                # プロセッサーの初期化
                glm_processor = AutoProcessor.from_pretrained(
                    MODEL_PATH, 
                    use_fast=True
                )
                
                # モデルの初期化
                if selected_device != "cpu":
                    
                    # メモリ効率的な段階的読み込み
                    print("Loading model on CPU first to minimize VRAM usage...", file=sys.stderr)
                    
                    # 1. CPU上でモデルを読み込み（VRAM使用量ゼロ）
                    glm_model = Glm4vForConditionalGeneration.from_pretrained(
                        pretrained_model_name_or_path=MODEL_PATH,
                        torch_dtype=torch.float32,  # 一旦float32で読み込み
                        device_map="cpu",  # 明示的にCPU指定
                        attn_implementation="eager",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    )
                    
                    print("Model loaded on CPU, converting to bfloat16...", file=sys.stderr)
                    
                    # 2. bfloat16に変換（CPU上で実行）
                    glm_model = glm_model.to(dtype=torch.bfloat16)
                    
                    print("Converting to bfloat16 completed, moving to GPU...", file=sys.stderr)
                    
                    # 3. 指定されたGPU に移動（この時点でVRAM使用量が22GB程度になる）
                    glm_model = glm_model.to(device=selected_device)
                    
                    print("Model successfully moved to GPU", file=sys.stderr)
                    
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

def unload_glm_model():
    """GLM-4Vモデルをアンロードして VRAM を解放"""
    global glm_model, glm_processor
    
    try:
        with model_lock:
            if glm_model is not None:
                print("Unloading GLM-4V model...", file=sys.stderr)
                
                # モデルをCPUに移動（VRAMから削除）
                glm_model = glm_model.to('cpu')
                
                # 明示的に削除
                del glm_model
                glm_model = None
                
                print("GLM-4V model unloaded from GPU", file=sys.stderr)
            
            if glm_processor is not None:
                del glm_processor
                glm_processor = None
                print("GLM-4V processor unloaded", file=sys.stderr)
            
            # GPU メモリクリーンアップ
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # GPU操作の完了を待機
                print("GPU memory cleared", file=sys.stderr)
                
            return True
            
    except Exception as e:
        print(f"Error unloading GLM-4V model: {e}", file=sys.stderr)
        return False

# タグカテゴリ関連の辞書読み込み機能は削除（C#側から分類済みデータを受け取る）

def generate_caption_streaming(image_path, prompt, tags, categorized_tags=None, max_tokens=512, temperature=0.7, top_p=0.9):
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
        # 単発モードでは引数から取得（インタラクティブモードでは既に初期化済み）
        gpu_id = getattr(args, 'gpu_id', None) if 'args' in globals() else None
        if not initialize_glm_model(gpu_id=gpu_id):
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
        
        # タグ情報を持つカテゴリのみを含むプロンプトを構築
        print(f"Creating tag info for prompt...", file=sys.stderr)
        tag_info_parts = []
        
        # 各カテゴリをチェックして、タグがあるものだけを追加
        if categorized_tags.get('character'):
            tag_info_parts.append(f"Character: {', '.join(categorized_tags['character'])}")
        if categorized_tags.get('copyright'):
            tag_info_parts.append(f"Copyright: {', '.join(categorized_tags['copyright'])}")
        if categorized_tags.get('general'):
            # generalタグは最初の20個に制限
            tag_info_parts.append(f"General tags: {', '.join(categorized_tags['general'][:20])}")
        if categorized_tags.get('artist'):
            tag_info_parts.append(f"Artist: {', '.join(categorized_tags['artist'])}")
        if categorized_tags.get('rating'):
            tag_info_parts.append(f"Rating: {', '.join(categorized_tags['rating'])}")
        if categorized_tags.get('quality'):
            tag_info_parts.append(f"Quality: {', '.join(categorized_tags['quality'])}")
        if categorized_tags.get('meta'):
            tag_info_parts.append(f"Meta: {', '.join(categorized_tags['meta'])}")
        if categorized_tags.get('model'):
            tag_info_parts.append(f"Model: {', '.join(categorized_tags['model'])}")
        
        # タグ情報を改行で結合
        tag_info = '\n'.join(tag_info_parts) if tag_info_parts else "No tag information available"
        print(f"Tag info created with {len(tag_info_parts)} categories", file=sys.stderr)
        
        # プロンプトをフォーマット
        print(f"Formatting prompt...", file=sys.stderr)
        formatted_prompt = prompt.format(tag_info=tag_info)
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
                'max_new_tokens': max_tokens,
                'do_sample': True,
                'temperature': temperature,
                'top_p': top_p,
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
                print(f"Raw response (length: {len(response_text)}): {repr(response_text)}", file=sys.stderr)
            except Exception as e:
                print(f"Error during generation: {e}", file=sys.stderr)
                print(f"Error type: {type(e)}", file=sys.stderr)
                import traceback
                print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
                raise e
                
            # <answer></answer>タグから内容を抽出（マルチターン会話での継続機能付き）
            import re
            generated_text = ""
            max_retries = 2
            
            # 初回の応答を確認
            answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
            if answer_match:
                generated_text = answer_match.group(1).strip()
                print(f"Extracted from <answer> tags (length: {len(generated_text)}): {repr(generated_text)}", file=sys.stderr)
            else:
                # <answer>タグが見つからない場合、マルチターン会話として続行
                print(f"No <answer> tags found. Continuing conversation to get answer...", file=sys.stderr)
                
                # 会話履歴を構築（初回のやり取り + assistantの応答 + 続きを促すプロンプト）
                # 注意: GLM-4.1Vでは、全てのcontentがlist形式である必要がある
                conversation_history = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": formatted_prompt}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": response_text}  # モデルの初回応答（<think>タグなど含む）
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Please continue and provide your final answer in <answer> tags."}
                        ]
                    }
                ]
                
                # マルチターンでの継続生成を試みる
                for retry in range(max_retries):
                    print(f"Multi-turn continuation attempt {retry + 1}/{max_retries}...", file=sys.stderr)
                    
                    # 継続生成（短めのトークン制限）
                    continuation_config = generation_config.copy()
                    continuation_config['max_new_tokens'] = 256
                    
                    try:
                        with torch.no_grad():
                            continuation_inputs = glm_processor.apply_chat_template(
                                conversation_history,
                                add_generation_prompt=True,
                                tokenize=True,
                                return_tensors="pt",
                                return_dict=True
                            ).to(glm_model.device)
                            
                            continuation_output = glm_model.generate(**continuation_inputs, **continuation_config)
                            continuation_text = glm_processor.batch_decode(
                                continuation_output[:, continuation_inputs['input_ids'].size(1):],
                                skip_special_tokens=True
                            )[0]
                            print(f"Continuation response (length: {len(continuation_text)}): {repr(continuation_text)}", file=sys.stderr)
                            
                            # 継続応答から<answer>タグを探す
                            answer_match = re.search(r'<answer>(.*?)</answer>', continuation_text, re.DOTALL)
                            if answer_match:
                                generated_text = answer_match.group(1).strip()
                                print(f"Successfully extracted from continuation (length: {len(generated_text)}): {repr(generated_text)}", file=sys.stderr)
                                break
                            else:
                                # 継続応答を会話履歴に追加して次のリトライに備える
                                conversation_history.append({
                                    "role": "assistant",
                                    "content": [
                                        {"type": "text", "text": continuation_text}
                                    ]
                                })
                                conversation_history.append({
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": "I need your answer in <answer> tags. Please provide a concise description starting with <answer> and ending with </answer>."}
                                    ]
                                })
                                
                    except Exception as e:
                        print(f"Error during multi-turn continuation: {e}", file=sys.stderr)
                        print(f"Error type: {type(e)}", file=sys.stderr)
                        import traceback
                        print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
                        break
                
                # それでも<answer>タグが得られない場合、response_textから情報を抽出
                if not generated_text:
                    print(f"Multi-turn attempts failed. Extracting from available content...", file=sys.stderr)
                    
                    # まず、部分的な<answer>タグがないか確認（閉じタグがない場合）
                    partial_answer = re.search(r'<answer>([^<]+)', response_text, re.DOTALL)
                    if partial_answer:
                        generated_text = partial_answer.group(1).strip()
                        # 文の途中で切れている場合、最後の完全な文までを取得
                        if generated_text and not generated_text[-1] in '.!?':
                            last_sentence = re.search(r'^(.*[.!?])', generated_text, re.DOTALL)
                            if last_sentence:
                                generated_text = last_sentence.group(1).strip()
                        if generated_text and len(generated_text) > 20:
                            print(f"Extracted from partial <answer> (length: {len(generated_text)}): {repr(generated_text)}", file=sys.stderr)
                    
                    # <think>タグから有用な情報を抽出
                    if not generated_text:
                        think_patterns = [
                            r'The image shows\s+([^.]+\.)',  # "The image shows..."から始まる文
                            r'This is\s+([^.]+\.)',  # "This is..."から始まる文
                            r'A\s+([^.]+(?:character|person|girl|boy|woman|man)[^.]+\.)',  # キャラクター説明
                            r'([^.]{30,150}\.)'  # 任意の完全な文（最初に見つかったもの）
                        ]
                        
                        all_text = response_text  # 初回応答全体から探す
                        for pattern in think_patterns:
                            think_match = re.search(pattern, all_text, re.IGNORECASE | re.DOTALL)
                            if think_match:
                                generated_text = think_match.group(0).strip()
                                if len(generated_text) > 20:
                                    print(f"Extracted from content (length: {len(generated_text)}): {repr(generated_text)}", file=sys.stderr)
                                    break
                    
                    # 最終フォールバック
                    if not generated_text:
                        character_name = ", ".join(categorized_tags['character']) if categorized_tags and categorized_tags['character'] else "character"
                        copyright_name = ", ".join(categorized_tags['copyright']) if categorized_tags and categorized_tags['copyright'] else "anime series"
                        generated_text = f"A character from {copyright_name}."
                        print(f"Using fallback description: {generated_text}", file=sys.stderr)
            
            # Unicodeエスケープシーケンスを正常な文字に変換
            generated_text = generated_text.encode('utf-8').decode('unicode_escape').encode('latin1').decode('utf-8', errors='ignore')
            
            # 追加の文字クリーンアップ
            import unicodedata
            generated_text = unicodedata.normalize('NFKC', generated_text)  # 正規化
            generated_text = generated_text.replace('\u2019', "'")  # 右シングルクォート
            generated_text = generated_text.replace('\u201c', '"')  # 左ダブルクォート
            generated_text = generated_text.replace('\u201d', '"')  # 右ダブルクォート
            generated_text = generated_text.replace('\u2013', '-')  # enダッシュ
            generated_text = generated_text.replace('\u2014', '--')  # emダッシュ
            
            print(f"Cleaned caption (length: {len(generated_text)}): {repr(generated_text)}", file=sys.stderr)
            
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
                # 終了時にモデルをアンロード
                unload_glm_model()
                break
            elif line == "UNLOAD":
                # モデルアンロードコマンド
                success = unload_glm_model()
                if success:
                    print("MODEL_UNLOADED", flush=True)
                else:
                    print("UNLOAD_FAILED", flush=True)
                continue
            
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
    parser.add_argument('--max-tokens', type=int, default=2048, help='Maximum tokens for generation')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for generation')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-p for generation')
    parser.add_argument('--gpu-id', type=int, default=None, help='GPU ID to use (0, 1, 2...). If not specified, uses GPU 0')
    parser.add_argument('--list-gpus', action='store_true', help='List available GPUs and exit')
    parser.add_argument('--system-prompt-base64', help='Base64 encoded system prompt')
    parser.add_argument('--user-prompt-base64', help='Base64 encoded user prompt')
    parser.add_argument('--detailed', action='store_true', help='Generate detailed description')
    
    args = parser.parse_args()
    
    # GPU一覧表示のみの場合
    if args.list_gpus:
        try:
            if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    for i in range(gpu_count):
                        gpu_name = torch.cuda.get_device_name(i)
                        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                        print(f"GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
                else:
                    print("GPU 0: デフォルト (CUDA使用不可)")
            else:
                print("GPU 0: デフォルト (CUDA使用不可)")
        except Exception as e:
            print("GPU 0: デフォルト (GPU検出エラー)")
            print(f"Error: {e}", file=sys.stderr)
        return
    
    # Base64エンコードされたプロンプトをデコード
    system_prompt = None
    user_prompt = None
    
    if args.system_prompt_base64:
        try:
            system_prompt = base64.b64decode(args.system_prompt_base64).decode('utf-8')
        except Exception as e:
            print(f"Error decoding system prompt: {e}", file=sys.stderr)
    
    if args.user_prompt_base64:
        try:
            user_prompt = base64.b64decode(args.user_prompt_base64).decode('utf-8')
        except Exception as e:
            print(f"Error decoding user prompt: {e}", file=sys.stderr)
    
    # インタラクティブモードの場合
    if args.interactive:
        # モデルを事前に初期化（GPU ID指定対応）
        success = initialize_glm_model(gpu_id=args.gpu_id)
        if not success:
            print("INIT_FAILED", flush=True)
            return
        
        # インタラクティブモードに入る
        interactive_mode()
        return
    
    # モデル初期化のみの場合
    if args.init_only:
        success = initialize_glm_model(gpu_id=args.gpu_id)
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
    
    # プロンプト設定の適用
    if user_prompt:
        prompt = user_prompt
    elif args.detailed:
        prompt = "Please provide a detailed and comprehensive description of this image, including all visible elements, colors, composition, mood, and any notable artistic or technical aspects."
    
    # キャプション生成（カテゴライズ情報付き）
    caption = generate_caption_streaming(
        args.image, 
        prompt, 
        tags, 
        categorized_tags,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
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
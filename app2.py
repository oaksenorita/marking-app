import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image
import fitz  # pymupdf
import datetime
import io
import openai
import base64
import json
import zipfile
import os
import re
import shutil
from pathlib import Path
from collections import defaultdict

# ==========================================
# ★設定エリア
# ==========================================
GEMINI_API_KEY_DEFAULT = "" 
GEMINI_MODEL_NAME = "gemini-flash-latest"
OPENAI_MODEL_NAME = "gpt-4o-mini"
USD_JPY_RATE = 155.0
COST_INPUT_PER_1M = 0.15
COST_OUTPUT_PER_1M = 0.60

# デフォルト保存先
DEFAULT_BASE_DIR = r"C:\Users\seory\OneDrive\添削用フォルダ"

# ==========================================
# 初期化・セッション管理
# ==========================================
if "history" not in st.session_state:
    st.session_state.history = []
if "draft_text" not in st.session_state:
    st.session_state.draft_text = ""
if "total_cost_usd" not in st.session_state:
    st.session_state.total_cost_usd = 0.0

if "student_img_cache" not in st.session_state:
    st.session_state.student_img_cache = []
if "ref_img_cache" not in st.session_state:
    st.session_state.ref_img_cache = [] 
if "registry_ref_img_cache" not in st.session_state:
    st.session_state.registry_ref_img_cache = [] 

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "question_registry" not in st.session_state:
    st.session_state.question_registry = {}

if "active_rules" not in st.session_state:
    st.session_state.active_rules = None
if "active_memos" not in st.session_state:
    st.session_state.active_memos = ""
if "pending_overwrite_data" not in st.session_state:
    st.session_state.pending_overwrite_data = None
if "pending_delete_id" not in st.session_state:
    st.session_state.pending_delete_id = None

# デフォルトプロンプト
DEFAULT_SYSTEM_PROMPT = """
あなたは教育的配慮のできる英語教師です。
提示された「生徒の答案」を「基準資料」に基づいて添削・採点してください。

【最重要：添削の心構え】
* コメントは言い方のきつい攻撃的なものには決してせずに、生徒がやる気を出せたり棘のないようなコメントにしてください。
* 添削や採点をしてコメントを追加した答案は生徒の手元に返すということを念頭に置いてください。

【具体的な添削指示】
1. **添削スタイル**:
   - 画像に直接書き込めないため、テキスト上で「下線部(1)〜」のように該当箇所を引用し、番号を振って指摘してください。
   - 各指摘の下に、対応する修正・解説を記述してください。

2. **各ミスの指摘について**:
   - なぜその部分が誤りなのか（理由）
   - どのように訂正すればよいのか（改善案）
   - なぜそう訂正するのか（文法的・文脈的理由）
   上記を丁寧に述べてください。

3. **各問題ごとのコメント**:
   - 間違いの指摘だけでなく、できている点（良い点）も必ず見つけてコメントしてください。

4. **全体の総評**:
   - 最後に、大問を通した総評コメントを記述してください。
   - 全体を通して良かった点・改善点を挙げてください。
   - 今後の学習指針となるアドバイスを提示してください。

出力はMarkdown形式で見やすく整形してください。
"""

# ==========================================
# 関数群: 共通
# ==========================================
def process_uploaded_file(uploaded_file):
    images = []
    if uploaded_file is None:
        return images
    try:
        uploaded_file.seek(0)
        if uploaded_file.type == "application/pdf":
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in doc:
                zoom = 3.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
        else:
            img = Image.open(uploaded_file)
            images.append(img)
    except Exception as e:
        st.error(f"ファイル読み込みエラー: {e}")
    return images

def pil_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def base64_to_pil(base64_str):
    return Image.open(io.BytesIO(base64.b64decode(base64_str)))

def call_ai_hybrid(prompt_text, text_input, images, gemini_key, openai_key, text_label="テキスト情報"):
    try:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        request_content = [prompt_text]
        if text_input:
            request_content.append(f"\n\n【{text_label}】\n{text_input}")
        request_content.extend(images)

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        response = model.generate_content(request_content, safety_settings=safety_settings)
        if response.text:
            return response.text, "Gemini (Free)"
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "Quota" in error_msg or "limit" in error_msg.lower():
            st.warning("⚠️ Gemini制限。OpenAIへ切り替えます...")
        else:
            st.warning(f"⚠️ Geminiエラー({error_msg})。OpenAIへ切り替えます...")

    if not openai_key:
        return "エラー: OpenAI APIキー未設定。", "Error"

    try:
        client = openai.OpenAI(api_key=openai_key)
        messages = [{"role": "system", "content": prompt_text}]
        user_content = []
        if text_input:
            user_content.append({"type": "text", "text": f"【{text_label}】\n{text_input}"})
        else:
             user_content.append({"type": "text", "text": "画像を処理してください。"})

        for img in images:
            b64_str = pil_to_base64(img)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_str}", "detail": "high"}
            })
            
        messages.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME, messages=messages, max_tokens=4000
        )
        result_text = response.choices[0].message.content
        
        usage = response.usage
        cost = (usage.prompt_tokens / 1_000_000 * COST_INPUT_PER_1M) + (usage.completion_tokens / 1_000_000 * COST_OUTPUT_PER_1M)
        st.session_state.total_cost_usd += cost
        return result_text, f"OpenAI ({OPENAI_MODEL_NAME})"

    except Exception as e:
        return f"OpenAI失敗: {e}", "Error"

# ==========================================
# 関数群: 答案仕分け (Auto Sorter v26: 7-8桁対応)
# ==========================================
def parse_ice_table_robust(text):
    """
    ICEのコピーテキストから {生徒コード: [テスト名...]} を作成
    ★変更: 生徒コードを7桁または8桁に対応
    """
    mapping = defaultdict(list)
    lines = text.strip().split('\n')
    
    ignore_patterns = [
        r'\d{4}/\d{2}/\d{2}', 
        r'未対応|対応|完了|添削中|NaN', 
        r'単元ジャンル別演習|過去問演習講座|答案練習講座', 
        r'^\d+$', 
        r'^\d+/\d+$', 
    ]
    
    for line in lines:
        line = line.strip()
        if not line: continue

        # 1. 生徒コード(7または8桁)を探す
        # 日付(2026...)やID(110...)と区別するため、前後に数字がないものを探す
        # AS_ID(9桁)は除外される
        code_matches = list(re.finditer(r'(?<!\d)(\d{7,8})(?!\d)', line))
        
        if not code_matches:
            continue
            
        # 複数ある場合、通常は行の最後尾が生徒コード
        # 例: ...未対応 6193803 NaN
        student_code = code_matches[-1].group(1) 
        
        # 2. テスト名抽出（引き算ロジック）
        parts = re.split(r'\t|\s{2,}| ', line)
        candidate_parts = []
        
        for part in parts:
            part = part.strip()
            if not part: continue
            if part == student_code: continue
            
            is_ignore = False
            for pat in ignore_patterns:
                if re.fullmatch(pat, part):
                    is_ignore = True
                    break
            if re.fullmatch(r'\d{9,}', part): # 9桁以上のIDを除外
                is_ignore = True
                
            if not is_ignore:
                candidate_parts.append(part)
        
        if candidate_parts:
            final_parts = [p for p in candidate_parts if len(p) > 1 or re.match(r'[A-Za-z0-9]', p)]
            test_name = " ".join(final_parts)
            
            if len(test_name) > 3:
                if test_name not in mapping[student_code]:
                    mapping[student_code].append(test_name)

    return mapping

def normalize_folder_name(test_name):
    clean_name = re.sub(r'[\s　]+第\d+回目?', '', test_name)
    return clean_name.strip()

def backup_existing_file(target_path):
    if not target_path.exists():
        return None
    counter = 1
    while True:
        suffix = "_pre" if counter == 1 else f"_pre{counter}"
        backup_name = f"{target_path.stem}{suffix}{target_path.suffix}"
        backup_path = target_path.parent / backup_name
        if not backup_path.exists():
            try:
                target_path.rename(backup_path)
                return backup_name
            except OSError:
                return None
        counter += 1

def save_file_logic(file_bytes, filename, mapping, base_dir, logs):
    """
    ファイル保存ロジック (逆引きマッチング版)
    """
    # ★変更: ファイル名からコードを抽出するのではなく、
    # マッピングにあるコードがファイル名に含まれているかを確認する
    
    target_code = None
    
    for code in mapping.keys():
        # ファイル名がそのコードで終わっているかチェック (.pdfの前)
        # 例: ...06193803.pdf ends with 6193803.pdf -> True
        if filename.endswith(f"{code}.pdf"):
            target_code = code
            break
    
    if not target_code:
        logs.append(f"⚠️ スキップ (一覧にあるコードと一致しません): {filename}")
        return

    tests = mapping[target_code]
    
    # 重複チェック
    if len(tests) > 1:
        normalized_names = set([normalize_folder_name(t) for t in tests])
        if len(normalized_names) > 1:
            manual_folder = base_dir / "_⚠️重複・手動仕分け" / target_code
            manual_folder.mkdir(parents=True, exist_ok=True)
            target_path = manual_folder / f"{target_code}.pdf"
            
            if target_path.exists(): backup_existing_file(target_path)
            
            with open(target_path, "wb") as dest:
                dest.write(file_bytes)
            logs.append(f"⚠️ 重複隔離: {target_code} (複数の異なる問題あり)")
            return

    # 通常処理
    raw_test_name = tests[0]
    folder_test_name = normalize_folder_name(raw_test_name)
    
    # 親フォルダ生成 (英語 の前まで)
    parent_match = re.search(r'^(.*?)(\s+英語|$)', folder_test_name)
    if parent_match:
        parent_name = parent_match.group(1).strip()
    else:
        parent_name = folder_test_name

    target_folder = base_dir / parent_name / folder_test_name
    
    try:
        target_folder.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logs.append(f"❌ フォルダ作成エラー: {e}")
        return
    
    target_path = target_folder / f"{target_code}.pdf"
    
    renamed_backup = None
    if target_path.exists():
        renamed_backup = backup_existing_file(target_path)
    
    with open(target_path, "wb") as dest:
        dest.write(file_bytes)
    
    msg = f"✅ 配置: {target_code} -> {parent_name}/{folder_test_name}"
    if renamed_backup:
        msg += f" (旧: {renamed_backup})"
    logs.append(msg)


def sort_files_zip(zip_file, text_data, base_dir_str):
    logs = []
    path_str = base_dir_str.strip().strip('"').strip("'")
    if path_str.lower() == "desktop":
        base_dir = Path(os.path.expanduser("~/Desktop")) / "Answers"
    else:
        base_dir = Path(os.path.abspath(path_str))
    
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return [f"❌ エラー: 保存先フォルダ異常: {e}"], base_dir

    mapping = parse_ice_table_robust(text_data)
    if not mapping:
        return ["❌ エラー: テスト名と生徒コードを読み取れませんでした。"], base_dir
    
    logs.append(f"📋 {len(mapping)}件の生徒情報を認識")

    try:
        with zipfile.ZipFile(zip_file) as z:
            for filename in z.namelist():
                if not filename.endswith('.pdf'): continue
                with z.open(filename) as source:
                    file_bytes = source.read()
                    save_file_logic(file_bytes, filename, mapping, base_dir, logs)
    except Exception as e:
        return [f"❌ ZIP処理エラー: {e}"], base_dir
        
    return logs, base_dir

def sort_single_file(pdf_file, text_data, base_dir_str):
    logs = []
    path_str = base_dir_str.strip().strip('"').strip("'")
    if path_str.lower() == "desktop":
        base_dir = Path(os.path.expanduser("~/Desktop")) / "Answers"
    else:
        base_dir = Path(os.path.abspath(path_str))
    
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return [f"❌ エラー: 保存先フォルダ異常: {e}"], base_dir

    mapping = parse_ice_table_robust(text_data)
    if not mapping:
        return ["❌ エラー: テキストから情報を読み取れませんでした。"], base_dir
    
    logs.append(f"📋 {len(mapping)}件の情報を認識")

    try:
        file_bytes = pdf_file.read()
        save_file_logic(file_bytes, pdf_file.name, mapping, base_dir, logs)
    except Exception as e:
        return [f"❌ PDF処理エラー: {e}"], base_dir
        
    return logs, base_dir

# ==========================================
# メイン処理
# ==========================================
def main():
    st.set_page_config(page_title="添削くんv26", page_icon="🗂️", layout="wide")
    st.title("🗂️ 添削くん v26 (7桁コード対応版)")

    # --- サイドバー ---
    with st.sidebar:
        st.header("🔑 API設定")
        try:
            default_gemini = st.secrets.get("GEMINI_API_KEY", GEMINI_API_KEY_DEFAULT)
            default_openai = st.secrets.get("OPENAI_API_KEY", "")
        except:
            default_gemini = GEMINI_API_KEY_DEFAULT
            default_openai = ""
        
        gemini_key = st.text_input("Gemini API Key", value=default_gemini, type="password")
        openai_key = st.text_input("OpenAI API Key (予備)", value=default_openai, type="password")
        
        st.divider()
        st.header("📊 Cost")
        st.caption(f"Total: ${st.session_state.total_cost_usd:.4f}")
        
        st.divider()
        st.header("📥 データ管理")
        st.warning("【注意】ブラウザを閉じると登録データは消えます。", icon="⚠️")
        
        if not st.session_state.question_registry:
            json_str = "{}"
        else:
            json_str = json.dumps(st.session_state.question_registry, ensure_ascii=False, indent=2)
            
        st.download_button("設定ファイルを保存 (Export)", json_str, "marking_config.json", "application/json")
        
        uploaded_config = st.file_uploader("設定ファイルを読込 (Import)", type=["json"])
        if uploaded_config is not None:
            if st.button("読み込む"):
                try:
                    data = json.load(uploaded_config)
                    st.session_state.question_registry = data
                    st.success("読み込みました！")
                    st.rerun()
                except Exception as e:
                    st.error(f"読込エラー: {e}")
        
        if st.button("全リセット"):
            st.session_state.clear()
            st.rerun()

        if st.session_state.draft_text and st.session_state.active_memos:
            st.divider()
            st.info("📖 **この問題の採点メモ**")
            st.text_area("参照用", value=st.session_state.active_memos, height=300, disabled=True)

    if not gemini_key or gemini_key == "AIza...":
        st.warning("APIキーを入力してください。")
        return

    tab_sort, tab_mark, tab_reg, tab_hist = st.tabs(["📂 答案仕分け", "📝 採点・添削", "⚙️ 基準データ登録", "🕒 履歴"])

    # ==========================================
    # タブ0: 答案仕分け (v26)
    # ==========================================
    with tab_sort:
        st.subheader("🧹 ICE答案の自動仕分け・保存")
        st.info("ICEからダウンロードしたZIPと表を貼り付けるだけで、あなたのPCのフォルダに自動で振り分けます。")
        
        base_dir_input = st.text_input("保存先の親フォルダ (「Desktop」でデスクトップに作成)", value=DEFAULT_BASE_DIR)
        
        st.markdown("---")
        
        sort_mode = st.radio("モード選択", ["一括 (ZIPファイル)", "個別 (PDF単体)"], horizontal=True)
        
        col_sort1, col_sort2 = st.columns(2)
        
        with col_sort1:
            if sort_mode == "一括 (ZIPファイル)":
                st.markdown("**1. ICEの表をコピペ (全体)**")
            else:
                st.markdown("**1. ICEの行をコピペ (その生徒の行だけ)**")
                
            ice_text = st.text_area("テキスト貼り付け", height=150, placeholder="状態\tCT受付日...\n2026/01/20...")
            
        with col_sort2:
            if sort_mode == "一括 (ZIPファイル)":
                st.markdown("**2. ZIPをアップロード**")
                ice_zip = st.file_uploader("ICEのzipファイル", type=["zip"])
            else:
                st.markdown("**2. PDFをアップロード**")
                ice_pdf = st.file_uploader("生徒のPDFファイル", type=["pdf"])
            
        if st.button("🚀 仕分けを実行する", type="primary"):
            if not ice_text or not base_dir_input:
                st.error("保存先パスとテキスト情報は必須です。")
            elif sort_mode == "一括 (ZIPファイル)" and not ice_zip:
                st.error("ZIPファイルをアップロードしてください。")
            elif sort_mode == "個別 (PDF単体)" and not ice_pdf:
                st.error("PDFファイルをアップロードしてください。")
            else:
                with st.spinner("解析中..."):
                    if sort_mode == "一括 (ZIPファイル)":
                        result = sort_files_zip(ice_zip, ice_text, base_dir_input)
                    else:
                        result = sort_single_file(ice_pdf, ice_text, base_dir_input)
                    
                    if isinstance(result, list) and len(result) > 0 and "❌" in result[0]:
                         st.error(result[0])
                    else:
                        logs, actual_path = result
                        st.success(f"完了！ 保存先: `{actual_path}`")
                        with st.expander("詳細ログ", expanded=True):
                            for log in logs:
                                if "❌" in log: st.error(log)
                                elif "⚠️" in log: st.warning(log)
                                else: st.write(log)

    # ==========================================
    # タブ2: 基準データ登録
    # ==========================================
    with tab_reg:
        st.subheader("1. 新しい問題の基準を登録")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            r_univ = st.text_input("大学名", placeholder="例: 東京大学")
            r_year = st.text_input("年度", placeholder="例: 2025")
        with col_r2:
            r_qnum = st.text_input("大問・問番号", placeholder="例: 大問1 (A)")
            r_files = st.file_uploader("基準画像/PDF (複数可)", type=["jpg","png","pdf"], key="reg_files", accept_multiple_files=True)

        st.markdown("---")
        st.subheader("2. ルール設定")
        col_rule1, col_rule2 = st.columns(2)
        with col_rule1:
            rule_slots = st.number_input("解答欄の数（0なら自動）", min_value=0, value=0)
            rule_ignore_grid = st.checkbox("格子線・枠線を無視する", value=False) 
            rule_ignore_header = st.checkbox("生徒情報ヘッダーを無視", value=True)
        with col_rule2:
            rule_has_word_limit = st.checkbox("語数制限がある設問", help="採点時に手動チェック欄を表示します")
            rule_strict_space = st.checkbox("記述スペース狭小（コメント短め）")
            
        rule_custom = st.text_area("特記事項 (カスタムプロンプト)", placeholder="例: 記号問題なので解説は不要。")
        
        st.markdown("---")
        st.subheader("3. 採点メモ")
        rule_memos = st.text_area("自分用のメモ・コメント集", placeholder="・配点: 10点\n・よくあるミス...\n・コメント例...", height=150)

        if st.button("この内容で登録/更新する", type="primary"):
            if not (r_univ and r_year and r_qnum and r_files):
                st.error("大学名・年度・番号・ファイルは必須です。")
            else:
                unique_id = f"{r_univ}_{r_year}_{r_qnum}"
                if unique_id in st.session_state.question_registry:
                    st.session_state.pending_overwrite_data = {
                        "id": unique_id, "files": r_files,
                        "rules": {"slots": rule_slots, "ignore_grid": rule_ignore_grid, "ignore_header": rule_ignore_header,
                                  "has_word_limit": rule_has_word_limit, "strict_space": rule_strict_space, "custom": rule_custom, "memos": rule_memos},
                        "univ": r_univ, "year": r_year, "q_num": r_qnum
                    }
                    st.rerun()
                else:
                    all_imgs = []
                    for f in r_files:
                        all_imgs.extend(process_uploaded_file(f))
                    b64_imgs = [pil_to_base64(img) for img in all_imgs]

                    st.session_state.question_registry[unique_id] = {
                        "univ": r_univ, "year": r_year, "q_num": r_qnum,
                        "images": b64_imgs,
                        "rules": {"slots": rule_slots, "ignore_grid": rule_ignore_grid, "ignore_header": rule_ignore_header,
                                  "has_word_limit": rule_has_word_limit, "strict_space": rule_strict_space, "custom": rule_custom, "memos": rule_memos}
                    }
                    st.success(f"新規登録しました: {unique_id}")
        
        if st.session_state.pending_overwrite_data:
            st.warning(f"⚠️ データ『{st.session_state.pending_overwrite_data['id']}』は既に存在します。更新しますか？")
            col_conf1, col_conf2 = st.columns(2)
            if col_conf1.button("はい、更新します"):
                data = st.session_state.pending_overwrite_data
                all_imgs = []
                for f in data['files']:
                    all_imgs.extend(process_uploaded_file(f))
                b64_imgs = [pil_to_base64(img) for img in all_imgs]
                st.session_state.question_registry[data['id']] = {
                    "univ": data['univ'], "year": data['year'], "q_num": data['q_num'],
                    "images": b64_imgs, "rules": data['rules']
                }
                st.session_state.pending_overwrite_data = None
                st.success("更新しました！")
                st.rerun()
            if col_conf2.button("キャンセル"):
                st.session_state.pending_overwrite_data = None
                st.rerun()

        if st.session_state.question_registry:
            st.markdown("---")
            st.subheader("📚 登録データの管理・削除")
            reg_keys = list(st.session_state.question_registry.keys())
            target_id = st.selectbox("登録済みデータ一覧", reg_keys)
            if st.button("選択したデータを削除"):
                st.session_state.pending_delete_id = target_id
                st.rerun()

            if st.session_state.pending_delete_id:
                st.error(f"⚠️ 本当に『{st.session_state.pending_delete_id}』を削除しますか？")
                col_del1, col_del2 = st.columns(2)
                if col_del1.button("削除実行"):
                    del st.session_state.question_registry[st.session_state.pending_delete_id]
                    st.session_state.pending_delete_id = None
                    st.success("削除しました。")
                    st.rerun()
                if col_del2.button("やめる"):
                    st.session_state.pending_delete_id = None
                    st.rerun()

    # ==========================================
    # タブ1: 採点作業エリア
    # ==========================================
    with tab_mark:
        current_ref_images_view = []
        if st.session_state.registry_ref_img_cache:
            current_ref_images_view = st.session_state.registry_ref_img_cache
        else:
            current_ref_images_view = st.session_state.ref_img_cache

        if st.session_state.latest_result:
            st.success("🎉 添削完了")
            st.markdown("---")
            st.markdown(st.session_state.latest_result)
            
            st.markdown("---")
            st.subheader("💬 AIへの追加指示・質問")
            with st.form("followup_form"):
                user_q = st.text_area("質問や指示を入力", placeholder="例: 問2の減点理由を詳しく / 問1のスペルミスは見逃して再採点して")
                submitted = st.form_submit_button("送信")
                if submitted and user_q:
                    with st.spinner("AIと思考中..."):
                        context_prompt = f"""
                        あなたは英語教師です。以下の添削結果について、追加の指示に従ってください。
                        【これまでの添削結果】{st.session_state.latest_result}
                        【追加指示】{user_q}
                        """
                        text_res, model_used = call_ai_hybrid(
                            prompt_text=context_prompt, text_input="", 
                            images=current_ref_images_view + st.session_state.student_img_cache, 
                            gemini_key=gemini_key, openai_key=openai_key, text_label="履歴"
                        )
                        new_block = f"\n\n---\n### 💬 追加指示: {user_q}\n\n### 🤖 AI ({model_used})\n{text_res}"
                        st.session_state.latest_result += new_block
                        st.rerun()
            
            if current_ref_images_view:
                with st.expander("📚 基準資料・配点基準を確認する", expanded=False):
                    for i, img in enumerate(current_ref_images_view):
                        st.image(img, caption=f"基準-{i+1}", use_container_width=True)

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            if c1.button("↩️ 修正して再採点", use_container_width=True):
                st.session_state.latest_result = None
                st.rerun()
            if c2.button("➡️ 次の生徒へ (基準維持)", type="primary", use_container_width=True):
                st.session_state.draft_text = ""
                st.session_state.student_img_cache = []
                st.session_state.latest_result = None
                st.session_state.uploader_key += 1
                st.rerun()
            if c3.button("🗑️ 次の問題へ (全クリア)", use_container_width=True):
                st.session_state.draft_text = ""
                st.session_state.student_img_cache = []
                st.session_state.ref_img_cache = []
                st.session_state.registry_ref_img_cache = []
                st.session_state.latest_result = None
                st.session_state.uploader_key += 1
                st.session_state.active_rules = None
                st.session_state.active_memos = ""
                st.rerun()

        elif not st.session_state.draft_text:
            st.subheader("1. 基準データを選択")
            input_mode = st.radio("入力方法", ["登録データから呼び出す", "手動でアップロード"], horizontal=True)
            selected_registry_data = None
            
            if input_mode == "登録データから呼び出す":
                if not st.session_state.question_registry:
                    st.warning("登録データがありません。")
                else:
                    options = ["選択してください"] + list(st.session_state.question_registry.keys())
                    selected_id = st.selectbox("問題を選択", options)
                    if selected_id != "選択してください":
                        data = st.session_state.question_registry[selected_id]
                        selected_registry_data = data
                        st.info(f"選択中: {data['univ']} {data['year']} {data['q_num']}")
                        if not st.session_state.registry_ref_img_cache:
                            imgs = [base64_to_pil(b64) for b64 in data['images']]
                            st.session_state.registry_ref_img_cache = imgs
                        with st.expander("基準画像を確認"):
                            for img in st.session_state.registry_ref_img_cache:
                                st.image(img, use_container_width=True)
            else:
                ref_files = st.file_uploader("基準ファイル", type=["jpg","png","pdf"], key="ref_manual", accept_multiple_files=True)
                if ref_files:
                    st.session_state.ref_img_cache = []
                    for f in ref_files:
                        st.session_state.ref_img_cache.extend(process_uploaded_file(f))

            st.subheader("2. 生徒の答案")
            s_key = f"student_{st.session_state.uploader_key}"
            student_files = st.file_uploader("答案ファイル", type=["jpg","png","pdf"], key=s_key, accept_multiple_files=True)
            if student_files:
                with st.expander("プレビュー", expanded=True):
                    for f in student_files:
                        for img in process_uploaded_file(f):
                            st.image(img, use_container_width=True)

            st.divider()

            if student_files:
                if st.button("① 読み取りを開始 (OCR)", type="primary", use_container_width=True):
                    with st.spinner("ルールに基づいて読み取り中..."):
                        if selected_registry_data:
                            st.session_state.active_rules = selected_registry_data['rules']
                            st.session_state.active_memos = selected_registry_data['rules'].get('memos', "")
                        else:
                            st.session_state.active_rules = None
                            st.session_state.active_memos = ""

                        st.session_state.student_img_cache = []
                        for f in student_files:
                            st.session_state.student_img_cache.extend(process_uploaded_file(f))
                        
                        ocr_prompt = "画像の英文を、スペルミスを含めて忠実にそのままテキスト化してください。解説不要。\n"
                        if selected_registry_data:
                            rules = selected_registry_data['rules']
                            if rules['ignore_grid']:
                                ocr_prompt += "【重要】解答欄の格子線、罫線、枠線は文字として読み取らないでください。\n"
                            if rules['ignore_header']:
                                ocr_prompt += "【重要】ページ上部の氏名・受験番号・点数欄などのヘッダー情報は無視し、解答のみを出力してください。\n"
                            if rules['slots'] > 0:
                                ocr_prompt += f"【重要】設問は(1)〜({rules['slots']})のような形式で{rules['slots']}つあります。それ以外の余計な情報は読み取らないでください。\n"
                        
                        text_res, model_used = call_ai_hybrid(
                            prompt_text=ocr_prompt, text_input="", images=st.session_state.student_img_cache,
                            gemini_key=gemini_key, openai_key=openai_key, text_label="画像"
                        )
                        st.session_state.draft_text = text_res
                        st.rerun()

        # Phase 2: 確認・修正
        else:
            st.info("✅ 読み取り完了。確認してください。")
            current_student_images = st.session_state.student_img_cache

            edit_col, img_col = st.columns([1, 1])
            with edit_col:
                edited_text = st.text_area("テキスト編集", value=st.session_state.draft_text, height=600)
                
                failed_word_limit = False
                if st.session_state.active_rules and st.session_state.active_rules.get('has_word_limit', False):
                    st.markdown("---")
                    st.warning("⚠️ **語数チェック (手動判定)**")
                    failed_word_limit = st.checkbox("語数制限を満たしていない / 大幅な過不足がある (AIに指摘させる)")
                
                if st.button("↩️ 最初から"):
                    st.session_state.draft_text = ""
                    st.session_state.student_img_cache = []
                    st.rerun()

            with img_col:
                tab_s_view, tab_r_view = st.tabs(["🔍 生徒の答案", "📚 基準・配点資料"])
                with tab_s_view:
                    for i, img in enumerate(current_student_images):
                        st.image(img, caption=f"生徒答案-{i+1}", use_container_width=True)
                with tab_r_view:
                    if current_ref_images_view:
                        for i, img in enumerate(current_ref_images_view):
                            st.image(img, caption=f"基準資料-{i+1}", use_container_width=True)
                    else:
                        st.warning("基準資料が読み込まれていません")

            st.divider()
            
            if st.button("② 添削を実行", type="primary", use_container_width=True):
                with st.spinner("ルールに基づいて添削中..."):
                    instruction_prefix = """
                    【⚠️ 重要指示：役割の厳格な区別】
                    1. 以下の「生徒の答案（採点対象）」というテキストのみを採点してください。
                    2. 添付されている画像はすべて「正解データ（基準資料）」です。
                    3. **絶対に画像を採点しないでください。**
                    """
                    final_prompt = instruction_prefix + "\n" + DEFAULT_SYSTEM_PROMPT

                    if st.session_state.active_rules:
                        rules = st.session_state.active_rules
                        if failed_word_limit:
                            final_prompt += "\n【減点指示】生徒の答案は語数制限を満たしていません（または過不足があります）。その旨を指摘し、減点してください。"
                        if rules['strict_space']:
                            final_prompt += "\n【フォーマット指示】記述スペースが狭いため、コメントは簡潔・短めにしてください。"
                        if rules['custom']:
                            final_prompt += f"\n【特記事項】{rules['custom']}"
                    
                    text_res, model_used = call_ai_hybrid(
                        prompt_text=final_prompt,
                        text_input=edited_text,
                        images=current_ref_images_view,
                        gemini_key=gemini_key,
                        openai_key=openai_key,
                        text_label="生徒の答案（採点対象）"
                    )

                    full_result = f"### 📝 修正済み答案\n```text\n{edited_text}\n```\n\n### 🤖 AI ({model_used})\n{text_res}"
                    st.session_state.latest_result = full_result
                    st.session_state.draft_text = edited_text
                    st.rerun()

    with tab_hist:
        if not st.session_state.history:
            st.info("履歴なし")
        else:
            for r in st.session_state.history:
                with st.expander(r['title']):
                    st.markdown(r['result'])

if __name__ == "__main__":
    if "active_rules" not in st.session_state:
        st.session_state.active_rules = None
    main()

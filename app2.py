import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image
import fitz  # pymupdf
import datetime
import io
import openai
import base64

# ==========================================
# ★設定エリア
# ==========================================
GEMINI_API_KEY_DEFAULT = "" 
GEMINI_MODEL_NAME = "gemini-flash-latest"
OPENAI_MODEL_NAME = "gpt-4o-mini"
USD_JPY_RATE = 155.0
COST_INPUT_PER_1M = 0.15
COST_OUTPUT_PER_1M = 0.60

# ==========================================
# 初期化・セッション管理
# ==========================================
if "history" not in st.session_state:
    st.session_state.history = []
if "draft_text" not in st.session_state:
    st.session_state.draft_text = ""
if "total_cost_usd" not in st.session_state:
    st.session_state.total_cost_usd = 0.0

# 画像キャッシュ
if "student_img_cache" not in st.session_state:
    st.session_state.student_img_cache = []
if "ref_img_cache" not in st.session_state:
    st.session_state.ref_img_cache = []

# 直近の採点結果
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

# アップローダー制御用キー
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

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
# 関数
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

def call_ai_hybrid(prompt_text, text_input, images, gemini_key, openai_key, text_label="テキスト情報"):
    """
    text_label引数を追加し、テキストが何を指すのか明示できるように改良
    """
    # 1. Gemini Try
    try:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        
        # プロンプト構築
        request_content = [prompt_text]
        if text_input:
            # ラベルを使って明確に役割を示す
            request_content.append(f"\n\n【{text_label}】\n{text_input}")
        
        # 画像を追加
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
            st.warning("⚠️ Gemini制限発生。OpenAI (gpt-4o-mini) へ切り替えます...")
        else:
            st.warning(f"⚠️ Geminiエラー({error_msg})。OpenAIへ切り替えます...")

    # 2. OpenAI Fallback
    if not openai_key:
        return "エラー: OpenAI APIキーが未設定のためバックアップ起動不可。", "Error"

    try:
        client = openai.OpenAI(api_key=openai_key)
        messages = [{"role": "system", "content": prompt_text}]
        user_content = []
        if text_input:
            user_content.append({"type": "text", "text": f"【{text_label}】\n{text_input}"})
        else:
             user_content.append({"type": "text", "text": "以下の画像を処理してください。"})

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
        return f"OpenAIへの切り替えも失敗しました: {e}", "Error"

# ==========================================
# メイン処理
# ==========================================
def main():
    st.set_page_config(page_title="添削くんv16", page_icon="📝", layout="wide")
    st.title("📝 添削くん v16 (役割誤認修正版)")

    # --- サイドバー設定 ---
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
        st.header("📊 OpenAI コスト")
        cost_usd = st.session_state.total_cost_usd
        col_c1, col_c2 = st.columns(2)
        col_c1.metric("USD", f"${cost_usd:.4f}")
        col_c2.metric("JPY", f"¥{cost_usd * USD_JPY_RATE:.2f}")
        
        st.divider()
        mode = st.radio("モード選択", ["厳密採点（基準資料あり）", "一般添削", "シンプル文字起こし（OCRのみ）"])
        
        if st.button("全履歴・作業クリア"):
            st.session_state.history = []
            st.session_state.draft_text = ""
            st.session_state.student_img_cache = [] 
            st.session_state.ref_img_cache = []
            st.session_state.latest_result = None
            st.session_state.total_cost_usd = 0.0
            st.session_state.uploader_key += 1
            st.rerun()

    if not gemini_key or gemini_key == "AIza...":
        st.warning("APIキーを入力してください。")
        return

    # --- プロンプト編集 ---
    with st.expander("🛠️ プロンプト編集", expanded=False):
        custom_prompt = st.text_area("指示内容", value=DEFAULT_SYSTEM_PROMPT, height=200)

    # --- タブ ---
    tab_main, tab_history = st.tabs(["📝 採点作業", "🕒 採点履歴"])

    # ==========================================
    # タブ1: 作業エリア
    # ==========================================
    with tab_main:
        
        # ----------------------------------------------
        # Phase 3: 結果表示モード
        # ----------------------------------------------
        if st.session_state.latest_result:
            st.success("🎉 添削が完了しました！")
            st.markdown("---")
            st.markdown(st.session_state.latest_result)
            st.markdown("---")
            
            col_act1, col_act2, col_act3 = st.columns([1, 1, 1])
            with col_act1:
                if st.button("↩️ 修正して再採点", use_container_width=True):
                    st.session_state.latest_result = None
                    st.rerun()
            with col_act2:
                if st.button("➡️ 次の生徒へ (基準維持)", type="primary", use_container_width=True):
                    st.session_state.draft_text = ""
                    st.session_state.student_img_cache = []
                    st.session_state.latest_result = None
                    st.session_state.uploader_key += 1
                    st.rerun()
            with col_act3:
                if st.button("🗑️ 次の問題へ (全クリア)", use_container_width=True):
                    st.session_state.draft_text = ""
                    st.session_state.student_img_cache = []
                    st.session_state.ref_img_cache = []
                    st.session_state.latest_result = None
                    st.session_state.uploader_key += 1
                    st.rerun()

        # ----------------------------------------------
        # Phase 1: 初期アップロード画面
        # ----------------------------------------------
        elif not st.session_state.draft_text:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("1. 基準資料")
                if st.session_state.ref_img_cache:
                    st.success(f"📚 基準資料ロード済み ({len(st.session_state.ref_img_cache)}ページ)")
                    with st.expander("基準資料プレビュー"):
                        for img in st.session_state.ref_img_cache:
                            st.image(img, use_container_width=True)
                ref_files = st.file_uploader("基準ファイル (追加・変更)", type=["jpg", "png", "pdf"], key="ref", accept_multiple_files=True)

            with col2:
                st.subheader("2. 生徒の答案")
                student_key = f"student_{st.session_state.uploader_key}"
                student_files = st.file_uploader("答案ファイル", type=["jpg", "png", "pdf"], key=student_key, accept_multiple_files=True)
                if student_files:
                    with st.expander("プレビュー", expanded=True):
                        for f in student_files:
                            for img in process_uploaded_file(f):
                                st.image(img, use_container_width=True)

            st.divider()

            if student_files:
                st.subheader("Step 1: 読み取り開始")
                if st.button("① 読み取りを開始 (OCR)", type="primary", use_container_width=True):
                    with st.spinner("画像を処理中..."):
                        st.session_state.student_img_cache = []
                        for f in student_files:
                            st.session_state.student_img_cache.extend(process_uploaded_file(f))
                        
                        if ref_files:
                            st.session_state.ref_img_cache = []
                            for f in ref_files:
                                st.session_state.ref_img_cache.extend(process_uploaded_file(f))
                        
                        ocr_prompt = "画像の英文を、スペルミスを含めて忠実にそのままテキスト化してください。解説不要。"
                        # OCR時は生徒の答案のみを渡す
                        text_res, model_used = call_ai_hybrid(
                            prompt_text=ocr_prompt,
                            text_input="",
                            images=st.session_state.student_img_cache,
                            gemini_key=gemini_key,
                            openai_key=openai_key,
                            text_label="画像"
                        )
                        st.session_state.draft_text = text_res
                        st.rerun()

        # ----------------------------------------------
        # Phase 2: 確認・修正画面
        # ----------------------------------------------
        else:
            st.info("✅ 読み取り完了。誤りがないか確認・修正してください。")
            current_student_images = st.session_state.student_img_cache
            current_ref_images = st.session_state.ref_img_cache

            edit_col, img_col = st.columns([1, 1])
            with edit_col:
                st.subheader("✏️ テキスト編集")
                edited_text = st.text_area("答案テキスト", value=st.session_state.draft_text, height=600)
                if st.button("↩️ 画像読み込みからやり直す"):
                    st.session_state.draft_text = ""
                    st.session_state.student_img_cache = []
                    st.rerun()

            with img_col:
                st.subheader("🔍 元画像")
                for i, img in enumerate(current_student_images):
                    st.image(img, caption=f"Img {i+1}", use_container_width=True)

            st.divider()
            st.subheader("Step 2: 添削実行")
            
            if st.button("② 添削を実行", type="primary", use_container_width=True):
                if mode == "シンプル文字起こし（OCRのみ）":
                    st.success("完了！")
                    st.session_state.latest_result = f"```text\n{edited_text}\n```"
                    st.rerun()
                else:
                    with st.spinner("AIが添削中..."):
                        
                        # --- 修正ポイント: 役割定義を強化 ---
                        final_prompt = custom_prompt
                        images_to_send = []
                        text_label = "採点対象テキスト"

                        if mode == "厳密採点（基準資料あり）" and current_ref_images:
                            # 厳密採点モードの場合
                            # 1. プロンプトの先頭に強力な注意書きを追加
                            instruction_prefix = """
                            【⚠️ 重要指示：役割の厳格な区別】
                            1. 以下の「生徒の答案（採点対象）」というテキストのみを採点してください。
                            2. 添付されている画像はすべて「正解データ（基準資料）」です。
                            3. **絶対に画像を採点しないでください。** 画像は正解として扱い、テキストと比較するために使ってください。
                            """
                            final_prompt = instruction_prefix + "\n" + custom_prompt
                            
                            # 2. 画像は基準資料のみを送る
                            images_to_send = current_ref_images
                            
                            # 3. テキストラベルを明確化
                            text_label = "生徒の答案（採点対象）"
                            
                        elif mode == "一般添削":
                            final_prompt = "英語講師として、以下のテキストを添削してください。"
                            images_to_send = current_student_images
                            text_label = "生徒の答案テキスト"
                        
                        else: # フォールバック
                            images_to_send = current_student_images

                        # AI呼び出し
                        text_res, model_used = call_ai_hybrid(
                            prompt_text=final_prompt,
                            text_input=edited_text,
                            images=images_to_send,
                            gemini_key=gemini_key,
                            openai_key=openai_key,
                            text_label=text_label # 明確化されたラベルを渡す
                        )

                        timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                        full_result = f"### 📝 修正済み答案\n```text\n{edited_text}\n```\n\n### 🤖 AI ({model_used})\n{text_res}"
                        
                        st.session_state.history.insert(0, {
                            "time": timestamp,
                            "title": f"結果 ({model_used})",
                            "mode": mode,
                            "result": full_result
                        })
                        
                        st.session_state.latest_result = full_result
                        st.session_state.draft_text = edited_text
                        st.rerun()

    with tab_history:
        st.subheader("🕒 採点履歴")
        if not st.session_state.history:
            st.info("履歴はありません。")
        else:
            for record in st.session_state.history:
                with st.expander(f"[{record['time']}] {record['title']}"):
                    st.markdown(record['result'])

if __name__ == "__main__":
    main()
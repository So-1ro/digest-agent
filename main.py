from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import os, json

# ---- .env 読み込み & OpenAI 初期化（キーが無い場合は None にするだけで起動は止めない）----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title="Digest Agent", version="0.1.0")

# ルート（トップページ）
@app.get("/")
def root():
    return {"message": "Digest Agent is running. Try /health or /docs or POST /digest"}

# ヘルスチェック
@app.get("/health")
def health():
    return {"ok": True, "service": "digest-agent", "version": "0.1.0"}

# 入力スキーマ
class Input(BaseModel):
    text: str

# /digest 本番：OpenAIで要約（APIキーが無ければ丁寧にエラーを返す）
@app.post("/digest")
def digest(inp: Input):
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY が設定されていません。.env に OPENAI_API_KEY=sk-... を追加して再起動してください。"
        )

    system = (
        "あなたは日本語の議事録要約アシスタントです。入力テキストから、"
        '次のJSONスキーマに厳密に従って出力してください: {"要点": [""], "ToDo": [""], "提案": [""]}。'
        "各配列は空でもよいが事実ベースで簡潔に。可能なら ToDo に「担当: 」「期日: YYYY-MM-DD」を含めて。"
        "不要な文章は一切出さず、純粋なJSONのみを返してください。"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"テキスト:\n{inp.text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = resp.choices[0].message.content
        data = json.loads(content)

        # 必須キーを補正（念のため）
        for k in ["要点", "ToDo", "提案"]:
            if k not in data or not isinstance(data[k], list):
                data[k] = []

        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Digest failed: {repr(e)}")

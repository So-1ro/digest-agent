
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
    # 環境変数が読めていない/クライアント未初期化の安全チェック
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY が設定されていません。.env ではなく Render の Environment に OPENAI_API_KEY=sk-... を設定し、再デプロイしてください。"
        )

    # モデルへの指示（JSONのみ返す）
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

        content = resp.choices[0].message.content or "{}"

        # --- JSONを読み取り（壊れていたら丁寧にエラー） ---
        try:
            data = json.loads(content)
        except json.JSONDecodeError as je:
            raise HTTPException(
                status_code=502,
                detail=f"LLM出力がJSONとして読み取れませんでした: {content[:200]}..."
            ) from je

        # --- 正規化：配列で来なかった場合や空要素を整える ---
        for k in ("要点", "ToDo", "提案"):
            v = data.get(k, [])
            if isinstance(v, str):
                v = [v]
            if not isinstance(v, list):
                v = []
            data[k] = [str(x).strip() for x in v if str(x).strip()]

        # --- ToDo が辞書の配列でもきれいに1行へ ---
        def to_todo_line(x):
            if isinstance(x, dict):
                title = x.get("内容") or x.get("title") or x.get("task") or ""
                assignee = x.get("担当") or x.get("owner") or x.get("assignee")
                due = x.get("期日") or x.get("due") or x.get("deadline")
                parts = [p for p in [
                    title,
                    f"担当: {assignee}" if assignee else None,
                    f"期日: {due}" if due else None,
                ] if p]
                if parts:
                    return " / ".join(parts)
                return json.dumps(x, ensure_ascii=False)
            return str(x)

        todos_lines = [to_todo_line(x) for x in data["ToDo"]]

        # --- 改行つきの整形テキスト ---
        def bullets(items):
            return "\n".join(f"- {item}" for item in items) or "- なし"

        formatted = (
            "【要点】\n" + bullets(data["要点"]) +
            "\n\n【ToDo】\n" + bullets(todos_lines) +
            "\n\n【提案】\n" + bullets(data["提案"])
        )

        # Notionには formatted を使う。元の配列も欲しければ raw を参照
        return {"raw": data, "formatted": formatted}

    except HTTPException:
        raise
    except Exception as e:
        # 予期しないエラーは500で返す
        raise HTTPException(status_code=500, detail=f"Digest failed: {repr(e)}")

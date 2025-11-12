
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
            detail=(
                "OPENAI_API_KEY が設定されていません。"
                ".env ではなく Render の Environment に "
                "OPENAI_API_KEY=sk-... を設定し、再デプロイしてください。"
            ),
        )

    # --- モデルへの指示（JSONのみ返す） ---
    # ToDo には必ず「タスク内容 / 担当: ◯◯ / 期日: YYYY-MM-DD」を含めるように明示
    system = (
        "あなたは日本語の議事録要約アシスタントです。入力テキストから、"
        '次のJSONスキーマに厳密に従って出力してください: '
        '{"要点": [""], "ToDo": [""], "提案": [""]}。'
        "各配列は空でもよいが、事実ベースで簡潔に書いてください。"
        "ToDo の各要素は必ず『何をするか / 担当: ◯◯ / 期日: YYYY-MM-DD』という1行テキストにしてください。"
        "例: 『議事録を共有する / 担当: 佐藤さん / 期日: 2023-11-20』。"
        "タスク内容（何をするか）は省略せず、会議で決まった具体的なアクションを書いてください。"
        "不要な文章は一切出さず、純粋なJSONのみを返してください。"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"テキスト:\n{inp.text}"},
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
                detail=f"LLM出力がJSONとして読み取れませんでした: {content[:200]}...",
            ) from je

        # --- 正規化：要点 / 提案 は「文字列の配列」に揃える ---
        for k in ("要点", "提案"):
            v = data.get(k, [])
            if isinstance(v, str):
                v = [v]
            if not isinstance(v, list):
                v = []
            data[k] = [str(x).strip() for x in v if str(x).strip()]

        # --- ToDo は文字列・辞書どちらでも受けて「1行テキスト」に揃える ---
        todos_raw = data.get("ToDo", [])

        # まず配列に揃える
        if isinstance(todos_raw, str):
            todos_raw = [todos_raw]
        if not isinstance(todos_raw, list):
            todos_raw = []

        def to_todo_line(x):
            """辞書でも文字列でも、必ず
            『タスク内容 / 担当: ◯◯ / 期日: YYYY-MM-DD』
            の形に近づけるための正規化関数
            """
            # dict の場合（将来スキーマを辞書に変えても対応できるようにしておく）
            if isinstance(x, dict):
                title = (
                    x.get("内容")
                    or x.get("タスク")
                    or x.get("task")
                    or x.get("title")
                    or ""
                )
                assignee = x.get("担当") or x.get("owner") or x.get("assignee")
                due = x.get("期日") or x.get("due") or x.get("deadline")

                # 内容が空なら、担当・期日などから仮タイトルを作る
                if not title:
                    parts_for_title = [x.get("メモ"), x.get("備考"), assignee]
                    parts_for_title = [p for p in parts_for_title if p]
                    title = " / ".join(parts_for_title) or "内容未記載"

                parts = [
                    title,
                    f"担当: {assignee}" if assignee else None,
                    f"期日: {due}" if due else None,
                ]
                parts = [p for p in parts if p]
                return " / ".join(parts)

            # 文字列の場合は、そのまま使いつつ最低限のチェックだけ
            s = str(x).strip()
            if not s:
                return ""
            # 「担当」「期日」が含まれていない場合は、ざっくり補足だけ入れておく
            if "担当:" not in s and "担当：" not in s:
                s = s + " / 担当: 未設定"
            if "期日:" not in s and "期日：" not in s:
                s = s + " / 期日: 未設定"
            return s

        todos_lines = [to_todo_line(x) for x in todos_raw]
        todos_lines = [t for t in todos_lines if t]  # 空行は落とす

        # raw の ToDo も、Zapier から扱いやすいように整形済み文字列に置き換える
        data["ToDo"] = todos_lines

        # --- 改行つきの整形テキスト ---
        def bullets(items):
            return "\n".join(f"- {item}" for item in items) or "- なし"

        formatted = (
            "【要点】\n"
            + bullets(data["要点"])
            + "\n\n【ToDo】\n"
            + bullets(todos_lines)
            + "\n\n【提案】\n"
            + bullets(data["提案"])
        )

        # Zapier / Notion には formatted を使う。配列が欲しければ raw を参照
        return {"raw": data, "formatted": formatted}

    except HTTPException:
        # すでに意味のあるHTTPExceptionならそのまま投げ直す
        raise
    except Exception as e:
        # 予期しないエラーは500で返す
        raise HTTPException(status_code=500, detail=f"Digest failed: {repr(e)}")

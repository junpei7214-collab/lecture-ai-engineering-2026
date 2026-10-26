"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask            # 対話モード（exit / quit で終了）
    uv run python -m app.ask "LoRAとは何ですか？"

対話モードでは会話履歴を保持する。ツールは持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai.chats import Chat
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, chat: Chat | None = None) -> str:
    """質問に回答する。chat が指定されると、その会話の履歴を引き継ぐ。"""
    client = None
    if chat is None:
        client = genai.Client(api_key=get_api_key())
        chat = client.chats.create(model=MODEL)

    try:
        response = chat.send_message(question)
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count if usage is not None else "不明"
        output_tokens = usage.candidates_token_count if usage is not None else "不明"
        print(f"[トークン] 入力: {input_tokens}, 出力: {output_tokens}")

        return response.text
    finally:
        if client is not None:
            client.close()


def run_interactive(chat: Chat) -> None:
    """同じチャットを使って質問を受け付け、会話履歴を保ったまま回答する。"""
    print("質問を入力してください（exit / quit で終了）。")
    while True:
        try:
            question = input("質問> ").strip()
        except EOFError:
            print()
            break

        if question.casefold() in {"exit", "quit"}:
            break
        if not question:
            continue

        print(f"[質問] {question}")
        try:
            print(f"[回答] {ask(question, chat)}\n")
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    if args.question is None:
        client = genai.Client(api_key=get_api_key())
        try:
            chat = client.chats.create(model=MODEL)
            run_interactive(chat)
        finally:
            client.close()
        return

    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

from types import SimpleNamespace

import pytest

from app import ask as ask_module


class FakeChat:
    def __init__(self):
        self.questions = []

    def send_message(self, question):
        self.questions.append(question)
        return SimpleNamespace(
            text=f"回答: {question}",
            usage_metadata=SimpleNamespace(
                prompt_token_count=1,
                candidates_token_count=2,
            ),
        )


class FakeChats:
    def __init__(self, chat):
        self.chat = chat

    def create(self, model):
        assert model == ask_module.MODEL
        return self.chat


@pytest.mark.parametrize("exit_command", ["exit", "quit", "EXIT", "Quit"])
def test_interactive_keeps_history_and_stops_on_exit(monkeypatch, capsys, exit_command):
    chat = FakeChat()
    client = SimpleNamespace(chats=FakeChats(chat), close=lambda: None)

    def make_client(api_key):
        assert api_key == "test-key"
        return client

    monkeypatch.setattr(ask_module.genai, "Client", make_client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(
        "sys.argv",
        ["app.ask"],
    )
    answers = iter(["最初の質問", "", "続きの質問", exit_command])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    ask_module.main()

    output = capsys.readouterr().out
    assert chat.questions == ["最初の質問", "続きの質問"]
    assert "[回答] 回答: 最初の質問" in output
    assert "[回答] 回答: 続きの質問" in output


def test_ask_uses_provided_chat_and_prints_token_usage(capsys):
    chat = FakeChat()

    answer = ask_module.ask("LoRAとは？", chat)

    assert answer == "回答: LoRAとは？"
    assert chat.questions == ["LoRAとは？"]
    assert "[トークン] 入力: 1, 出力: 2" in capsys.readouterr().out

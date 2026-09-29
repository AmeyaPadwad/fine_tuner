import sys
from pathlib import Path

# tests/ sits directly under the project root; make the root importable as
# `src.train....` without needing an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeTokenizer:
    """Minimal stand-in for an HF chat tokenizer, good enough to exercise
    dataset.py's masking logic with no torch/transformers installed.

    Tokenizes by whitespace-splitting a flattened "role: content" rendering
    of the conversation and mapping each unique word to a stable integer id,
    so `apply_chat_template(turns)` and
    `apply_chat_template(turns[:-1], add_generation_prompt=True)` agree on
    ids for the shared prefix — the property build_masked_example relies on
    to find the assistant span.
    """

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def _id_for(self, word: str) -> int:
        if word not in self.vocab:
            self.vocab[word] = len(self.vocab) + 1
        return self.vocab[word]

    def _render(self, conversation: list[dict], add_generation_prompt: bool) -> str:
        parts = [f"<{t['role']}>: {t['content']}" for t in conversation]
        if add_generation_prompt:
            parts.append("<assistant>:")
        return " ".join(parts)

    def apply_chat_template(self, conversation, tokenize=True, add_generation_prompt=False):
        text = self._render(conversation, add_generation_prompt)
        words = text.split()
        ids = [self._id_for(w) for w in words]
        return ids if tokenize else text

    def __call__(self, text: str, add_special_tokens: bool = True):
        return {"input_ids": [self._id_for(w) for w in text.split()]}

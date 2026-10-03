import pytest

from lightrag.operate import chunking_by_token_size


class CharacterTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


@pytest.mark.parametrize("suffix", [".md", ".txt", ".json", ".html"])
def test_generic_text_formats_produce_chunks(suffix: str) -> None:
    content = "故障原因与改进措施" * 20

    chunks = chunking_by_token_size(
        file_path=f"report{suffix}",
        tokenizer=CharacterTokenizer(),
        content=content,
        max_token_size=40,
        overlap_token_size=10,
    )

    assert len(chunks) > 1
    assert all(chunk["content"] for chunk in chunks)
    assert all(chunk["tokens"] <= 40 for chunk in chunks)

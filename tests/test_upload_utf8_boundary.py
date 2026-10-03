from lightrag.product_accounts import content_matches_extension


def test_text_upload_accepts_utf8_codepoint_split_at_sample_boundary() -> None:
    payload = ("a" * 63 + "文程的三个产品").encode("utf-8")

    assert content_matches_extension("portfolio.md", payload)

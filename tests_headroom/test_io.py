import json

from kvheadroom.io import read_jsonl


def test_read_jsonl_preserves_unicode_line_separator(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [{"question": "before\u2028after", "answer": "D"}, {"question": "plain"}]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")

    assert read_jsonl(path) == rows

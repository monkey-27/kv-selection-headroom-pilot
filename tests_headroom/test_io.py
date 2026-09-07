import json

import kvheadroom.io as io
from kvheadroom.io import read_jsonl


def test_read_jsonl_preserves_unicode_line_separator(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [{"question": "before\u2028after", "answer": "D"}, {"question": "plain"}]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")

    assert read_jsonl(path) == rows


def test_append_jsonl_can_defer_remote_commit(tmp_path, monkeypatch):
    commits = []
    monkeypatch.setattr(io, "commit_remote_volume", lambda: commits.append(True))
    path = tmp_path / "rows.jsonl"

    io.append_jsonl(path, [{"mask": 1}], commit=False)
    assert commits == []
    assert read_jsonl(path) == [{"mask": 1}]

    io.append_jsonl(path, [{"mask": 2}])
    assert commits == [True]

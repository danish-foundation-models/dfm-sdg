from __future__ import annotations

import pytest

from sdg.cli import main
from sdg.commons import Artifact, store
from sdg.commons import publish as common_publish


class FakeDataset:
    def __init__(self) -> None:
        self.num_rows = 2
        self.column_names = ["prompt", "target"]
        self.push_calls: list[dict[str, object]] = []

    def push_to_hub(
        self,
        repo_id: str,
        *,
        split: str,
        private: bool,
        commit_message: str | None,
    ) -> None:
        self.push_calls.append(
            {
                "repo_id": repo_id,
                "split": split,
                "private": private,
                "commit_message": commit_message,
            }
        )


def test_upload_dataset_artifact_pushes_named_split(tmp_path, monkeypatch) -> None:
    path = tmp_path / "dataset.jsonl"
    store.write_jsonl(
        [
            {"prompt": "one", "target": "uno"},
            {"prompt": "two", "target": "dos"},
        ],
        path,
    )
    artifact = Artifact(name="dataset", path=str(path), kind="jsonl", meta={})
    fake_dataset = FakeDataset()

    monkeypatch.setattr(common_publish, "_load_hub_dataset", lambda source, split: fake_dataset)

    payload = common_publish.upload_dataset_artifact(
        artifact,
        repo_id="synquid/wiki-instruct-da",
        split="train",
        private=True,
        commit_message="Upload dataset",
    )

    assert fake_dataset.push_calls == [
        {
            "repo_id": "synquid/wiki-instruct-da",
            "split": "train",
            "private": True,
            "commit_message": "Upload dataset",
        }
    ]
    assert payload == {
        "repo_id": "synquid/wiki-instruct-da",
        "split": "train",
        "private": True,
        "source_path": str(path),
        "rows": 2,
        "columns": ["prompt", "target"],
    }


def test_upload_dataset_artifact_rejects_unsupported_suffix(tmp_path) -> None:
    path = tmp_path / "dataset.txt"
    path.write_text("hello\n")

    with pytest.raises(ValueError, match=r"supports \.jsonl, \.json, and \.parquet"):
        common_publish._load_hub_dataset(path, split="train")


def test_upload_hf_cli_uploads_named_artifact(monkeypatch, capsys) -> None:
    artifact = Artifact(name="dataset", path="/tmp/dataset.jsonl", kind="jsonl", meta={})
    result = type("FakeResult", (), {"run_id": "run-123", "pack": "backtranslation", "artifacts": {"dataset": artifact}})()
    captured: dict[str, object] = {}

    monkeypatch.setattr("sdg.cli.load", lambda target: result)

    def fake_upload(artifact_ref, *, repo_id, split, private, commit_message):
        captured.update(
            {
                "artifact_path": artifact_ref.path,
                "repo_id": repo_id,
                "split": split,
                "private": private,
                "commit_message": commit_message,
            }
        )
        return {"repo_id": repo_id, "split": split, "private": private, "rows": 2, "columns": ["prompt", "target"]}

    monkeypatch.setattr(common_publish, "upload_dataset_artifact", fake_upload)

    exit_code = main(
        [
            "upload-hf",
            "run-123",
            "--artifact",
            "dataset",
            "--repo",
            "synquid/wiki-instruct-da",
            "--split",
            "train",
            "--private",
            "--commit-message",
            "Upload dataset",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "artifact_path": "/tmp/dataset.jsonl",
        "repo_id": "synquid/wiki-instruct-da",
        "split": "train",
        "private": True,
        "commit_message": "Upload dataset",
    }
    assert '"artifact": "dataset"' in capsys.readouterr().out

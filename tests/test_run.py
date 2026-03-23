from __future__ import annotations

from pathlib import Path

from sdg.commons.run import load, progress, run_status
from sdg.commons.utils import read_json, read_yaml
from sdg.packs.demo.build import build


def test_run_status_derives_completed_state_from_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))

    cfg_path = Path(__file__).resolve().parents[1] / "sdg" / "packs" / "demo" / "configs" / "base.yaml"
    cfg = read_yaml(cfg_path)

    result = build(cfg)
    manifest = read_json(Path(result.run_dir) / "manifest.json")

    assert "status" not in manifest
    assert run_status(manifest) == "completed"
    assert load(result.run_id).status == "completed"
    assert progress(result.run_id)["status"] == "completed"


def test_run_status_supports_derived_and_legacy_manifests() -> None:
    assert run_status({}) == "running"
    assert run_status({"finished_at": "2026-03-23T10:00:00Z"}) == "completed"
    assert run_status({"finished_at": "2026-03-23T10:00:00Z", "error": {"message": "boom"}}) == "failed"
    assert run_status({"status": "running"}) == "running"
    assert run_status({"status": "completed"}) == "completed"
    assert run_status({"status": "failed"}) == "failed"

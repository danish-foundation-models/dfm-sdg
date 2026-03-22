from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from sdg.commons.utils import packs_root, read_yaml


@dataclass(frozen=True)
class PackSpec:
    name: str
    root_dir: Path
    manifest: dict[str, Any]
    build: Callable[..., Any]
    verify: Callable[..., Any]
    summarize: Callable[..., Any]
    publish: Callable[..., Any]
    viewer: Callable[..., Any] | None


def list_packs() -> list[str]:
    return sorted(path.parent.name for path in packs_root().glob("*/pack.yaml"))


def load_pack(name: str) -> PackSpec:
    pack_dir = packs_root() / name
    manifest_path = pack_dir / "pack.yaml"
    if not manifest_path.exists():
        raise ValueError(f"Unknown pack: {name}")

    manifest = read_yaml(manifest_path)
    entrypoints = manifest.get("entrypoints", {})

    return PackSpec(
        name=manifest["name"],
        root_dir=pack_dir,
        manifest=manifest,
        build=_load_callable(entrypoints["build"]),
        verify=_load_callable(entrypoints["verify"]),
        summarize=_load_callable(entrypoints["summarize"]),
        publish=_load_callable(entrypoints["publish"]),
        viewer=_load_optional_callable(entrypoints.get("viewer")),
    )


def find_pack_for_path(path: str | Path) -> PackSpec:
    current = Path(path).expanduser().resolve()
    for candidate in (current, *current.parents):
        manifest_path = candidate / "pack.yaml"
        if manifest_path.exists():
            manifest = read_yaml(manifest_path)
            return load_pack(manifest["name"])
    raise ValueError(f"Could not infer pack from path: {path}")


def _load_callable(target: str) -> Callable[..., Any]:
    module_name, func_name = target.split(":")
    module = import_module(module_name)
    return getattr(module, func_name)


def _load_optional_callable(target: str | None) -> Callable[..., Any] | None:
    if not target:
        return None
    return _load_callable(target)

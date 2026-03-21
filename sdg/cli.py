from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sdg.commons.registry import find_pack_for_path, list_packs, load_pack
from sdg.commons.run import compare, load, read_events
from sdg.commons.utils import read_yaml


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "build":
        return _run_build(args.config)

    if args.command == "verify":
        return _run_verify(args.target)

    if args.command == "summarize":
        return _run_summarize(args.target)

    if args.command == "publish":
        return _run_publish(args.target, args.out_dir)

    if args.command == "compare":
        return _run_compare(args.left, args.right)

    if args.command == "events":
        return _run_events(args.target, component=args.component, limit=args.limit)

    if args.command == "list-packs":
        for pack_name in list_packs():
            print(pack_name)
        return 0

    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sdg")
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("config")

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("target")

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("target")

    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("target")
    publish_parser.add_argument("--out-dir")

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("left")
    compare_parser.add_argument("right")

    events_parser = subparsers.add_parser("events")
    events_parser.add_argument("target")
    events_parser.add_argument("--component")
    events_parser.add_argument("--limit", type=int)

    subparsers.add_parser("list-packs")

    return parser


def _run_build(config_path: str) -> int:
    cfg_path = Path(config_path).expanduser().resolve()
    cfg = read_yaml(cfg_path)
    pack = load_pack(cfg.get("pack")) if cfg.get("pack") else find_pack_for_path(cfg_path)
    result = pack.build(cfg)
    _print_json(
        {
            "pack": result.pack,
            "run_id": result.run_id,
            "run_dir": result.run_dir,
            "status": result.status,
            "spec_hash": result.spec_hash,
            "artifacts": {
                name: {"path": artifact.path, "kind": artifact.kind}
                for name, artifact in result.artifacts.items()
            },
        }
    )
    return 0


def _run_verify(target: str) -> int:
    result = load(target)
    pack = load_pack(result.pack)
    _print_json(pack.verify(target))
    return 0


def _run_summarize(target: str) -> int:
    result = load(target)
    pack = load_pack(result.pack)
    _print_json(pack.summarize(target))
    return 0


def _run_publish(target: str, out_dir: str | None) -> int:
    result = load(target)
    pack = load_pack(result.pack)
    _print_json(pack.publish(target, out_dir=out_dir))
    return 0


def _run_compare(left: str, right: str) -> int:
    _print_json(compare(left, right))
    return 0


def _run_events(target: str, *, component: str | None, limit: int | None) -> int:
    _print_json({"events": read_events(target, component=component, limit=limit)})
    return 0


def _print_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))

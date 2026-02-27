#!/usr/bin/env python3
"""Install Copilot CLI custom agents globally (user-level).

Copies this repo's generated agent profiles from:
  .github/agents/*.agent.md
into the user-level Copilot agents directory:
  ~/.copilot/agents/

This makes the agents available in *all* folders on this machine.

Usage:
  python install_copilot_agents.py
  python install_copilot_agents.py --dry-run
  python install_copilot_agents.py --src .github/agents --dest ~/.copilot/agents
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def _expand_path(path_str: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(path_str))
    return Path(expanded)


def install_agents(*, src_dir: Path, dest_dir: Path, dry_run: bool) -> list[tuple[Path, Path]]:
    if not src_dir.exists() or not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    agent_files = sorted(src_dir.glob("*.agent.md"))
    if not agent_files:
        raise FileNotFoundError(f"No '*.agent.md' files found in: {src_dir}")

    copied: list[tuple[Path, Path]] = []
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)

    for src in agent_files:
        dest = dest_dir / src.name
        copied.append((src, dest))
        if dry_run:
            continue
        shutil.copy2(src, dest)

    return copied


def main() -> int:
    repo_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Install Copilot CLI custom agents globally (user-level).",
    )
    parser.add_argument(
        "--src",
        default=str(repo_root / ".github" / "agents"),
        help="Directory containing *.agent.md files (default: .github/agents in this repo)",
    )
    parser.add_argument(
        "--dest",
        default="~/.copilot/agents",
        help="Destination directory (default: ~/.copilot/agents)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without modifying anything",
    )

    args = parser.parse_args()

    src_dir = _expand_path(args.src)
    dest_dir = _expand_path(args.dest)

    copied = install_agents(src_dir=src_dir, dest_dir=dest_dir, dry_run=args.dry_run)

    if args.dry_run:
        print("Dry run. Would copy:")
    else:
        print("Installed Copilot CLI agents:")

    for src, dest in copied:
        print(f"- {src} -> {dest}")

    if not args.dry_run:
        print("\nNext:")
        print("- Restart 'copilot' (or use /agent to refresh the agent list)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

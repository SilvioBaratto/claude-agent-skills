#!/usr/bin/env python3
"""Install Copilot CLI skills globally (user-level).

Copies this repo's skills from:
  .github/skills/*
into the user-level Copilot skills directory:
  ~/.copilot/skills/

Usage:
  python install_copilot_skills.py
  python install_copilot_skills.py --dry-run
  python install_copilot_skills.py --src .github/skills --dest ~/.copilot/skills
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def _expand_path(path_str: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(path_str))
    return Path(expanded)


def _is_skill_dir(dir_path: Path) -> bool:
    return dir_path.is_dir() and (dir_path / "SKILL.md").is_file()


def install_skills(*, src_dir: Path, dest_dir: Path, dry_run: bool) -> list[tuple[Path, Path]]:
    if not src_dir.exists() or not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    skill_dirs = sorted(p for p in src_dir.iterdir() if _is_skill_dir(p))
    if not skill_dirs:
        raise FileNotFoundError(f"No skill directories (containing SKILL.md) found in: {src_dir}")

    copied: list[tuple[Path, Path]] = []
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)

    for src_skill_dir in skill_dirs:
        dest_skill_dir = dest_dir / src_skill_dir.name
        copied.append((src_skill_dir, dest_skill_dir))
        if dry_run:
            continue

        shutil.copytree(src_skill_dir, dest_skill_dir, dirs_exist_ok=True)

    return copied


def main() -> int:
    repo_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Install Copilot CLI skills globally (user-level).",
    )
    parser.add_argument(
        "--src",
        default=str(repo_root / ".github" / "skills"),
        help="Directory containing skill subfolders (default: .github/skills in this repo)",
    )
    parser.add_argument(
        "--dest",
        default="~/.copilot/skills",
        help="Destination directory (default: ~/.copilot/skills)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without modifying anything",
    )

    args = parser.parse_args()

    src_dir = _expand_path(args.src)
    dest_dir = _expand_path(args.dest)

    copied = install_skills(src_dir=src_dir, dest_dir=dest_dir, dry_run=args.dry_run)

    if args.dry_run:
        print("Dry run. Would copy:")
    else:
        print("Installed Copilot CLI skills:")

    for src, dest in copied:
        print(f"- {src} -> {dest}")

    if not args.dry_run:
        print("\nNext:")
        print("- Restart 'copilot' (or refresh skills in your Copilot CLI session)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

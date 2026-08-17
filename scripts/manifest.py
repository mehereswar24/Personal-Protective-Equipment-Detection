"""Record what each model checkpoint is, so weights stay identifiable.

The checkpoints are correctly gitignored (they are hundreds of MB), but that
also means nothing in the repository says which code revision produced them,
what they scored, or whether the file on disk is the one the logs describe.
This writes models/MANIFEST.json with a checksum and metadata per checkpoint,
and that manifest IS tracked.

Usage:
    python scripts/manifest.py            # write/refresh the manifest
    python scripts/manifest.py --verify   # check on-disk files still match
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
MANIFEST = MODELS / "MANIFEST.json"
CHUNK = 1024 * 1024


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def torch_versions() -> dict:
    try:
        import torch
        import torchvision
        return {"torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "cuda": torch.version.cuda}
    except Exception:
        return {}


def checkpoints() -> list[Path]:
    if not MODELS.exists():
        return []
    return sorted(p for p in MODELS.rglob("*")
                  if p.suffix in (".pth", ".pt") and p.is_file())


def describe(path: Path) -> dict:
    stat = path.stat()
    entry = {
        "path": path.relative_to(MODELS).as_posix(),
        "bytes": stat.st_size,
        "modified": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256(path),
    }
    # Checkpoints often carry their own epoch/score metadata.
    try:
        import torch
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(blob, dict):
            for key in ("epoch", "best_map", "best_loss", "val_loss",
                        "map", "score", "num_classes", "anchors"):
                if key in blob:
                    value = blob[key]
                    entry[key] = (value if isinstance(value, (int, float, str))
                                  else str(value))
    except Exception as e:
        entry["inspect_error"] = f"{type(e).__name__}: {e}"
    return entry


def build() -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "environment": torch_versions(),
        "note": "Checkpoints are gitignored. This manifest identifies them; "
                "keep an off-machine backup of models/ - they are the "
                "expensive artifact and the only irreplaceable one.",
        "checkpoints": [describe(p) for p in checkpoints()],
    }


def verify() -> int:
    if not MANIFEST.exists():
        print("No manifest yet. Run without --verify first.")
        return 1
    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    known = {c["path"]: c for c in recorded.get("checkpoints", [])}
    on_disk = {p.relative_to(MODELS).as_posix(): p for p in checkpoints()}

    problems = 0
    for name, entry in known.items():
        path = on_disk.pop(name, None)
        if path is None:
            print(f"MISSING  {name}")
            problems += 1
            continue
        actual = sha256(path)
        if actual != entry["sha256"]:
            print(f"CHANGED  {name}\n         recorded {entry['sha256'][:16]}…"
                  f"\n         on disk  {actual[:16]}…")
            problems += 1
        else:
            print(f"ok       {name}")
    for name in on_disk:
        print(f"UNTRACKED {name} (not in the manifest - regenerate it)")
        problems += 1

    print(f"\n{problems} problem(s)")
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="check on-disk checkpoints against the manifest")
    args = parser.parse_args()

    if args.verify:
        return verify()

    found = checkpoints()
    if not found:
        print(f"No checkpoints under {MODELS}")
        return 1
    print(f"Hashing {len(found)} checkpoint(s)…")
    MANIFEST.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

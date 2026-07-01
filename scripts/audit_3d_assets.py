#!/usr/bin/env python3
"""Audit local mesh sources and web 3D asset manifest for AutoDex gallery."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


MESH_EXTS = {".obj", ".ply", ".stl", ".glb", ".gltf"}


def find_meshes(root: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in MESH_EXTS:
            continue
        rel = path.relative_to(root)
        obj = rel.parts[1] if len(rel.parts) > 1 and rel.parts[0] == "paradex" else rel.parts[0]
        out.setdefault(obj, []).append(path)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mesh-root",
        default=str(Path.home() / "shared_data/AutoDex/object"),
        help="Root containing object meshes, e.g. ~/shared_data/AutoDex/object.",
    )
    parser.add_argument(
        "--manifest",
        default="docs/assets3d.json",
        help="Gallery 3D asset manifest to inspect.",
    )
    args = parser.parse_args()

    mesh_root = Path(args.mesh_root).expanduser()
    manifest_path = Path(args.manifest)
    meshes = find_meshes(mesh_root)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"objects": {}}
    uploaded = manifest.get("objects", {}) or {}

    print(f"mesh_root: {mesh_root}")
    print(f"objects_with_source_meshes: {len(meshes)}")
    print(f"source_mesh_files: {sum(len(v) for v in meshes.values())}")
    print(f"manifest_3d_objects: {len(uploaded)}")
    print()
    print("conversion_tools:")
    for tool in ("blender", "assimp", "obj2gltf", "meshlabserver"):
        print(f"  {tool}: {shutil.which(tool) or 'missing'}")
    print()
    missing_uploaded = sorted(set(meshes) - set(uploaded))
    print(f"source_meshes_not_in_manifest: {len(missing_uploaded)}")
    for obj in missing_uploaded[:20]:
        print(f"  {obj}: {meshes[obj][0]}")
    if len(missing_uploaded) > 20:
        print(f"  ... {len(missing_uploaded) - 20} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

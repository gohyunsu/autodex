"""Decimate raw textured meshes for fast viser visualization.

Reads:  {obj_path}/{obj}/raw_mesh/{obj}.obj
Writes: {obj_path}/{obj}/raw_mesh_decimated/{obj}.obj (+ mtl + textures)

Uses pymeshlab quadric edge collapse with texture preservation.
"""

import argparse
import os
import re
import shutil
import sys

import pymeshlab


def referenced_textures(mtl_path):
    refs = []
    with open(mtl_path) as f:
        for line in f:
            m = re.match(r"\s*(map_[A-Za-z_]+)\s+(.+?)\s*$", line)
            if m:
                refs.append(m.group(2))
    return refs


def decimate_one(obj_name, in_dir, out_dir, target_faces):
    in_obj = os.path.join(in_dir, f"{obj_name}.obj")
    if not os.path.isfile(in_obj):
        print(f"  skip (no raw obj): {obj_name}")
        return False

    os.makedirs(out_dir, exist_ok=True)
    out_obj = os.path.join(out_dir, f"{obj_name}.obj")

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(in_obj)
    in_faces = ms.current_mesh().face_number()
    if in_faces <= target_faces:
        print(f"  {obj_name}: already small ({in_faces} faces), copying as-is")
        shutil.copy2(in_obj, out_obj)
    else:
        ms.apply_filter(
            "meshing_decimation_quadric_edge_collapse_with_texture",
            targetfacenum=target_faces,
            preserveboundary=True,
            preservenormal=True,
            optimalplacement=True,
        )
        ms.save_current_mesh(
            out_obj,
            save_vertex_color=False,
            save_vertex_normal=False,
            save_wedge_texcoord=True,
        )
        out_faces = ms.current_mesh().face_number()
        print(f"  {obj_name}: {in_faces} → {out_faces} faces")

    # pymeshlab writes its own {stem}.obj.mtl; just bring the texture files over.
    in_mtl = os.path.join(in_dir, f"{obj_name}.mtl")
    if os.path.isfile(in_mtl):
        for tex in referenced_textures(in_mtl):
            src = os.path.join(in_dir, tex)
            dst = os.path.join(out_dir, os.path.basename(tex))
            if os.path.isfile(src) and not os.path.isfile(dst):
                shutil.copy2(src, dst)

    in_size = os.path.getsize(in_obj) / 1e6
    out_size = os.path.getsize(out_obj) / 1e6
    print(f"    obj size: {in_size:.1f} MB → {out_size:.1f} MB")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_path", required=True)
    parser.add_argument("--target_faces", type=int, default=50_000)
    parser.add_argument("--obj", default=None, help="single object name (optional)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.obj is not None:
        objs = [args.obj]
    else:
        objs = sorted(
            d for d in os.listdir(args.obj_path)
            if os.path.isdir(os.path.join(args.obj_path, d, "raw_mesh"))
        )

    print(f"Decimating {len(objs)} object(s) under {args.obj_path}")
    print(f"  target faces: {args.target_faces}")

    for obj in objs:
        in_dir = os.path.join(args.obj_path, obj, "raw_mesh")
        out_dir = os.path.join(args.obj_path, obj, "raw_mesh_decimated")
        if os.path.isdir(out_dir) and not args.overwrite:
            print(f"  skip (exists): {obj} — pass --overwrite to redo")
            continue
        try:
            decimate_one(obj, in_dir, out_dir, args.target_faces)
        except Exception as e:
            print(f"  FAILED {obj}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""Convert a GMR motion pickle file into the CSV format consumed by csv_to_npz.py.

Expected pkl schema (GMR output)::

    {
        "fps":            int,
        "root_pos":       (T, 3) float,         base position (x, y, z), world frame
        "root_rot":       (T, 4) float,         base orientation quaternion in xyzw order
        "dof_pos":        (T, N) float,         joint positions, N joints (e.g. 23 for G1)
        "local_body_pos": optional / None,
        "link_body_list": optional / None,
    }

Output CSV columns (matches csv_to_npz.py loader)::

    [0:3]  root_pos (x, y, z)
    [3:7]  root_rot (x, y, z, w)   -- csv_to_npz.py reorders to wxyz internally
    [7: ]  dof_pos  (N joints, in the same order as csv_to_npz.py joint_names)

Usage::

    python scripts/pkl_to_csv.py --input g123_qie_motion.pkl --output g123_qie_motion.csv
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


def inspect_pkl(data: dict) -> None:
    print("=" * 60)
    print(f"top-level type: {type(data).__name__}")
    if not isinstance(data, dict):
        print(data)
        return
    print(f"keys ({len(data)}):")
    for k, v in data.items():
        if hasattr(v, "shape"):
            print(f"  {k!r}: ndarray shape={tuple(v.shape)}, dtype={v.dtype}")
        elif isinstance(v, (list, tuple)):
            print(f"  {k!r}: {type(v).__name__} len={len(v)}")
        else:
            print(f"  {k!r}: {type(v).__name__} value={v!r}")
    print("=" * 60)


def build_csv_matrix(data: dict) -> np.ndarray:
    required = ("root_pos", "root_rot", "dof_pos")
    missing = [k for k in required if k not in data or data[k] is None]
    if missing:
        raise KeyError(f"pkl is missing required fields: {missing}")

    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    root_rot = np.asarray(data["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must be (T, 3), got {root_pos.shape}")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(f"root_rot must be (T, 4), got {root_rot.shape}")
    if dof_pos.ndim != 2:
        raise ValueError(f"dof_pos must be (T, N), got {dof_pos.shape}")

    T = root_pos.shape[0]
    if root_rot.shape[0] != T or dof_pos.shape[0] != T:
        raise ValueError(
            f"frame count mismatch: root_pos={root_pos.shape[0]}, "
            f"root_rot={root_rot.shape[0]}, dof_pos={dof_pos.shape[0]}"
        )

    # Normalize quaternion (GMR usually outputs unit quats, but be defensive against drift).
    norms = np.linalg.norm(root_rot, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    root_rot = root_rot / norms

    matrix = np.concatenate([root_pos, root_rot, dof_pos], axis=1)
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a GMR motion pkl to a CSV consumable by csv_to_npz.py.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input .pkl file.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output .csv file.")
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()

    if not in_path.is_file():
        raise FileNotFoundError(f"input pkl not found: {in_path}")

    with open(in_path, "rb") as f:
        data = pickle.load(f)

    inspect_pkl(data)

    matrix = build_csv_matrix(data)
    print(f"csv matrix shape: {matrix.shape}  (= [T, 3 root_pos + 4 root_rot(xyzw) + {matrix.shape[1] - 7} dof])")
    if "fps" in data:
        print(f"source fps: {data['fps']}  -> pass --input_fps {data['fps']} to csv_to_npz.py")
    print(f"first row: {matrix[0]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, matrix, delimiter=",", fmt="%.8f")
    print(f"wrote {matrix.shape[0]} frames to {out_path}")


if __name__ == "__main__":
    main()

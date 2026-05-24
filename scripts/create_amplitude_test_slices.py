#!/usr/bin/env python3
import argparse
import os
import random
from pathlib import Path

import numpy as np


EXPECTED_SHAPE = (1006, 782)
EXPECTED_LABEL_MIN = 0
EXPECTED_LABEL_MAX = 5
MARKER_NAME = ".created_by_create_amplitude_test_slices_v2"


def resolve_ampl3d_dir(data_root):
    candidates = [
        data_root / "ampl3d",
        data_root / "ampl_3d",
    ]
    for candidate in candidates:
        if (candidate / "parihaka_data.npz").is_file() and (
            candidate / "parihaka_labels.npz"
        ).is_file():
            return candidate
    raise FileNotFoundError(
        "Cannot find parihaka_data.npz and parihaka_labels.npz in "
        + " or ".join(str(path) for path in candidates)
    )


def load_single_array(path):
    archive = np.load(path)
    if len(archive.files) != 1:
        raise ValueError(f"Expected one array in {path}, got {archive.files}")
    return archive[archive.files[0]]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create amplitude/test .dat slices from Parihaka 3D npz files."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[2] / "data")),
        help="Directory that contains amplitude and ampl3d/ampl_3d.",
    )
    parser.add_argument("--count", type=int, default=45, help="Number of slices to write.")
    parser.add_argument(
        "--axis",
        type=int,
        default=2,
        help="Axis along which consecutive 2D slices are selected. Use 2 for 1006x782 slices.",
    )
    parser.add_argument("--start", type=int, default=None, help="First slice index.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for start index.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing .dat files in amplitude/test/input and target.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    ampl3d_dir = resolve_ampl3d_dir(data_root)
    output_input_dir = data_root / "amplitude" / "test" / "input"
    output_target_dir = data_root / "amplitude" / "test" / "target"

    data = load_single_array(ampl3d_dir / "parihaka_data.npz")
    labels = load_single_array(ampl3d_dir / "parihaka_labels.npz")

    if data.shape != labels.shape:
        raise ValueError(f"Data and labels shapes differ: {data.shape} vs {labels.shape}")
    if not 0 <= args.axis < data.ndim:
        raise ValueError(f"Axis {args.axis} is out of bounds for shape {data.shape}")
    if args.count <= 0:
        raise ValueError("--count must be positive")

    max_start = data.shape[args.axis] - args.count
    if max_start < 0:
        raise ValueError(
            f"Cannot take {args.count} slices from axis {args.axis} with length {data.shape[args.axis]}"
        )

    if args.start is None:
        rng = random.Random(args.seed)
        start = rng.randint(0, max_start)
    else:
        start = args.start
        if not 0 <= start <= max_start:
            raise ValueError(f"--start must be in [0, {max_start}], got {start}")

    output_input_dir.mkdir(parents=True, exist_ok=True)
    output_target_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for directory in (output_input_dir, output_target_dir):
            for path in directory.glob("*.dat"):
                path.unlink()

    for out_index, slice_index in enumerate(range(start, start + args.count)):
        data_slice = np.take(data, slice_index, axis=args.axis)
        label_slice = np.take(labels, slice_index, axis=args.axis)

        if data_slice.shape != EXPECTED_SHAPE:
            raise ValueError(
                f"Slice shape is {data_slice.shape}, expected {EXPECTED_SHAPE}. "
                f"For Parihaka shape {data.shape}, use --axis 2."
            )
        if label_slice.shape != EXPECTED_SHAPE:
            raise ValueError(
                f"Label slice shape is {label_slice.shape}, expected {EXPECTED_SHAPE}"
            )

        label_min = int(label_slice.min())
        label_max = int(label_slice.max())
        if label_min >= 1 and label_max <= 6:
            label_slice = label_slice - 1
            label_min -= 1
            label_max -= 1
        if label_min < EXPECTED_LABEL_MIN or label_max > EXPECTED_LABEL_MAX:
            raise ValueError(
                f"Label values must be in [{EXPECTED_LABEL_MIN}, {EXPECTED_LABEL_MAX}], "
                f"got [{label_min}, {label_max}] at source slice {slice_index}"
            )

        np.ascontiguousarray(data_slice, dtype=np.float32).tofile(
            output_input_dir / f"{out_index}.dat"
        )
        np.ascontiguousarray(label_slice, dtype=np.int8).tofile(
            output_target_dir / f"{out_index}.dat"
        )

    (output_target_dir.parent / MARKER_NAME).write_text(
        f"axis={args.axis}\nstart={start}\ncount={args.count}\nlabels={EXPECTED_LABEL_MIN}..{EXPECTED_LABEL_MAX}\n",
        encoding="utf-8",
    )

    print(f"data_root={data_root}")
    print(f"ampl3d_dir={ampl3d_dir}")
    print(f"shape={data.shape}")
    print(f"axis={args.axis}")
    print(f"start={start}")
    print(f"count={args.count}")
    print(f"input_dir={output_input_dir}")
    print(f"target_dir={output_target_dir}")


if __name__ == "__main__":
    main()

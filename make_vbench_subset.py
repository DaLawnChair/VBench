#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np


VBENCH_16_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "temporal_style",
    "appearance_style",
    "overall_consistency",
]


def normalize_dims(dim_field):
    if isinstance(dim_field, str):
        return [dim_field]
    if isinstance(dim_field, list):
        return dim_field
    raise TypeError(f"Invalid dimension field: {dim_field!r}")


def load_vbench_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected VBench_full_info.json to be a JSON list.")

    for i, item in enumerate(data):
        if "prompt_en" not in item:
            raise ValueError(f"Entry {i} is missing `prompt_en`.")
        if "dimension" not in item:
            raise ValueError(f"Entry {i} is missing `dimension`.")
        item["dimension"] = normalize_dims(item["dimension"])

    return data


def dimension_counts(data, dimensions):
    counts = Counter()
    for item in data:
        for d in item["dimension"]:
            if d in dimensions:
                counts[d] += 1
    return counts


def solve_exact_balanced_subset(
    data,
    dimensions,
    per_dim,
    seed=0,
    min_multidim=0,
    exact_num_prompts=None,
):
    """
    Selects a subset of prompts such that every dimension appears exactly
    `per_dim` times.

    Multi-dimension prompts are allowed and count toward every dimension
    listed in their original `dimension` field.
    """
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as e:
        raise ImportError(
            "This script requires scipy for exact balanced sampling.\n"
            "Install it with:\n\n"
            "    pip install scipy\n"
        ) from e

    rng = np.random.default_rng(seed)

    n_prompts = len(data)
    n_dims = len(dimensions)

    dim_to_row = {d: i for i, d in enumerate(dimensions)}

    # A[row, col] = 1 if prompt col belongs to dimension row.
    A = np.zeros((n_dims, n_prompts), dtype=float)

    for j, item in enumerate(data):
        for d in item["dimension"]:
            if d in dim_to_row:
                A[dim_to_row[d], j] = 1.0

    # Binary variable x_j = 1 if prompt j is selected.
    integrality = np.ones(n_prompts, dtype=int)
    bounds = Bounds(lb=np.zeros(n_prompts), ub=np.ones(n_prompts))

    constraints = []

    # Exact balance constraint:
    # for each dimension d:
    #     sum_j A[d, j] * x_j == per_dim
    target = np.full(n_dims, per_dim, dtype=float)
    constraints.append(LinearConstraint(A, lb=target, ub=target))

    labels_per_prompt = A.sum(axis=0)
    multidim_mask = labels_per_prompt > 1

    # Optionally require at least K multi-dimension prompts.
    if min_multidim > 0:
        constraints.append(
            LinearConstraint(
                multidim_mask.astype(float)[None, :],
                lb=np.array([min_multidim], dtype=float),
                ub=np.array([np.inf], dtype=float),
            )
        )

    # Optionally require exactly N prompt rows.
    #
    # Be careful: with multi-dimension prompts, `per_dim * 16` is no longer
    # necessarily the desired number of prompt rows.
    if exact_num_prompts is not None:
        constraints.append(
            LinearConstraint(
                np.ones((1, n_prompts), dtype=float),
                lb=np.array([exact_num_prompts], dtype=float),
                ub=np.array([exact_num_prompts], dtype=float),
            )
        )

    # Objective:
    # - Prefer multi-dimension prompts.
    # - Prefer prompts covering more labels.
    # - Add tiny randomness for different subsets with different seeds.
    #
    # scipy minimizes, so negative values mean "prefer selecting this".
    tiny_noise = rng.random(n_prompts) * 1e-6
    c = -(labels_per_prompt - 1.0) + tiny_noise

    result = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"time_limit": 300},
    )

    if not result.success:
        raise RuntimeError(
            "Could not find an exact balanced subset.\n"
            f"Solver status: {result.message}\n\n"
            "Try one of the following:\n"
            "  1. Lower --per-dim\n"
            "  2. Lower --min-multidim\n"
            "  3. Remove --exact-num-prompts if you used it\n"
        )

    selected_indices = np.where(result.x > 0.5)[0].tolist()
    selected = [data[i] for i in selected_indices]

    random.Random(seed).shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Create a balanced VBench subset while preserving multi-dimension prompts."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("VBench_full_info.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("VBench_subset_balanced_multidim.json"),
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=Path("VBench_subset_balanced_multidim.txt"),
    )
    parser.add_argument(
        "--per-dim",
        type=int,
        required=True,
        help="Exact number of prompt incidences per VBench dimension.",
    )
    parser.add_argument(
        "--min-multidim",
        type=int,
        default=0,
        help="Require at least this many selected prompts to have multiple dimensions.",
    )
    parser.add_argument(
        "--exact-num-prompts",
        type=int,
        default=None,
        help="Optional: require exactly this many prompt rows in the subset.",
    )
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    data = load_vbench_json(args.input)

    original_counts = dimension_counts(data, VBENCH_16_DIMENSIONS)

    print("Original dimension counts:")
    for d in VBENCH_16_DIMENSIONS:
        print(f"  {d:24s} {original_counts[d]}")

    subset = solve_exact_balanced_subset(
        data=data,
        dimensions=VBENCH_16_DIMENSIONS,
        per_dim=args.per_dim,
        seed=args.seed,
        min_multidim=args.min_multidim,
        exact_num_prompts=args.exact_num_prompts,
    )

    subset_counts = dimension_counts(subset, VBENCH_16_DIMENSIONS)

    num_multidim = sum(len(item["dimension"]) > 1 for item in subset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)

    print()
    print(f"Wrote {len(subset)} prompt rows to: {args.output}")
    print(f"Multi-dimension prompts selected: {num_multidim}")
    print(f"Target count per dimension: {args.per_dim}")

    print()
    print("Subset dimension counts:")
    for d in VBENCH_16_DIMENSIONS:
        print(f"  {d:24s} {subset_counts[d]}")

    assert all(
        subset_counts[d] == args.per_dim for d in VBENCH_16_DIMENSIONS
    ), "Subset is not exactly balanced."


    # save the prompts used as a txt file for easy inference
    prompts = []
    for item in subset:
        if "prompt_en" in item:
            prompts.append(item["prompt_en"])
    args.output_txt.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_txt, "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(prompt.strip() + "\n")
    print(f"Saved {len(prompts)} prompts to {args.output_txt}")


if __name__ == "__main__":
    main()


# python make_vbench_subset.py \
#   --input vbench/VBench_full_info.json \
#   --output vbench/VBench_subset_balanced_10_each.json \
#   --output-txt vbench/VBench_subset_balanced_10_each.txt \
#   --per-dim 10 \
#   --min-multidim 20 \
#   --seed 42
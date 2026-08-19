#!/usr/bin/env python3
import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


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
        if not isinstance(item, dict):
            raise ValueError(f"Entry {i} is not a dict.")
        if "prompt_en" not in item:
            raise ValueError(f"Entry {i} is missing `prompt_en`.")
        if "dimension" not in item:
            raise ValueError(f"Entry {i} is missing `dimension`.")
        normalize_dims(item["dimension"])

    return data


def group_by_dimension(data, dimensions):
    groups = defaultdict(list)

    for idx, item in enumerate(data):
        dims = normalize_dims(item["dimension"])
        for dim in dims:
            if dim in dimensions:
                groups[dim].append((idx, item))

    missing = [dim for dim in dimensions if dim not in groups]
    if missing:
        raise ValueError(f"These dimensions were not found in the JSON: {missing}")

    return groups


def count_dimensions(data, dimensions):
    counts = Counter()
    for item in data:
        for dim in normalize_dims(item["dimension"]):
            if dim in dimensions:
                counts[dim] += 1
    return counts


def sample_even_subset(
    data,
    dimensions,
    per_dim,
    seed,
    dedupe_prompt=True,
    canonicalize_dimension=True,
):
    """
    canonicalize_dimension=True means each sampled record is written as:
        "dimension": ["the_dimension_it_was_sampled_for"]

    This guarantees exact equal counts in the output JSON even if an original
    VBench prompt record had multiple dimensions.
    """
    rng = random.Random(seed)
    groups = group_by_dimension(data, dimensions)

    selected = []
    used_prompts = set()

    # Sample dimensions with fewer candidates first to reduce greedy failures.
    dim_order = sorted(dimensions, key=lambda d: len(groups[d]))

    for dim in dim_order:
        pool = groups[dim][:]
        rng.shuffle(pool)

        picked = []
        for _, item in pool:
            prompt = item["prompt_en"]

            if dedupe_prompt and prompt in used_prompts:
                continue

            out = copy.deepcopy(item)

            if canonicalize_dimension:
                out["dimension"] = [dim]

            picked.append(out)

            if dedupe_prompt:
                used_prompts.add(prompt)

            if len(picked) == per_dim:
                break

        if len(picked) < per_dim:
            raise RuntimeError(
                f"Could only sample {len(picked)} prompts for `{dim}`, "
                f"but requested {per_dim}. Try lowering --per-dim or disabling "
                f"--dedupe-prompt."
            )

        selected.extend(picked)

    # Shuffle final output so prompts are not ordered by dimension.
    rng.shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Create an evenly balanced VBench prompt subset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("VBench_full_info.json"),
        help="Path to VBench_full_info.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("VBench_subset_even.json"),
        help="Where to save the subset JSON",
    )
    parser.add_argument(
        "--per-dim",
        type=int,
        default=None,
        help="Number of prompts to sample per VBench dimension.",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=None,
        help="Total number of prompts. Must be divisible by 16.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-dedupe-prompt",
        action="store_true",
        help="Allow the same prompt text to appear more than once.",
    )
    parser.add_argument(
        "--preserve-original-dimensions",
        action="store_true",
        help=(
            "Keep each record's original `dimension` list. By default, the script "
            "sets `dimension` to only the sampled dimension to guarantee exact balance."
        ),
    )

    args = parser.parse_args()

    if args.per_dim is not None and args.total is not None:
        raise ValueError("Use only one of --per-dim or --total.")

    if args.total is not None:
        if args.total % len(VBENCH_16_DIMENSIONS) != 0:
            raise ValueError(
                f"--total must be divisible by {len(VBENCH_16_DIMENSIONS)}."
            )
        per_dim = args.total // len(VBENCH_16_DIMENSIONS)
    else:
        data_preview = load_vbench_json(args.input)
        groups_preview = group_by_dimension(data_preview, VBENCH_16_DIMENSIONS)

        # If --per-dim is omitted, use the smallest dimension pool.
        per_dim = args.per_dim
        if per_dim is None:
            per_dim = min(len(groups_preview[d]) for d in VBENCH_16_DIMENSIONS)

    data = load_vbench_json(args.input)

    original_counts = count_dimensions(data, VBENCH_16_DIMENSIONS)

    subset = sample_even_subset(
        data=data,
        dimensions=VBENCH_16_DIMENSIONS,
        per_dim=per_dim,
        seed=args.seed,
        dedupe_prompt=not args.no_dedupe_prompt,
        canonicalize_dimension=not args.preserve_original_dimensions,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)

    subset_counts = count_dimensions(subset, VBENCH_16_DIMENSIONS)

    print(f"Wrote {len(subset)} prompts to: {args.output}")
    print(f"Target per dimension: {per_dim}")
    print("\nOriginal counts:")
    for dim in VBENCH_16_DIMENSIONS:
        print(f"  {dim:24s} {original_counts[dim]}")

    print("\nSubset counts:")
    for dim in VBENCH_16_DIMENSIONS:
        print(f"  {dim:24s} {subset_counts[dim]}")


if __name__ == "__main__":
    main()


# python make_vbench_subset.py \
#   --input vbench/VBench_full_info.json \
#   --output vbench/VBench_subset_160.json \
#   --per-dim 10 \
#   --seed 42 \
#   --preserve-original-dimensions

# python make_vbench_subset.py \
#   --input vbench/VBench_subset_160.json \
#   --output vbench/remove_later.json \
#   --per-dim 10 \
#   --seed 42
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import defaultdict, deque


def load_prompts(path: Path):
    """
    Supports:
      1. .txt files with one prompt per line
      2. VBench-style .json files containing [{"prompt_en": "..."}]
    """
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        prompts = []
        for item in data:
            if isinstance(item, dict) and "prompt_en" in item:
                prompts.append(item["prompt_en"].strip())
            else:
                raise ValueError(f"Invalid JSON item without prompt_en: {item}")

        return prompts

    else:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]


def build_subset_to_superset_mapping(subset_prompts, superset_prompts):
    """
    Returns:
        mapping: dict[int, int]
            subset index -> superset index

    Duplicate behavior:
        If a prompt appears multiple times in the superset, each subset occurrence
        gets mapped to the next available superset occurrence.
    """
    superset_lookup = defaultdict(deque)

    for superset_idx, prompt in enumerate(superset_prompts):
        superset_lookup[prompt].append(superset_idx)

    mapping = {}
    missing = []

    for subset_idx, prompt in enumerate(subset_prompts):
        if superset_lookup[prompt]:
            mapping[subset_idx] = superset_lookup[prompt].popleft()
        else:
            missing.append((subset_idx, prompt))

    if missing:
        print(f"Warning: {len(missing)} subset prompts could not be mapped.")
        print("First few missing prompts:")
        for subset_idx, prompt in missing[:10]:
            print(f"  subset_idx={subset_idx}: {prompt}")

    return mapping, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--superset", type=Path, required=True)
    parser.add_argument("--superset-extended", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("subset_to_superset_mapping.json"),
    )
    parser.add_argument("--output-subset-extended", type=Path, required=True)
    
    args = parser.parse_args()

    subset_prompts = load_prompts(args.subset)
    superset_prompts = load_prompts(args.superset)
    superset_extended_prompts = load_prompts(args.superset_extended)

    mapping, missing = build_subset_to_superset_mapping(
        subset_prompts=subset_prompts,
        superset_prompts=superset_prompts,
    )
    import ipdb;ipdb.set_trace()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    subset_extended_prompts = [superset_extended_prompts[mapping[i]] for i in range(len(subset_prompts)) if i in mapping]
    

    args.output_subset_extended.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_subset_extended, "w", encoding="utf-8") as f:
        for prompt in subset_extended_prompts:
            f.write(prompt.strip() + "\n")

    import ipdb;ipdb.set_trace()

    # with open(args.output, "w", encoding="utf-8") as f:
    #     json.dump(mapping, f, indent=2)

    # print()
    # print(f"Subset prompts: {len(subset_prompts)}")
    # print(f"Superset prompts: {len(superset_prompts)}")
    # print(f"Mapped prompts: {len(mapping)}")
    # print(f"Missing prompts: {len(missing)}")
    # print(f"Saved mapping to: {args.output}")


if __name__ == "__main__":
    main()


# python map_subset_to_superset.py \
#   --subset  /shared/john/VBench/custom_small_vbench_set/VBench_subset_balanced_10_each.json \
#   --superset /shared/john/VBench/custom_small_vbench_set/sf_vbench_prompts/all_dimension.txt \
#   --superset-extended /shared/john/VBench/custom_small_vbench_set/sf_vbench_prompts/all_dimension_extended.txt \
#   --output-subset-extended  /shared/john/VBench/custom_small_vbench_set/VBench_subset_balanced_10_each_extended.txt
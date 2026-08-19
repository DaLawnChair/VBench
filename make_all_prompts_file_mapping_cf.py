import argparse
import json
from pathlib import Path


def load_prompts(path: Path):
    """
    Supports:
      1. .txt files with one prompt per line
      2. VBench-style .json files containing [{"prompt_en": "..."}]
    """
    if path is None:
        return []

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        prompts = []
        for item in data:
            if isinstance(item, dict) and "prompt_en" in item:
                prompt = item["prompt_en"].strip()
                if prompt:
                    prompts.append(prompt)
            else:
                raise ValueError(f"Invalid JSON item without prompt_en: {item}")

        return prompts

    else:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Load two prompt files, concatenate them, and create repeated filename-to-prompt mappings."
    )

    parser.add_argument(
        "--input-1",
        type=Path,
        required=True,
        help="Path to first prompt file, either .txt or VBench-style .json",
    )

    # parser.add_argument(
    #     "--input-2",
    #     type=Path,
    #     default=None,
    #     help="Path to second prompt file, either .txt or VBench-style .json",
    # )

    parser.add_argument(
        "--output_mapping",
        type=Path,
        required=True,
        help="Where to save the output JSON mapping",
    )

    parser.add_argument(
        "--output_full_dim_data",
        type=Path,
        required=True,
        help="Where to save the output JSON full dimension data",
    )


    parser.add_argument(
        "--dimension",
        type=str,
        required=True,
        help="The dimension for the full dimension data",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of times to repeat each prompt",
    )
    parser.add_argument(
        "--prefix_idx",
        type=int,
        default=150,
        help="Ending index for filenames prefixes",
    )
    
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    input_1_data = load_prompts(args.input_1)
    # input_2_data = load_prompts(args.input_2)

    total_data = input_1_data

    mapping = {}

    for idx, prompt in enumerate(total_data):
        for repeat_i in range(args.repeats):
            # filename = f"{idx:05d}-{repeat_i}.mp4"
            filename = f"{prompt[:args.prefix_idx]}-{repeat_i}.mp4"
            mapping[filename] = prompt

    full_data = []
    for idx, prompt in enumerate(total_data):
        entry = {'prompt_en': prompt, 'dimension': [args.dimension]}
        full_data.append(entry)

    args.output_mapping.parent.mkdir(parents=True, exist_ok=True)
    args.output_full_dim_data.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_mapping, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

        
    print(f"Loaded {len(input_1_data)} prompts from input 1")
    print(f"Total unique prompt entries before repeat: {len(total_data)}")
    print(f"Saved {len(mapping)} filename-to-prompt mappings to {args.output_mapping}")

    with open(args.output_full_dim_data, "w", encoding="utf-8") as f:
        json.dump(full_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(full_data)} full dimension entries to {args.output_full_dim_data}")


if __name__ == "__main__":
    main()

## usage
# python3 make_all_prompts_file_mapping_cf.py \
#   --input-1 /shared/john/VBench/custom_small_vbench_set/text_align.txt \
#   --output_mapping /shared/john/VBench/custom_small_vbench_set/cf_prompts_extended_dynamic_degree_mapping.json \
#   --output_full_dim_data /shared/john/VBench/custom_small_vbench_set/cf_prompts_extended_dynamic_degree.json \
#   --dimension dynamic_degree \
#   --prefix_idx 150 \
#   --repeats 5

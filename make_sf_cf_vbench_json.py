import argparse
from email import parser
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
		"--original",
		type=Path,
		required=True,
		help="Path to first prompt file, either .txt or VBench-style .json",
	)

	parser.add_argument(
		"--original-prompts",
		type=Path,
		default=None,
		help="Path to second prompt file, either .txt or VBench-style .json",
	)

	parser.add_argument(
		"--new-prompts",
		type=Path,
		default=None,
		help="Path to second prompt file, either .txt or VBench-style .json",
	)

	parser.add_argument(
		"--output",
		type=Path,
		required=True,
		help="Where to save the output JSON mapping",
	)

	args = parser.parse_args()



	original_json = json.load(open(args.original, "r"))
	original_prompts = load_prompts(args.original_prompts)
	new_prompts = load_prompts(args.new_prompts)


	mapping = {}
	import ipdb;ipdb.set_trace()
	assert len(original_prompts) == len(new_prompts), "Original and new prompts must have the same length"
	assert len(original_prompts) == len(original_json), "Original and original_json must have the same length"


	data = []
	for idx, og_prompt, new_prompt in zip(range(len(original_prompts)), original_prompts, new_prompts):
		assert og_prompt == original_json[idx]["prompt_en"], f"Prompt mismatch at index {idx}"
		entry = original_json[idx].copy()
		entry["prompt_en"] = new_prompt
		data.append(entry)

	args.output.parent.mkdir(parents=True, exist_ok=True)

	with open(args.output, "w", encoding="utf-8") as f:
		 json.dump(data, f, indent=2, ensure_ascii=False)

	# print(f"Loaded {len(input_1_data)} prompts from input 1")
	# print(f"Loaded {len(input_2_data)} prompts from input 2")
	# print(f"Total unique prompt entries before repeat: {len(total_data)}")
	# print(f"Saved {len(mapping)} filename-to-prompt mappings to {args.output}")


if __name__ == "__main__":
	main()

## usage
# python3 make_sf_cf_vbench_json.py \
#   --original /shared/john/VBench/custom_small_vbench_set/VBench_full_info.json \
#   --original-prompts /shared/john/VBench/custom_small_vbench_set/sf_vbench_prompts/all_dimension.txt \
#   --new-prompts /shared/john/VBench/custom_small_vbench_set/sf_vbench_prompts/all_dimension_extended.txt \
#   --output /shared/john/VBench/custom_small_vbench_set/sf_cf_all_dimensions_extended.json
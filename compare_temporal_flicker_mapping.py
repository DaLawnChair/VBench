


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


vbench_data = json.load(open("/shared/john/VBench/custom_small_vbench_set/VBench_full_info.json", "r"))
raw_prompts = load_prompts(Path("/shared/john/VBench/custom_small_vbench_set/sf_vbench_prompts/all_dimension.txt"))



def get_temporal_flicker(data):
	temporal_flicker_prompts = []
	indicies = []
	for idx, item in enumerate(data):
		if isinstance(item, dict) and "dimension" in item and "prompt_en" in item:	
			if "temporal_flickering" in item.get("dimension", []):
				temporal_flicker_prompts.append(item["prompt_en"])
				indicies.append(idx)
	return temporal_flicker_prompts, indicies	


temporal_flicker_prompts, indicies = get_temporal_flicker(vbench_data)
vbench_data[74]['prompt_en']== temporal_flicker_prompts[-1]
import ipdb;ipdb.set_trace()

# Goal: classes like color, apperance_style, multiple_objects, object_class, and spatial_relationship
# only work on the non-extended prompt, and for the case for color, give errors. This is due to
# Vbench's parsing of the prompt.
# Solution: make a the mapping of theses videos to the original prompt instead of the extended prompt


# # saving files:
json_save_file = "sf_cf_all_dimensions_extended_to_base_special_prompts.json"
mapping_save_file="sf_cf_prompts_extended_to_base_special_prompts_mapping.json"
# loading files:
json_data_file = "sf_cf_all_dimensions_extended.json"
mapping_file = "sf_cf_prompts_extended_mapping.json"


## saving files:
# json_save_file = "sf_all_dimensions_extended_to_base_special_prompts.json"
# mapping_save_file="sf_prompts_extended_to_base_special_prompts_mapping.json"
# # loading files:
# json_data_file = "sf_all_dimension_extended.json"
# mapping_file = "sf_prompts_extended_mapping.json"

## update: humman action score reported as 0, when previously it would be 100 or 90 on a smaller test set
## likely human action prompt is not usable for extended prompts
# DIMENSIONS_TO_NOT_EXTEND = ["color", "appearance_style", "multiple_objects", "object_class", "spatial_relationship"]
DIMENSIONS_TO_NOT_EXTEND = ['object_class', 'multiple_objects', 'scene', 'appearance_style', 'color', 'spatial_relationship', 'human_action']
PREFIX_IDX = 150

import json

data = json.load(open(json_data_file, 'r'))	
mapping = json.load(open(mapping_file, 'r'))


with open("/shared/john/VBench/custom_small_vbench_set/sf_vbench_prompts/all_dimension.txt", 'r') as f:
	base_prompts = f.read().splitlines()
	
with open("/shared/john/VBench/custom_small_vbench_set/sf_vbench_prompts/all_dimension_extended.txt", 'r') as f:
	extended_prompts = f.read().splitlines()

assert len(base_prompts) == len(extended_prompts)
extended_to_base_mapping = {extended_prompts[i]: base_prompts[i] for i in range(len(base_prompts))}


# change the prompt_en for the json file
changed_enteries = []
changed_enteries_idx = []

map_extended_to_base_prompt = {}
for idx, entry in enumerate(data):
	dimensions = entry['dimension']
	for dim in dimensions:
		if dim in DIMENSIONS_TO_NOT_EXTEND:
			# import ipdb;ipdb.set_trace()
			value = entry['prompt_en']
			# if value not in mapping:
			# 	print(f"Value {value} not found in mapping")
			# 	continue
			base_prompt = extended_to_base_mapping[value]
			entry['prompt_en'] = base_prompt
			changed_enteries.append(entry)

			map_extended_to_base_prompt[value[:PREFIX_IDX]] = base_prompt
			changed_enteries_idx.append(idx)

for key, value in mapping.items():
	# import ipdb;ipdb.set_trace()
	# import ipdb;ipdb.set_trace()
	if value[:PREFIX_IDX] in map_extended_to_base_prompt:
		# import ipdb;ipdb.set_trace()
		# print('lol')
		mapping[key] = map_extended_to_base_prompt[value[:PREFIX_IDX]]
		


## save the json file
# json.dump(data, open("/shared/john/VBench/custom_small_vbench_set/sf_cf_all_dimensions_extended_to_base_special_prompts.json", 'w'), indent=2, ensure_ascii=False)
json.dump(data, open(json_save_file, 'w'), indent=2, ensure_ascii=False)

# len(changed_enteries_idx)==420 = 85+84+90+82+79 # which is the count for the enteries to not change
# data[changed_enteries_idx[0]]

## save the mapping file
json.dump(mapping, open(mapping_save_file, 'w'), indent=2, ensure_ascii=False)
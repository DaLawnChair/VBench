# nevermind, not worth really making it since len(new_data_prompts)=799 already... might as well infer over the whole generation
import ipdb; ipdb.set_trace()
import json
data = json.load(open("sf_semantic_dimensions_extended_to_base_special_prompts.json",'r'))

semantic_dims = set(['object_class', 'multiple_objects', 'human_action', 'color', 'spatial_relationship', 'scene', 'appearance_style', 'temporal_style', 'overall_consistency'])

new_data = []
new_data_prompts = []
for entry in data:
    if set(entry['dimension']) & semantic_dims:
        new_data.append(entry)
        new_data_prompts.append(entry['prompt_en']+'\n')
import ipdb;ipdb.set_trace()


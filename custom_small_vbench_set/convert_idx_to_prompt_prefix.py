


import json
data = json.load(open("sf_cf_prompts_extended_mapping_old.json", 'r'))

new_data = {}
for key, value in data.items():
	identifier = f"{value[:150]}{key[key.rfind("-", len(key)-1 - 7,len(key)-1):]}"
	new_data[identifier] = value

json.dump(new_data, open("sf_cf_prompts_extended_mapping.json", 'w'), indent=2, ensure_ascii=False)
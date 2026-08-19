from vbench.unitemp_flickering_ratio import compute_unitemp_flickering_ratio
from vbench2_beta_long.utils import reorganize_clips_results


def compute_long_unitemp_flickering_ratio(json_dir, device, submodules_list, **kwargs):
    all_results, detailed_results = compute_unitemp_flickering_ratio(json_dir, device, submodules_list, **kwargs)

    return reorganize_clips_results(detailed_results)
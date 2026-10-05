"""Sort VBench per-dimension results lowest score first.

Kept free of third-party imports: `vbench.utils` re-exports `sort_eval_results`
for `VBench.evaluate`, and `scripts/sort_eval_results.py` loads this file
directly to re-sort existing json files.
"""


def _video_results_sort_key(entry):
    """Sort key for one per-video result dict: lowest score first.

    Scores are floats (most dimensions) or bools (human_action, dynamic_degree;
    False = failed video sorts first). Anything else (missing key, dict, None)
    sorts after every number so it is never compared against a float.
    """
    v = entry.get('video_results') if isinstance(entry, dict) else None
    if isinstance(v, bool):
        return (0, float(v))
    if isinstance(v, (int, float)):
        return (0, float(v))
    return (1, 0.0)


def sort_eval_results(results_dict):
    """Sort each dimension's per-video results from lowest to highest score.

    `results_dict` is what `VBench.evaluate` dumps to `*_eval_results.json`:
    {dimension: [aggregate_score, [{'video_path': ..., 'video_results': ...}, ...]]}.
    The aggregate score and every per-video dict are left untouched; only the
    order of the per-video list changes (stable sort, so ties keep evaluation
    order). Dimensions whose value does not have that shape are returned as-is.
    """
    sorted_dict = {}
    for dimension, value in results_dict.items():
        if (isinstance(value, (list, tuple)) and len(value) == 2
                and isinstance(value[1], list)):
            per_video = sorted(value[1], key=_video_results_sort_key)
            sorted_dict[dimension] = [value[0], per_video]
        else:
            sorted_dict[dimension] = value
    return sorted_dict

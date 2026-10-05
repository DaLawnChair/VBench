#!/usr/bin/env python3
"""Re-sort existing VBench `*_eval_results.json` files lowest score first.

New evaluations already come out sorted (see `sort_eval_results` in
vbench/utils.py, applied in `VBench.evaluate`). This script applies the same
sort to result files written before that change, e.g.

    python3 scripts/sort_eval_results.py \
        --folder <video_path>/vbench_scores/full_sf_all_dims/

Each `*_eval_results.json` in the folder is rewritten in place (aggregate
scores and per-video entries unchanged, only the per-video order changes).
Pass --dry-run to only report which dimensions would be reordered.
"""
import argparse
import glob
import json
import os

# load the sort helper from its file so importing `vbench` (decord, torch, ...)
# is not needed to re-sort json files
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    'eval_results_sort',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'vbench', 'eval_results_sort.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
sort_eval_results = _mod.sort_eval_results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--folder', nargs='+', required=True,
                   help='folder(s) containing *_eval_results.json (or explicit json paths)')
    p.add_argument('--dry-run', action='store_true', help='report only, do not rewrite files')
    args = p.parse_args()

    files = []
    for f in args.folder:
        if os.path.isdir(f):
            files += sorted(glob.glob(os.path.join(f, '*_eval_results.json')))
        elif f.endswith('.json'):
            files.append(f)
        else:
            raise SystemExit(f'not a folder or json file: {f}')
    if not files:
        raise SystemExit('no *_eval_results.json found')

    for path in files:
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
        sorted_data = sort_eval_results(data)
        changed = [d for d in data if sorted_data[d] != data[d]]
        state = 'reordered' if changed else 'already sorted'
        print(f'{path}: {state} ({", ".join(changed) if changed else ", ".join(data)})')
        if changed and not args.dry_run:
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(sorted_data, fh, indent=4)


if __name__ == '__main__':
    main()

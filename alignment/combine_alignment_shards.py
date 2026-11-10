import argparse
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import directories

RESULTS_ROOT = os.path.join(directories.DATA_CACHE_DIR, "alignment_results")
SHARD_RE = re.compile(r"qualitative_prompt_output_shard_(\d+)_of_(\d+)_shards\.json$")


def _parse_min_mtime(value):
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    return datetime.fromisoformat(value).timestamp()


def _mtime_iso(path):
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def _matches_filters(path, args):
    path_str = str(path)
    if args.seed and not any(f"/seed_{seed}/" in path_str for seed in args.seed):
        return False
    if args.model and not any(f"/{model}/" in path_str for model in args.model):
        return False
    if args.dataset and not any(f"/{dataset}/" in path_str for dataset in args.dataset):
        return False
    if args.method and not any(f"/{method}/" in path_str for method in args.method):
        return False
    if args.path_contains and not all(fragment in path_str for fragment in args.path_contains):
        return False
    return True


def _shards_in_dir(root, min_mtime=None):
    groups = {}
    stale_paths = []
    for path in root.iterdir():
        match = SHARD_RE.match(path.name)
        if not match:
            continue
        shard_idx, num_shards = map(int, match.groups())
        if min_mtime is not None and path.stat().st_mtime < min_mtime:
            stale_paths.append(path)
            continue
        groups.setdefault(num_shards, {})[shard_idx] = path
    return groups, stale_paths


def _group_summary(groups):
    parts = []
    for num_shards, shards in sorted(groups.items()):
        indices = ",".join(str(idx) for idx in sorted(shards))
        parts.append(f"of_{num_shards}=[{indices}]")
    return " ".join(parts)


def _load_json_list(path):
    with path.open("r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return [data]


def _write_combined(shard_paths, out_path, copy_single_shard):
    if copy_single_shard and len(shard_paths) == 1:
        shutil.copyfile(shard_paths[0], out_path)
        return None

    merged = []
    for shard_path in shard_paths:
        merged.extend(_load_json_list(shard_path))
    with out_path.open("w") as f:
        json.dump(merged, f, indent=2)
    return len(merged)


def combine_shards(args):
    base = Path(args.base_dir)
    min_mtime = _parse_min_mtime(args.min_mtime)
    stats = {
        "written": 0,
        "would_write": 0,
        "skipped_existing": 0,
        "skipped_filtered": 0,
        "skipped_stale_shard": 0,
        "skipped_incomplete": 0,
        "skipped_ambiguous": 0,
    }

    for root, dirs, files in os.walk(base):
        root_path = Path(root)
        if not args.include_calibration and "calibration" in root_path.parts:
            continue

        groups, stale_paths = _shards_in_dir(root_path, min_mtime=min_mtime)
        if not groups:
            if stale_paths:
                stats["skipped_stale_shard"] += 1
                oldest = min(stale_paths, key=lambda path: path.stat().st_mtime)
                print(f"stale     oldest_mtime={_mtime_iso(oldest)} path={root_path}")
            continue
        if not _matches_filters(root_path, args):
            stats["skipped_filtered"] += 1
            continue
        if len(groups) != 1:
            stats["skipped_ambiguous"] += 1
            print(f"ambiguous  {_group_summary(groups)} path={root_path}")
            continue

        num_shards, shards = next(iter(groups.items()))
        missing = sorted(set(range(num_shards)) - set(shards))
        if missing:
            stats["skipped_incomplete"] += 1
            print(f"incomplete missing={missing} of={num_shards} path={root_path}")
            continue

        shard_paths = [shards[idx] for idx in range(num_shards)]
        out_path = root_path / args.output_name
        newest_shard_mtime = max(path.stat().st_mtime for path in shard_paths)
        if out_path.exists():
            if args.overwrite_old_only and out_path.stat().st_mtime >= newest_shard_mtime:
                stats["skipped_existing"] += 1
                continue
            if not args.overwrite and not args.overwrite_old_only:
                stats["skipped_existing"] += 1
                print(f"exists    mtime={_mtime_iso(out_path)} path={out_path}")
                continue

        action = "write" if args.write else "would_write"
        print(
            f"{action:11} shards={num_shards} newest_shard_mtime="
            f"{datetime.fromtimestamp(newest_shard_mtime).isoformat(timespec='seconds')} "
            f"path={out_path}"
        )
        if args.write:
            count = _write_combined(shard_paths, out_path, args.copy_single_shard)
            stats["written"] += 1
            if count is not None:
                print(f"wrote     records={count} path={out_path}")
        else:
            stats["would_write"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Materialize combined.json files from qualitative_prompt_output shard files."
    )
    parser.add_argument("--base_dir", default=RESULTS_ROOT)
    parser.add_argument("--output_name", default="combined.json")
    parser.add_argument("--seed", action="append", default=[], help="Seed number to include. Repeatable.")
    parser.add_argument("--model", action="append", default=[], help="Parsed model dir to include. Repeatable.")
    parser.add_argument("--dataset", action="append", default=[], help="Dataset dir to include. Repeatable.")
    parser.add_argument("--method", action="append", default=[], help="Method dir to include. Repeatable.")
    parser.add_argument("--path_contains", action="append", default=[], help="Required path substring. Repeatable.")
    parser.add_argument("--min_mtime", default=None, help="Only combine shard files newer than this ISO timestamp or epoch seconds.")
    parser.add_argument("--write", action="store_true", help="Actually write combined.json. Without this, only prints what would happen.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing combined.json files.")
    parser.add_argument("--overwrite_old_only", action="store_true", help="Overwrite only if combined.json is older than the newest shard.")
    parser.add_argument("--copy_single_shard", action="store_true", help="For one-shard runs, copy the shard file instead of parsing and re-dumping JSON.")
    parser.add_argument("--include_calibration", action="store_true")
    args = parser.parse_args()

    stats = combine_shards(args)
    print("Summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

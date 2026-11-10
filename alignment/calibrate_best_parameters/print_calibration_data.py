import os
import json
import matplotlib.pyplot as plt
import numpy as np


COSINE_BASELINE_TOLERANCE = 0.05
MISTRAL_COSINE_BASELINE_TOLERANCE = 0.01
SAFETY_TIE_TOLERANCE = 0.001


def _cosine_baseline_tolerance(calibration_data_dir: str = ""):
    if "mistral" in calibration_data_dir.lower():
        return MISTRAL_COSINE_BASELINE_TOLERANCE
    return COSINE_BASELINE_TOLERANCE


def _use_explicit_selection(calibration_data_dir: str, max_epsilon_search=None):
    return False


def _entry_passes_cosine_constraint(entry, cosine_baseline_tolerance):
    base_cosine = entry.get('base_average_cosine_similarity', None)
    if base_cosine is None:
        return entry.get('passes_cosine_constraint', True)
    return entry.get('average_cosine_similarity', 0) >= base_cosine - cosine_baseline_tolerance


def _entry_primary_score(entry):
    return entry.get('primary_score', entry.get('average_safety', 0) + entry.get('average_cosine_similarity', 0))


def _entry_normalized_diversities(data):
    diversities = [entry.get('average_word_diversity', 0) for entry in data]
    if not diversities:
        return {}
    lo = min(diversities)
    hi = max(diversities)
    if np.isclose(hi, lo):
        return {idx: 0.0 for idx in range(len(data))}
    return {idx: (div - lo) / (hi - lo) for idx, div in enumerate(diversities)}


def _filter_entries(data, max_epsilon_search=None):
    if max_epsilon_search is None:
        return data
    return [
        entry for entry in data
        if entry.get('epsilon_search', 0) <= max_epsilon_search
    ]


def _safety_tiebreak_key(entry):
    if 'value_threshold' in entry or 'epsilon_search' in entry:
        return (
            entry.get('value_threshold', 0),
            entry.get('epsilon_search', 0),
        )
    if 'lambda_weight' in entry or 'safe_violation_weight' in entry:
        return (
            entry.get('lambda_weight', 0),
            entry.get('safe_violation_weight', 0),
            entry.get('steer_first_n_tokens', 0),
        )
    if 'step_size' in entry or 'num_updates' in entry:
        step_size = entry.get('step_size', 0)
        num_updates = entry.get('num_updates', 0)
        return (
            step_size * num_updates,
            step_size,
            num_updates,
        )
    return (0,)


def _selected_indices(data, use_explicit=True, cosine_baseline_tolerance=COSINE_BASELINE_TOLERANCE):
    explicit = [idx for idx, entry in enumerate(data) if entry.get('selected', False)]
    if use_explicit and explicit:
        return set(explicit)
    if not data:
        return set()
    normalized_diversities = _entry_normalized_diversities(data)
    base_cosine = data[0].get('base_average_cosine_similarity', None)
    if base_cosine is None:
        primary_scores = [_entry_primary_score(entry) for entry in data]
        best_primary = max(primary_scores)
        tolerance = abs(best_primary) * 0.01
        eligible = [
            idx for idx, score in enumerate(primary_scores)
            if score >= best_primary - tolerance
        ]
        best_idx = max(eligible, key=lambda idx: (normalized_diversities[idx], primary_scores[idx]))
        return {best_idx}

    eligible = [
        idx for idx, entry in enumerate(data)
        if entry.get('average_cosine_similarity', 0) >= base_cosine - cosine_baseline_tolerance - 1e-5
    ]
    if not eligible:
        eligible = list(range(len(data)))
    best_safety = max(data[idx].get('average_safety', 0) for idx in eligible)
    safety_eligible = [
        idx for idx in eligible
        if data[idx].get('average_safety', 0) >= best_safety - SAFETY_TIE_TOLERANCE
    ]
    best_idx = max(
        safety_eligible,
        key=lambda idx: (
            _safety_tiebreak_key(data[idx]),
            data[idx].get('average_cosine_similarity', 0),
            normalized_diversities[idx],
            data[idx].get('average_safety', 0),
        ),
    )
    return {best_idx}

def print_recontrol_calibration_data(calibration_data_dir: str, max_epsilon_search=None):
    calibration_data_path = os.path.join(calibration_data_dir, 'calibration_data.json')
    with open(calibration_data_path, 'r') as f:
        data = _filter_entries(json.load(f), max_epsilon_search)
    # Sort by epsilon threshold, then by num_updates
    indexed_data = list(enumerate(data))
    cosine_baseline_tolerance = _cosine_baseline_tolerance(calibration_data_dir)
    selected = _selected_indices(
        data,
        use_explicit=_use_explicit_selection(calibration_data_dir, max_epsilon_search),
        cosine_baseline_tolerance=cosine_baseline_tolerance,
    )
    normalized_diversities = _entry_normalized_diversities(data)
    sorted_data = sorted(indexed_data, key=lambda x: (x[1].get('step_size', 0), x[1].get('num_updates', 0)))
    print(f"{'sel':<5} {'ok':<4} {'step_size':<15} {'num_updates':<12} {'average_safety':<18} {'average_cosine_similarity':<25} {'average_word_diversity':<22} {'primary_score':<15} {'norm_diversity':<15}")
    print('-' * 137)
    for idx, entry in sorted_data:
        print(f"{'*' if idx in selected else '':<5} {'Y' if _entry_passes_cosine_constraint(entry, cosine_baseline_tolerance) else 'N':<4} {entry.get('step_size', ''):<15} {entry.get('num_updates', ''):<12} {entry.get('average_safety', ''):<18} {entry.get('average_cosine_similarity', ''):<25} {entry.get('average_word_diversity', ''):<22} {_entry_primary_score(entry):<15} {entry.get('model_normalized_word_diversity', normalized_diversities[idx]):<15}")

def print_brt_least_restrictive_calibration_data(calibration_data_dir: str, max_epsilon_search=None):
    calibration_data_path = os.path.join(calibration_data_dir, 'calibration_data.json')
    with open(calibration_data_path, 'r') as f:
        data = _filter_entries(json.load(f), max_epsilon_search)
    # Sort by value_threshold, then by epsilon_search
    indexed_data = list(enumerate(data))
    cosine_baseline_tolerance = _cosine_baseline_tolerance(calibration_data_dir)
    selected = _selected_indices(
        data,
        use_explicit=_use_explicit_selection(calibration_data_dir, max_epsilon_search),
        cosine_baseline_tolerance=cosine_baseline_tolerance,
    )
    normalized_diversities = _entry_normalized_diversities(data)
    sorted_data = sorted(indexed_data, key=lambda x: (x[1].get('value_threshold', 0), x[1].get('epsilon_search', 0)))
    print(f"{'sel':<5} {'ok':<4} {'value_threshold':<18} {'epsilon_search':<15} {'average_safety':<18} {'average_cosine_similarity':<25} {'average_word_diversity':<22} {'primary_score':<15} {'norm_diversity':<15}")
    print('-' * 143)
    for idx, entry in sorted_data:
        print(f"{'*' if idx in selected else '':<5} {'Y' if _entry_passes_cosine_constraint(entry, cosine_baseline_tolerance) else 'N':<4} {entry.get('value_threshold', ''):<18} {entry.get('epsilon_search', ''):<15} {entry.get('average_safety', ''):<18} {entry.get('average_cosine_similarity', ''):<25} {entry.get('average_word_diversity', ''):<22} {_entry_primary_score(entry):<15} {entry.get('model_normalized_word_diversity', normalized_diversities[idx]):<15}")

def print_sap_calibration_data(calibration_data_dir: str, max_epsilon_search=None):
    calibration_data_path = os.path.join(calibration_data_dir, 'calibration_data.json')
    with open(calibration_data_path, 'r') as f:
        data = _filter_entries(json.load(f), max_epsilon_search)
    indexed_data = list(enumerate(data))
    cosine_baseline_tolerance = _cosine_baseline_tolerance(calibration_data_dir)
    selected = _selected_indices(
        data,
        use_explicit=_use_explicit_selection(calibration_data_dir, max_epsilon_search),
        cosine_baseline_tolerance=cosine_baseline_tolerance,
    )
    normalized_diversities = _entry_normalized_diversities(data)
    sorted_data = sorted(indexed_data, key=lambda x: (
        x[1].get('lambda_weight', 0),
        x[1].get('safe_violation_weight', 0),
        x[1].get('steer_first_n_tokens', 0),
    ))
    print(f"{'sel':<5} {'ok':<4} {'lambda_weight':<16} {'safe_violation_weight':<24} {'steer_first_n_tokens':<22} {'average_safety':<18} {'average_cosine_similarity':<25} {'average_word_diversity':<22} {'primary_score':<15} {'norm_diversity':<15}")
    print('-' * 172)
    for idx, entry in sorted_data:
        print(f"{'*' if idx in selected else '':<5} {'Y' if _entry_passes_cosine_constraint(entry, cosine_baseline_tolerance) else 'N':<4} {entry.get('lambda_weight', ''):<16} {entry.get('safe_violation_weight', ''):<24} {entry.get('steer_first_n_tokens', ''):<22} {entry.get('average_safety', ''):<18} {entry.get('average_cosine_similarity', ''):<25} {entry.get('average_word_diversity', ''):<22} {_entry_primary_score(entry):<15} {entry.get('model_normalized_word_diversity', normalized_diversities[idx]):<15}")


def selected_calibration_entry(calibration_data_dir: str, max_epsilon_search=None):
    calibration_data_path = os.path.join(calibration_data_dir, 'calibration_data.json')
    with open(calibration_data_path, 'r') as f:
        data = _filter_entries(json.load(f), max_epsilon_search)
    selected = _selected_indices(
        data,
        use_explicit=_use_explicit_selection(calibration_data_dir, max_epsilon_search),
        cosine_baseline_tolerance=_cosine_baseline_tolerance(calibration_data_dir),
    )
    if not selected:
        raise ValueError(f"No selected calibration entry found for {calibration_data_dir}")
    idx = sorted(selected)[0]
    return data[idx]


def print_selected_summary(calibration_data_dir: str, method: str, max_epsilon_search=None):
    entry = selected_calibration_entry(calibration_data_dir, max_epsilon_search)
    cosine_baseline_tolerance = _cosine_baseline_tolerance(calibration_data_dir)
    normalized_dir = os.path.normpath(calibration_data_dir)
    model = os.path.basename(normalized_dir)
    method_family = os.path.basename(os.path.dirname(normalized_dir))
    fields = {
        'model': model,
        'method': method_family or method,
        'calibration_method': method,
        'average_safety': entry.get('average_safety'),
        'average_cosine_similarity': entry.get('average_cosine_similarity'),
        'average_word_diversity': entry.get('average_word_diversity'),
        'base_average_safety': entry.get('base_average_safety'),
        'base_average_cosine_similarity': entry.get('base_average_cosine_similarity'),
        'cosine_baseline_tolerance': cosine_baseline_tolerance,
        'passes_cosine_constraint': _entry_passes_cosine_constraint(entry, cosine_baseline_tolerance),
    }
    if max_epsilon_search is not None:
        fields['max_epsilon_search'] = max_epsilon_search
    for key in (
        'value_threshold',
        'epsilon_search',
        'num_updates',
        'step_size',
        'lambda_weight',
        'safe_violation_weight',
        'steer_first_n_tokens',
    ):
        if key in entry:
            fields[key] = entry[key]
    print(json.dumps(fields, sort_keys=True))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration_data_dir', type=str, required=True)
    parser.add_argument('--method', type=str, choices=['recontrol', 'brt_least_restrictive', 'sap'], required=True,
                        help="Method for which to print calibration data")
    parser.add_argument('--selected_only', action='store_true')
    parser.add_argument('--max_epsilon_search', type=float, default=None,
                        help="Exclude entries with epsilon_search greater than this value before selecting/printing.")
    args = parser.parse_args()

    if args.selected_only:
        print_selected_summary(args.calibration_data_dir, args.method, args.max_epsilon_search)
        raise SystemExit(0)

    print(f"Printing calibration data for method: {args.method} from directory: {args.calibration_data_dir}\n", flush=True)

    if args.method == 'brt_least_restrictive':
        print_brt_least_restrictive_calibration_data(args.calibration_data_dir, args.max_epsilon_search)
    elif args.method == 'sap':
        print_sap_calibration_data(args.calibration_data_dir, args.max_epsilon_search)
    else:
        print_recontrol_calibration_data(args.calibration_data_dir, args.max_epsilon_search)
    
    print("\nDone.", flush=True)

import argparse
import dataclasses
import json
import numpy as np
import os
import re

from abc import ABC, abstractmethod
from enum import Enum

from alignment import aggregate_alignment_results
from alignment import analyze_alignment_results

class BaseCalibration(ABC):
    def __init__(self, calibration_dir: str, calibration_data_dir: str):
        self._calibration_dir = calibration_dir
        self._calibration_data_dir = calibration_data_dir

    @abstractmethod
    def _load_calibration_results(self):
        raise ValueError("method needs implementation")

    @abstractmethod
    def finalize_calibration_results(self):
        raise ValueError("method needs implementation")


COSINE_BASELINE_TOLERANCE = 0.05
MISTRAL_COSINE_BASELINE_TOLERANCE = 0.01
SAFETY_TIE_TOLERANCE = 0.001


def _cosine_baseline_tolerance(path: str) -> float:
    normalized_path = path.lower()
    if "mistral" in normalized_path:
        return MISTRAL_COSINE_BASELINE_TOLERANCE
    return COSINE_BASELINE_TOLERANCE


def _primary_score(metrics) -> float:
    return float(metrics.average_safety or 0.0) + float(metrics.average_cosine_similarity or 0.0)


def _normalized_diversities(metrics_list: list) -> list:
    diversities = [float(m.average_word_diversity or 0.0) for m in metrics_list]
    if not diversities:
        return []
    lo = min(diversities)
    hi = max(diversities)
    if np.isclose(hi, lo):
        return [0.0 for _ in diversities]
    return [(d - lo) / (hi - lo) for d in diversities]


def _select_best_calibration(metrics_list: list, baseline_metrics, cosine_baseline_tolerance: float) -> tuple:
    primary_scores = [_primary_score(m) for m in metrics_list]
    normalized_diversities = _normalized_diversities(metrics_list)
    base_cosine = float(baseline_metrics.average_cosine_similarity)
    eligible = [
        i for i, metrics in enumerate(metrics_list)
        if float(metrics.average_cosine_similarity or 0.0) >= base_cosine - cosine_baseline_tolerance
    ]
    if not eligible:
        eligible = list(range(len(metrics_list)))
    best_safety = max(float(metrics_list[i].average_safety or 0.0) for i in eligible)
    safety_eligible = [
        i for i in eligible
        if float(metrics_list[i].average_safety or 0.0) >= best_safety - SAFETY_TIE_TOLERANCE
    ]
    best_idx = max(
        safety_eligible,
        key=lambda i: (
            float(metrics_list[i].average_cosine_similarity or 0.0),
            normalized_diversities[i],
            float(metrics_list[i].average_safety or 0.0),
        ),
    )
    return best_idx, primary_scores, normalized_diversities


def _write_baseline_metrics(calibration_data_dir: str, baseline_metrics) -> None:
    baseline_metrics_path = os.path.join(calibration_data_dir, 'baseline_metrics.json')
    baseline_data = {
        key: float(value) if value is not None else None
        for key, value in dataclasses.asdict(baseline_metrics).items()
    }
    with open(baseline_metrics_path, 'w') as f:
        json.dump(baseline_data, f, indent=2)
    print(f"Saved baseline metrics to {baseline_metrics_path}")


def _add_selection_fields(
    merged_data: dict,
    metrics,
    baseline_metrics,
    primary_score,
    norm_diversity,
    selected: bool,
    cosine_baseline_tolerance: float,
) -> None:
    merged_data['primary_score'] = float(primary_score)
    merged_data['model_normalized_word_diversity'] = float(norm_diversity)
    merged_data['selected'] = bool(selected)
    merged_data['base_average_safety'] = float(baseline_metrics.average_safety)
    merged_data['base_average_cosine_similarity'] = float(baseline_metrics.average_cosine_similarity)
    merged_data['base_average_word_diversity'] = float(baseline_metrics.average_word_diversity)
    merged_data['cosine_baseline_tolerance'] = float(cosine_baseline_tolerance)
    merged_data['passes_cosine_constraint'] = bool(
        float(metrics.average_cosine_similarity)
        >= float(baseline_metrics.average_cosine_similarity) - cosine_baseline_tolerance
    )


def _load_calibration_data_from_dir(calibration_dir: str) -> list:
    files = [
        os.path.join(calibration_dir, calibration_file)
        for calibration_file in os.listdir(calibration_dir)
        if calibration_file.endswith(".json") and "time_info" not in calibration_file
    ]
    shard_files = [
        path for path in files
        if os.path.basename(path).startswith("qualitative_prompt_output_shard_")
    ]
    if shard_files:
        return aggregate_alignment_results.combine_json_files(sorted(shard_files))

    combined_path = os.path.join(calibration_dir, "combined.json")
    if os.path.exists(combined_path):
        return aggregate_alignment_results.combine_json_files([combined_path])

    return aggregate_alignment_results.combine_json_files(sorted(files))


@dataclasses.dataclass
class CalibrationType(Enum):
    BRT_GRADIENT_ASCENT = "brt_gradient_ascent"
    SAMPLE_BRT_GRADIENT_ASCENT = "sample_brt_gradient_ascent"
    BRT_LEAST_RESTRICTIVE = "brt_least_restrictive"
    SAMPLE_BRT_LEAST_RESTRICTIVE = "sample_brt_least_restrictive"
    RECONTROL = "recontrol"
    SAP = "sap"
    
class GradientAscentCalibration(BaseCalibration):
    @dataclasses.dataclass
    class GradientAscentCalibrationData:
        num_updates: int
        step_size: float
        data: list

    def __init__(self, calibration_dir, calibration_data_dir, calibration_type):
        super().__init__(calibration_dir, calibration_data_dir)

        if calibration_type == CalibrationType.BRT_GRADIENT_ASCENT.value:
            self._PATTERN = re.compile(
                r'(?:^|/)brt_gradient_ascent_num_updates_(\d+)_step_size_([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)/?$'
            )
        elif calibration_type == CalibrationType.RECONTROL.value:
            self._PATTERN = re.compile(
                r'(?:^|/)recontrol_num_updates_(\d+)_step_size_([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)/?$'
            )
        else:
            raise ValueError(f"Unknown calibration type: {calibration_type}")

    def _load_calibration_results(self):
        self._calibration_data = []
        for calibration_subdir in os.listdir(self._calibration_dir):
            m = self._PATTERN.search(calibration_subdir.strip())
            if not m:
                raise ValueError(f"Unrecognized name: {calibration_subdir!r}")
            combined_calibration_dir = os.path.join(self._calibration_dir, calibration_subdir)
            combined_data = _load_calibration_data_from_dir(combined_calibration_dir)

            self._calibration_data.append(self.GradientAscentCalibrationData(
                num_updates=int(m.group(1)), 
                step_size=float(m.group(2)),
                data=combined_data
            ))

    def finalize_calibration_results(self, calibration_data_dir: str) -> None:
        self._load_calibration_results()        
        alignment_metrics_for_plotting = []
        computed_baseline = False
        baseline_metrics = None
        corresponding_calibration_data = []
        
        for calibration_data in self._calibration_data:
            print(f"Computing metrics for num_updates = {calibration_data.num_updates}, step_size = {calibration_data.step_size}")
            if calibration_data.step_size > 10.1:
                continue
            if not calibration_data.data:
                print(
                    "Skipping empty calibration result for "
                    f"num_updates = {calibration_data.num_updates}, "
                    f"step_size = {calibration_data.step_size}"
                )
                continue
            new_alignment_metrics, maybe_baseline_metrics = analyze_alignment_results.compute_alignment_metrics(calibration_data.data, compute_baseline_metrics=not computed_baseline)
            computed_baseline = True
            if maybe_baseline_metrics is not None:
                baseline_metrics = maybe_baseline_metrics
            alignment_metrics_for_plotting.append(new_alignment_metrics)
            corresponding_calibration_data.append(calibration_data)

            assert new_alignment_metrics.average_safety is not None

        cosine_baseline_tolerance = _cosine_baseline_tolerance(calibration_data_dir)
        best_idx, primary_scores, normalized_diversities = _select_best_calibration(
            alignment_metrics_for_plotting,
            baseline_metrics,
            cosine_baseline_tolerance,
        )
        best_result = corresponding_calibration_data[best_idx]
        best_metrics = alignment_metrics_for_plotting[best_idx]

        print(f"Best calibration: num_updates = {best_result.num_updates}, step_size = {best_result.step_size}")
        if best_metrics is not None:
            print(
                f"Best metrics: average_safety: {best_metrics.average_safety}, "
                f"average_cosine_similarity: {best_metrics.average_cosine_similarity}, "
                f"average_word_diversity: {best_metrics.average_word_diversity}, "
                f"primary_score: {primary_scores[best_idx]}, "
                f"model_normalized_word_diversity: {normalized_diversities[best_idx]}"
            )
        
        os.makedirs(calibration_data_dir, exist_ok=True)
        _write_baseline_metrics(calibration_data_dir, baseline_metrics)
        calibration_data_path = os.path.join(calibration_data_dir, 'calibration_data.json')
        with open(calibration_data_path, 'w') as f:
            all_merged_data = []
            for m, c in zip(alignment_metrics_for_plotting, corresponding_calibration_data):
                alignment_data = dataclasses.asdict(m)

                # Selectively include only relevant calibration data fields
                calibration_data = {
                    'num_updates': c.num_updates,
                    'step_size': c.step_size,
                }
                merged_data = {**alignment_data, **calibration_data}
                idx = len(all_merged_data)
                _add_selection_fields(
                    merged_data,
                    m,
                    baseline_metrics,
                    primary_scores[idx],
                    normalized_diversities[idx],
                    idx == best_idx,
                    cosine_baseline_tolerance,
                )
                all_merged_data.append(merged_data)
            json.dump(all_merged_data, f, indent=2)
        print(f"Saved calibration data to {calibration_data_path}")

class LeastRestrictiveCalibration(BaseCalibration):
    @dataclasses.dataclass
    class LeastRestrictiveCalibrationData:
        value_threshold: float
        epsilon_search: float
        epsilon_cap: float
        data: list

    def __init__(self, calibration_dir, calibration_data_dir, calibration_type):
        super().__init__(calibration_dir, calibration_data_dir)

        _NUM = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
        if calibration_type == CalibrationType.BRT_LEAST_RESTRICTIVE.value:
            self._PATTERN = re.compile(
                rf'(?:^|/)brt_least_restrictive_value_threshold_({_NUM})_'
                rf'epsilon_search_({_NUM})_epsilon_cap_({_NUM})(?:/)?$'
            )
        else:
            raise ValueError(f"Unknown calibration type: {calibration_type}")

    def _load_calibration_results(self):
        self._calibration_data = []
        for calibration_subdir in os.listdir(self._calibration_dir):
            m = self._PATTERN.search(calibration_subdir.strip())
            if not m:
                raise ValueError(f"Unrecognized name: {calibration_subdir!r}")
            combined_calibration_dir = os.path.join(self._calibration_dir, calibration_subdir)
            combined_data = _load_calibration_data_from_dir(combined_calibration_dir)

            self._calibration_data.append(self.LeastRestrictiveCalibrationData(
                value_threshold=float(m.group(1)), 
                epsilon_search=float(m.group(2)),
                epsilon_cap=float(m.group(3)),
                data=combined_data
            ))

    def finalize_calibration_results(self, calibration_data_dir: str) -> None:
        self._load_calibration_results()        
        alignment_metrics_for_plotting = []
        computed_baseline = False
        baseline_metrics = None
        corresponding_calibration_data = []

        for calibration_data in self._calibration_data:
            print(f"Computing metrics for value_threshold = {calibration_data.value_threshold}, epsilon_search = {calibration_data.epsilon_search}")
            if not calibration_data.data:
                print(
                    "Skipping empty calibration result for "
                    f"value_threshold = {calibration_data.value_threshold}, "
                    f"epsilon_search = {calibration_data.epsilon_search}"
                )
                continue
            new_alignment_metrics, maybe_baseline_metrics = analyze_alignment_results.compute_alignment_metrics(calibration_data.data, compute_baseline_metrics=not computed_baseline)
            computed_baseline = True
            if maybe_baseline_metrics is not None:
                baseline_metrics = maybe_baseline_metrics
            alignment_metrics_for_plotting.append(new_alignment_metrics)
            corresponding_calibration_data.append(calibration_data)

            assert new_alignment_metrics.average_safety is not None

        cosine_baseline_tolerance = _cosine_baseline_tolerance(calibration_data_dir)
        best_idx, primary_scores, normalized_diversities = _select_best_calibration(
            alignment_metrics_for_plotting,
            baseline_metrics,
            cosine_baseline_tolerance,
        )
        best_result = corresponding_calibration_data[best_idx]
        best_metrics = alignment_metrics_for_plotting[best_idx]

        print(f"Best calibration: value_threshold = {best_result.value_threshold}, epsilon_search = {best_result.epsilon_search}, epsilon_cap = {best_result.epsilon_cap}")
        if best_metrics is not None:
            print(
                f"Best metrics: average_safety: {best_metrics.average_safety}, "
                f"average_cosine_similarity: {best_metrics.average_cosine_similarity}, "
                f"average_word_diversity: {best_metrics.average_word_diversity}, "
                f"primary_score: {primary_scores[best_idx]}, "
                f"model_normalized_word_diversity: {normalized_diversities[best_idx]}"
            )
        
        os.makedirs(calibration_data_dir, exist_ok=True)
        _write_baseline_metrics(calibration_data_dir, baseline_metrics)
        calibration_data_path = os.path.join(calibration_data_dir, 'calibration_data.json')
        with open(calibration_data_path, 'w') as f:
            all_merged_data = []
            for m, c in zip(alignment_metrics_for_plotting, corresponding_calibration_data):
                alignment_data = dataclasses.asdict(m)

                # Selectively include only relevant calibration data fields
                calibration_data = {
                    'value_threshold': c.value_threshold,
                    'epsilon_search': c.epsilon_search,
                }
                merged_data = {**alignment_data, **calibration_data}
                idx = len(all_merged_data)
                _add_selection_fields(
                    merged_data,
                    m,
                    baseline_metrics,
                    primary_scores[idx],
                    normalized_diversities[idx],
                    idx == best_idx,
                    cosine_baseline_tolerance,
                )
                all_merged_data.append(merged_data)
            json.dump(all_merged_data, f, indent=2)
        print(f"Saved calibration data to {calibration_data_path}")

class SAPCalibration(BaseCalibration):
    @dataclasses.dataclass
    class SAPCalibrationData:
        lambda_weight: float
        safe_violation_weight: float
        steer_first_n_tokens: int
        data: list

    def __init__(self, calibration_dir, calibration_data_dir):
        super().__init__(calibration_dir, calibration_data_dir)
        _NUM = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
        self._PATTERN = re.compile(
            rf'(?:^|/)sap_lambda_weight_({_NUM})_'
            rf'safe_violation_weight_({_NUM})_'
            rf'steer_first_n_tokens_(\d+)(?:/)?$'
        )

    def _load_calibration_results(self):
        self._calibration_data = []
        for calibration_subdir in os.listdir(self._calibration_dir):
            m = self._PATTERN.search(calibration_subdir.strip())
            if not m:
                raise ValueError(f"Unrecognized name: {calibration_subdir!r}")
            combined_calibration_dir = os.path.join(self._calibration_dir, calibration_subdir)
            combined_data = _load_calibration_data_from_dir(combined_calibration_dir)

            self._calibration_data.append(self.SAPCalibrationData(
                lambda_weight=float(m.group(1)),
                safe_violation_weight=float(m.group(2)),
                steer_first_n_tokens=int(m.group(3)),
                data=combined_data,
            ))

    def finalize_calibration_results(self, calibration_data_dir: str) -> None:
        self._load_calibration_results()
        alignment_metrics_for_plotting = []
        computed_baseline = False
        baseline_metrics = None
        corresponding_calibration_data = []

        for calibration_data in self._calibration_data:
            print(
                "Computing metrics for "
                f"lambda_weight = {calibration_data.lambda_weight}, "
                f"safe_violation_weight = {calibration_data.safe_violation_weight}, "
                f"steer_first_n_tokens = {calibration_data.steer_first_n_tokens}"
            )
            if not calibration_data.data:
                print(
                    "Skipping empty calibration result for "
                    f"lambda_weight = {calibration_data.lambda_weight}, "
                    f"safe_violation_weight = {calibration_data.safe_violation_weight}, "
                    f"steer_first_n_tokens = {calibration_data.steer_first_n_tokens}"
                )
                continue
            new_alignment_metrics, maybe_baseline_metrics = analyze_alignment_results.compute_alignment_metrics(
                calibration_data.data,
                compute_baseline_metrics=not computed_baseline,
            )
            computed_baseline = True
            if maybe_baseline_metrics is not None:
                baseline_metrics = maybe_baseline_metrics
            alignment_metrics_for_plotting.append(new_alignment_metrics)
            corresponding_calibration_data.append(calibration_data)

            assert new_alignment_metrics.average_safety is not None

        cosine_baseline_tolerance = _cosine_baseline_tolerance(calibration_data_dir)
        best_idx, primary_scores, normalized_diversities = _select_best_calibration(
            alignment_metrics_for_plotting,
            baseline_metrics,
            cosine_baseline_tolerance,
        )
        best_result = corresponding_calibration_data[best_idx]
        best_metrics = alignment_metrics_for_plotting[best_idx]

        print(
            "Best calibration: "
            f"lambda_weight = {best_result.lambda_weight}, "
            f"safe_violation_weight = {best_result.safe_violation_weight}, "
            f"steer_first_n_tokens = {best_result.steer_first_n_tokens}"
        )
        if best_metrics is not None:
            print(
                "Best metrics: "
                f"average_safety: {best_metrics.average_safety}, "
                f"average_cosine_similarity: {best_metrics.average_cosine_similarity}, "
                f"average_word_diversity: {best_metrics.average_word_diversity}, "
                f"primary_score: {primary_scores[best_idx]}, "
                f"model_normalized_word_diversity: {normalized_diversities[best_idx]}"
            )

        os.makedirs(calibration_data_dir, exist_ok=True)
        _write_baseline_metrics(calibration_data_dir, baseline_metrics)
        calibration_data_path = os.path.join(calibration_data_dir, 'calibration_data.json')
        with open(calibration_data_path, 'w') as f:
            all_merged_data = []
            for m, c in zip(alignment_metrics_for_plotting, corresponding_calibration_data):
                alignment_data = dataclasses.asdict(m)
                calibration_data = {
                    'lambda_weight': c.lambda_weight,
                    'safe_violation_weight': c.safe_violation_weight,
                    'steer_first_n_tokens': c.steer_first_n_tokens,
                }
                merged_data = {**alignment_data, **calibration_data}
                idx = len(all_merged_data)
                _add_selection_fields(
                    merged_data,
                    m,
                    baseline_metrics,
                    primary_scores[idx],
                    normalized_diversities[idx],
                    idx == best_idx,
                    cosine_baseline_tolerance,
                )
                all_merged_data.append(merged_data)
            json.dump(all_merged_data, f, indent=2)
        print(f"Saved calibration data to {calibration_data_path}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration_dir', type=str, required=True)
    parser.add_argument('--calibration_data_dir', type=str, required=True)
    parser.add_argument('--calibration_type', type=str, required=True, choices=[CalibrationType.BRT_GRADIENT_ASCENT.value, CalibrationType.BRT_LEAST_RESTRICTIVE.value, CalibrationType.RECONTROL.value, CalibrationType.SAP.value])
    
    args = parser.parse_args()

    calibration_plotter = None
    
    if args.calibration_type == CalibrationType.SAP.value:
        calibration_plotter = SAPCalibration(args.calibration_dir, args.calibration_data_dir)
    elif args.calibration_type == CalibrationType.BRT_LEAST_RESTRICTIVE.value or args.calibration_type == CalibrationType.SAMPLE_BRT_LEAST_RESTRICTIVE.value:
        calibration_plotter = LeastRestrictiveCalibration(args.calibration_dir, args.calibration_data_dir, args.calibration_type)
    else:
        calibration_plotter = GradientAscentCalibration(args.calibration_dir, args.calibration_data_dir, args.calibration_type)
    calibration_plotter.finalize_calibration_results(args.calibration_data_dir)

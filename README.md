# BRT-Align Reproducibility Code

**Preemptive Detection and Steering of LLM Misalignment via Latent Reachability**\
Sathwik Karnik, Somil Bansal\
Safe and Intelligent Autonomy Lab, Stanford University

Accepted at Transactions on Machine Learning Research (TMLR), 2026.

This supplementary contains the code needed to reproduce the paper's data
generation, monitor training, inference-time alignment, runtime-monitoring
evaluation, and LLM-as-judge evaluation.

## Setup

Install dependencies from `requirements.txt` (Python 3.9.12).

The code uses local paths configured through environment variables:

```bash
export LLM_REACHABILITY_PROJECT_ROOT=<this-directory>
export LLM_REACHABILITY_DATA_ROOT=<generated-data-directory>
export LLM_REACHABILITY_OUTPUT_ROOT=<model-and-result-output-directory>
```

If these variables are unset, outputs default to subdirectories under the current
working directory.

## Data Generation

Dataset generation scripts are in `data/`:

- `generate_beavertails_finetune.py`
- `generate_real_toxicity_finetune.py`
- `generate_ultrasafety_finetune.py`
- `generate_stochastic_beavertails_finetune.py`
- `stack_finetune_shards.py`

Instruction/chat-tuned models use the tokenizer's chat template when available.
Base models are evaluated on raw prompt text.

## Monitor Training

Monitor training scripts are in `toxicity_value_function/`:

- `train_sample_brt_align.py` trains Sample-BRT-Align monitors.
- `train_rl_brt_align.py` trains RL-BRT-Align monitors.

## Alignment

Alignment code is in `alignment/`:

- `fast_experiment_backwards_reachable_alignment.py` runs sharded alignment.
- `interactive_backwards_reachable_alignment.py` runs a single prompt or small
  benchmark.
- `combine_alignment_shards.py` combines sharded outputs.
- `load_and_compute_final_alignment_results.py` computes final alignment metrics.
- `calibrate_best_parameters/print_calibration_data.py` prints calibration
  sweeps and selected hyperparameters.

Aligner implementations are in `alignment/aligners/`.

## Runtime Monitoring

Runtime-monitoring scripts are in `evaluation/`:

- `final_runtime_monitor_evaluation.py` evaluates fixed operating thresholds.
- `final_runtime_monitor_auc_evaluation.py` computes AUROC/AUPRC data.
- `final_text_classifier_baseline_evaluation.py` evaluates prefix-classifier and
  generate-then-classify references.
- `print_reactive_monitor_comparison.py` reports sequence-level miss rate,
  false-alarm rate, first-alert index, lead, and PreemptiveRecall.
- `plot_macro_f1_runtime_monitor_results.py` and
  `plot_runtime_monitor_roc_curves.py` generate summary plots.

## LLM-as-Judge and XSTest

- `evaluation/gemini_alignment_judge.py` prepares and runs the Gemini-based
  safety/coherence judge.
- `evaluation/print_gemini_alignment_judge_results.py` summarizes judge scores
  and win/tie/loss rates.
- `evaluation/evaluate_xstest_official_gpt.py` runs the XSTest over-refusal
  classifier.
- `evaluation/print_xstest_official_results.py` summarizes XSTest results.

## Representative Commands

The commands below show the workflow for one model (`Qwen/Qwen2-1.5B`) and one
training seed. Monitors are trained once per model and seed on the BeaverTails
training split (`330k_train`). The same checkpoints are then used, without
retraining, for runtime monitoring and alignment on all three datasets:
BeaverTails `330k_test`, RealToxicityPrompts `train`, and UltraSafety `train`.
The paper repeats this workflow over all reported models and seeds.

```bash
export LLM_REACHABILITY_PROJECT_ROOT=$PWD
export LLM_REACHABILITY_DATA_ROOT=/path/to/generated_data
export LLM_REACHABILITY_OUTPUT_ROOT=/path/to/outputs

export MODEL=Qwen/Qwen2-1.5B
export SEED=41
export LAYER=20
export TRAIN_DATASET=beavertails_llm_generated_layer_${LAYER}
```

Generate and stack hidden states. BeaverTails needs both `330k_train` (monitor
training) and `330k_test` (evaluation):

```bash
for SPLIT in 330k_train 330k_test; do
  for SHARD in 0 1 2 3 4 5 6 7; do
    python3 data/generate_beavertails_finetune.py \
      --model_name "$MODEL" \
      --split "$SPLIT" \
      --layer_idx "$LAYER" \
      --num_shards 8 \
      --shard_id "$SHARD"
  done

  python3 data/stack_finetune_shards.py \
    --model_name "$MODEL" \
    --dataset_name beavertails \
    --split "$SPLIT" \
    --num_shards 8 \
    --use_layer_dir \
    --layer_idx "$LAYER" \
    --skip_classifier

  python3 data/evaluate_ground_truth_classifier.py \
    --model_name "$MODEL" \
    --root_dir "$LLM_REACHABILITY_DATA_ROOT" \
    --dataset_name beavertails \
    --split "$SPLIT" \
    --embedding_layer_idx "$LAYER"
done
```

RealToxicityPrompts and UltraSafety are used only for evaluation. Their
generators always write the `train` split:

```bash
for DATASET in real_toxicity ultrasafety; do
  for SHARD in 0 1 2 3 4 5 6 7; do
    python3 data/generate_${DATASET}_finetune.py \
      --model_name "$MODEL" \
      --layer_idx "$LAYER" \
      --num_shards 8 \
      --shard_id "$SHARD"
  done

  python3 data/stack_finetune_shards.py \
    --model_name "$MODEL" \
    --dataset_name "$DATASET" \
    --split train \
    --num_shards 8 \
    --use_layer_dir \
    --layer_idx "$LAYER" \
    --skip_classifier

  python3 data/evaluate_ground_truth_classifier.py \
    --model_name "$MODEL" \
    --root_dir "$LLM_REACHABILITY_DATA_ROOT" \
    --dataset_name "$DATASET" \
    --split train \
    --embedding_layer_idx "$LAYER"
done
```

Train Sample-BRT-Align and RL-BRT-Align monitors on BeaverTails:

```bash
python3 toxicity_value_function/train_sample_brt_align.py \
  --model_name "$MODEL" \
  --dataset_name "$TRAIN_DATASET" \
  --data_dir "$LLM_REACHABILITY_DATA_ROOT" \
  --epochs 20 \
  --lr 1e-4 \
  --batch_size 8 \
  --hidden_dims '[16384,64]' \
  --unsafe_weight 2.0 \
  --seed "$SEED"

export SAMPLE_BRT_CKPT=/path/to/sample_brt_checkpoint.pth

python3 toxicity_value_function/train_rl_brt_align.py \
  --model_name "$MODEL" \
  --dataset_name "$TRAIN_DATASET" \
  --data_dir "$LLM_REACHABILITY_DATA_ROOT" \
  --value_model_path "$SAMPLE_BRT_CKPT" \
  --epochs 10 \
  --lr 1e-5 \
  --batch_size 8 \
  --hidden_dims '[16384,64]' \
  --gamma_start 0.99 \
  --gamma_end 0.99 \
  --anneal_epochs 40 \
  --tau 0.005 \
  --unsafe_weight 2.0 \
  --warmup_epochs 0 \
  --pairwise_loss_coeff 0.25 \
  --seed "$SEED"
```

Evaluate runtime monitors on all three datasets with the BeaverTails-trained
checkpoint. The evaluation scripts expand `--split test` to `330k_test` for
BeaverTails:

```bash
export RL_BRT_CKPT=/path/to/rl_brt_checkpoint.pth  # trained on BeaverTails above

for DATASET in beavertails real_toxicity ultrasafety; do
  if [ "$DATASET" = beavertails ]; then SPLIT=test; else SPLIT=train; fi

  python3 evaluation/final_runtime_monitor_evaluation.py \
    --checkpoint_path "$RL_BRT_CKPT" \
    --model_name "$MODEL" \
    --dataset_name "$DATASET" \
    --split "$SPLIT" \
    --seed "$SEED" \
    --method_type BRT \
    --zero_level_label \
    --embedding_layer_idx "$LAYER" \
    --batch_size 512

  python3 evaluation/final_runtime_monitor_auc_evaluation.py \
    --checkpoint_path "$RL_BRT_CKPT" \
    --model_name "$MODEL" \
    --dataset_name "$DATASET" \
    --split "$SPLIT" \
    --seed "$SEED" \
    --method_type BRT \
    --zero_level_label \
    --embedding_layer_idx "$LAYER" \
    --save_token_scores \
    --batch_size 512
done

python3 evaluation/print_reactive_monitor_comparison.py \
  --model_name "$MODEL" \
  --datasets beavertails real_toxicity ultrasafety \
  --seeds "$SEED" \
  --methods BRT \
  --view reliability
```

Run inference-time alignment on all three datasets with the same
BeaverTails-trained checkpoint, then summarize the saved responses:

```bash
for DATASET in beavertails real_toxicity ultrasafety; do
  if [ "$DATASET" = beavertails ]; then SPLIT=330k_test; else SPLIT=train; fi

  python3 alignment/fast_experiment_backwards_reachable_alignment.py \
    --model_name "$MODEL" \
    --dataset_name "$DATASET" \
    --split_name "$SPLIT" \
    --embedding_layer_idx "$LAYER" \
    --layer_idx "$LAYER" \
    --model_ckpt "$RL_BRT_CKPT" \
    --aligner_type brt_least_restrictive \
    --brt_type brt \
    --value_threshold 0.1 \
    --epsilon_search 100.0 \
    --epsilon_cap 80.0 \
    --num_steer_samples 5000 \
    --batch_size 32 \
    --seed "$SEED" \
    --only_unsafe
done

python3 alignment/combine_alignment_shards.py \
  --base_dir "$LLM_REACHABILITY_DATA_ROOT/alignment_results" \
  --seed "$SEED" \
  --model Qwen_Qwen2-1.5B \
  --dataset beavertails \
  --dataset real_toxicity \
  --dataset ultrasafety \
  --method brt \
  --write \
  --overwrite_old_only

python3 alignment/load_and_compute_final_alignment_results.py \
  --model_name "$MODEL" \
  --seed "$SEED" \
  --allow_missing \
  --prefer_shards \
  --force_recompute
```

The same alignment entry point runs the baselines by changing
`--aligner_type` and the corresponding method-specific flags:
`--aligner_type sap` for SaP, `--aligner_type recontrol` for Re-Control, and
`--brt_type sample_brt` with the Sample-BRT checkpoint for Sample-BRT-Align.

## Citation

```bibtex
@article{karnik2026preemptive,
  title   = {Preemptive Detection and Steering of {LLM} Misalignment via Latent Reachability},
  author  = {Karnik, Sathwik and Bansal, Somil},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  url     = {https://arxiv.org/abs/2509.21528}
}
```

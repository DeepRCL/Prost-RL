# ProstNFound Extensions and Innovations (Paper Notes)

This document summarizes the important additions implemented in this repository beyond the original ProstNFound/ProstNFound+ pipeline, with emphasis on RL, DRPO, and new loss designs.

## 1) Baseline Context: What "Normal ProstNFound" Covers

The baseline ProstNFound/ProstNFound+ setup is a weakly supervised prostate ultrasound framework that:

- uses a segmentation-style backbone (MedSAM-based encoder-decoder stack) for lesion heatmaps,
- leverages clinical prompts/metadata (e.g., age, PSA, PSA density, position),
- trains with region-aware supervision (needle/prostate masks),
- outputs both spatial heatmaps and core-level classification.

In this repo, that baseline is extended into an RL-guided variant and multiple optimization/loss innovations.

## 2) RL-Guided ProstNFound (Major System Extension)

Implemented in the RL training/evaluation stack (mainly `prostnfound/train_rl.py`, `prostnfound/test_rl.py`, `medAI/medAI/modeling/prostnfound_rl*.py`):

- **RL attention policy integrated into ProstNFound**: the model samples attention locations/regions that condition downstream prediction.
- **Batched multi-sample rollout design**: multiple policy samples per image are evaluated in one forward path for efficiency and stable within-image comparison.
- **Unified training support for multiple policy-optimization modes**:
  - PPO,
  - GRPO,
  - GDPO,
  - DRPO.

This turns ProstNFound from passive full-image processing into an active attention-selection framework.

## 3) Policy Optimization Innovations: GRPO -> GDPO -> DRPO

### 3.1 GRPO

GRPO is used for group-relative learning without mandatory value function estimation, with within-image normalization over multiple sampled actions.

### 3.2 GDPO (implemented)

`medAI/medAI/modeling/gdpo.py` adds **decoupled reward normalization**:

- reward components are normalized separately (not only after summation),
- then combined with configurable component weights,
- preventing one reward signal from dominating/canceling others.

This is important for multi-objective RL where classification and spatial/attention objectives have very different scales.

### 3.3 DRPO (new implementation focus)

`medAI/medAI/modeling/drpo.py` introduces **Domain-Robust Policy Optimization** as an extension of GDPO:

1. normalize each reward component (GDPO-style),
2. build per-image reward vectors,
3. perform **intra-domain K-means clustering** for difficulty grouping,
4. compute **domain temperature** and **cluster temperature**,
5. apply hierarchical scaling to advantages:
   - scaled advantage is inversely weighted by combined temperature,
6. optional global re-normalization.

In current training integration (`prostnfound/train_rl.py`):

- DRPO mode is selected with `rl_mode=drpo`,
- DRPO receives both reward components and `domain_ids`,
- domains are currently generated from grade/cancer presence (benign vs cancer) to address class imbalance in practice.

Practical goal: emphasize rare/hard samples (especially cancer-rich examples) while preserving stable policy updates.

## 4) Reward Engineering Contributions

Implemented in `prostnfound/src/rl_loss.py`:

- multi-mode reward computation (`combined_v2`, `classification_only`, `roi_only`, `attention_proportional`, `classification_gated_needle`, `sparse_attention`, `attention_contrast`, `hierarchical_gated`, etc.),
- decomposition into explicit reward components for GDPO/DRPO training (`compute_reward_components`),
- csPCa-sensitive weighting/bonus,
- optional:
  - diversity reward (reduce policy collapse),
  - benign sparsity penalties,
  - prostate-boundary penalties.

Notable methodological point:

- for GDPO compatibility in `combined_v2`, component rewards are returned unweighted before component-wise normalization, avoiding weight erasure from normalization.

## 5) New/Extended Loss Functions for Weak and Noisy Supervision

Implemented in `prostnfound/src/loss_new.py`:

- **Top-K MIL loss** (`topk_mil`, `topk_mil_soft`):
  - supervises most suspicious fraction of pixels rather than forcing all needle pixels positive in cancer cores.
- **Thresholded involvement loss** (`thresholded_involvement`):
  - aligns fraction of activated pixels (above threshold) with target involvement.
- **Symmetric Cross-Entropy loss** (`symmetric_ce`):
  - combines CE + reverse CE for robustness to noisy weak labels.
- **Decoder prostate-mask constraints**:
  - soft outside-prostate penalty and hard mask constraint wrappers for anatomically consistent outputs.
- standard MIL/involvement losses retained for compatibility and ablations.

These losses directly target a known weak-label issue: involvement labels are core-level proxies, not exact pixel ground truth.

## 6) Data and Metadata Pipeline Enhancements

### 6.1 Loader enhancements

`prostnfound/src/loaders.py` now includes:

- additional metadata features such as `base_apex_encoding` and `mid_lateral_encoding`,
- robust behavior for evaluation-only settings:
  - avoids shuffling logic that fails on empty train split in test/heatmap scenarios.

### 6.2 NCT metadata inspection utility

`medAI/medAI/scripts/inspect_nct_metadata.py` adds:

- metadata/h5 inspection for length-related fields,
- quick diagnostics for discovering usable biopsy/core length signals.

## 7) Evaluation and Reproducibility Engineering

### 7.1 UA zero-shot batch benchmarking

New scripts/config:

- `prostnfound/test_ua_models.py`
- `prostnfound/cfg/test_ua_models.yaml`

Capabilities:

- batch-evaluate multiple checkpoints on UA/Optimum using a single config,
- preflight validation of dataset directory structure,
- model-wise overrides for legacy checkpoints,
- automatic JSON/CSV aggregated summaries.

### 7.2 More reproducible experiment state

Training scripts now save resolved config snapshots into checkpoint directories and W&B artifacts:

- `medAI/medAI/engine/prostnfound/trainer.py`
- `medAI/medAI/scripts/prostnfound/train_prostnfound.py`
- `prostnfound/train_rl.py` (already saving train config to W&B temp artifact path)

This improves traceability from checkpoint -> exact training configuration.

### 7.3 Robust checkpoint compatibility in testing

`prostnfound/test_rl.py` now includes:

- safer parsing of legacy checkpoint arg formats (including OmegaConf `_content` cases),
- fallback model/model_kw resolution when metadata is absent,
- architecture-key sanitation (`policy_arch_version`) across RL v1 / RL v2 / non-RL checkpoints,
- explicit erroring for empty selected splits.

## 8) Suggested Paper Framing (Concise)

You can frame contributions as:

1. **Architecture contribution**: RL-guided attention extension to ProstNFound for weakly supervised ultrasound.
2. **Optimization contribution**: progression from GRPO to GDPO to DRPO with hierarchical domain/cluster-scaled advantages.
3. **Objective contribution**: noise-robust MIL/involvement losses (Top-K MIL, thresholded involvement, symmetric CE).
4. **Clinical/engineering contribution**: reproducible and scalable evaluation pipeline (UA zero-shot batch benchmarking, robust checkpoint handling, config provenance).

## 9) Key Files to Cite in Methods/Appendix

- RL training orchestration: `prostnfound/train_rl.py`
- RL reward design: `prostnfound/src/rl_loss.py`
- DRPO algorithm: `medAI/medAI/modeling/drpo.py`
- GDPO algorithm: `medAI/medAI/modeling/gdpo.py`
- New weak-label losses: `prostnfound/src/loss_new.py`
- Loader/data adaptations: `prostnfound/src/loaders.py`
- Robust test/eval entrypoint: `prostnfound/test_rl.py`
- UA batch benchmarking: `prostnfound/test_ua_models.py`, `prostnfound/cfg/test_ua_models.yaml`
- Metadata utility: `medAI/medAI/scripts/inspect_nct_metadata.py`

## 10) Notes on Current DRPO Domain Construction

Current DRPO integration uses a practical binary domain split (benign vs cancer-like grade labels) derived from batch metadata. This is a useful first implementation for imbalance mitigation; future work can extend domains to richer clinical strata (site/scanner/patient subgroup/risk bucket) for more explicit domain generalization studies.


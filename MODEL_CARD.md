# Model Card: SimLlama M1

Following Mitchell et al., "Model Cards for Model Reporting" (FAT* 2019).

## Model details

- **What it is:** a symbolic (MIDI) music model that generates a section of music related to a given section, the context, at a requested similarity: `sim:high`, `sim:mid` or `sim:low`.
- **Base model:** Moonbeam 839M, a pretrained MIDI foundation model (Apache-2.0), further trained with LoRA adapters. The exact training settings are in `scripts/slurm/train.sh` and `configs/simllama_839M.json`.
- **Conditioning:** a similarity label, and a context vector that summarises the context section.
- **Authors:** Shangxuan Luo, Zixun Guo, Jinjie Shi, Joshua Reiss (Centre for Digital Music, Queen Mary University of London), 2026.
- **Paper:** "How Close to the Theme? Similarity-Label Conditioning for Game Music Generation", submitted to Computer Music Journal; preprint [arXiv:2611.01047](https://arxiv.org/abs/2611.01047).
- **Licence:** Apache-2.0 (see `NOTICE`).

## Intended use

- Research on controllable symbolic music generation, and on how models vary musical material.
- A baseline for similarity-controlled generation of game-music sections.

Not intended for producing music for release without human review: outputs can restate the context closely (see Limitations).

## Training data

Section pairs from 334 Super Mario pieces transcribed by NinSheetMusic (3,304 directed pairs; 2,398 train, 478 validation, 428 test). Sections come from structure annotations made with Gemini 3.1 Pro and validated against two human annotators on 49 pieces. Each pair's label is the tertile of its compound similarity (CS): `sim:high` at 0.78 and above, `sim:low` below 0.60. The scores are not redistributed; the pair list, annotations and bar maps are in the `supermario-structure-dataset` repository.

## Evaluation

- **Data:** 157 test contexts; six outputs per context and label.
- **Measures:** twelve symbolic similarity measures; one main measure each for melody, pitch content (chroma), rhythm and note length, plus CS as a training check.
- **Results:** on average, outputs requested as similar are more similar to their context than those requested as dissimilar, and not to other pieces. All twelve measures favour them; each main measure stays significant after correction for multiple tests. Against an otherwise identical model without a label, only the `sim:high` request makes outputs more similar.
- **Listening test:** 26 listeners rated similarity on 0–100 scales, of whom 20 passed both attention checks; ratings followed melodic overlap with the context most closely of all measures.

The numbers, tables and figures reproduce from this repository: `bash scripts/eval/run_all.sh --analyses` (CPU).

## Limitations

- **`sim:mid` is not a separate level:** the effect is between `sim:high` and the other labels.
- **Restatement:** on melody, the gap between labels comes mainly from more frequent restatement of the context's melody, so `sim:high` outputs can copy parts of the context.
- **Label metric:** CS, the measure that defines the labels, has no melodic component; the model's control of melody follows from the context, not from the label metric.
- **Domain:** trained on piano transcriptions of one game series; other styles, instruments and lengths are untested.
- **Outputs need correction:** generated MIDI goes through `scripts/eval/correct_generated_output.py` (repairs one-tick notes and truncates to the context's length) before scoring.

## Ethical considerations

The training scores are third-party fan transcriptions of commercial game music; the model is released for research, and outputs that restate a context reproduce material from that piece. Listening-test participants gave written consent to the publication of anonymised ratings.

# SimLlama

This repository contains the code for the paper "How Close to the Theme? Similarity-Label Conditioning for Game Music Generation" (Luo, Guo, Shi, Reiss; submitted to Computer Music Journal; preprint [arXiv:2611.01047](https://arxiv.org/abs/2611.01047)). SimLlama fine-tunes the Moonbeam 839M MIDI foundation model on section pairs from Super Mario game music. The model conditions generation on similarity labels (`sim:high`, `sim:mid`, `sim:low`) and a prototype vector extracted from a context section. NinSheetMusic scores are third-party transcriptions and are not redistributed.

Listen to examples: https://shxluo-saxon.github.io/simllama/ (generated clips from the listening test).

## Repository Contents

| Directory / File | Description |
|---|---|
| `configs/` | Model configuration files for 309M and 839M parameter models. |
| `docs/` | The examples page (GitHub Pages), built from the listening test by the release scripts. |
| `data/annotations/gemini_prompt_v1/` | Gemini structural annotations in JSON format. |
| `data/annotations/human/` | Two human annotations of 49 pieces (validation), and `pieces.csv` mapping them to the Gemini annotations. |
| `data/sections/` | Section pair definitions file `pairs.csv`. |
| `listening_test/` | Listening test answer keys and anonymised ratings. |
| `outputs/eval/` | Precomputed evaluation tables and metric CSV files. |
| `scripts/annotation/` | Gemini API prompting scripts and configurations. |
| `scripts/data/` | Data preprocessing and pair extraction scripts. |
| `scripts/eval/` | Evaluation, analysis, and plotting scripts. |
| `scripts/inference/` | Inference scripts for sampling and triplet generation. |
| `scripts/model/` | Model architecture, tokenizer, embeddings, and LoRA modules. |
| `scripts/slurm/` | Slurm batch scripts for training and inference jobs. |
| `scripts/training/` | Dataset loader and model training scripts. |
| `scripts/utils/` | Utility scripts including checkpoint loaders. |
| `requirements.txt` | Python dependency requirements for evaluation and analysis. |
| `NOTICE` | Attribution notice for code derived from Moonbeam and Hugging Face. |
| `MODEL_CARD.md` | What the model is for, its data, evaluation and limitations. |
| `LICENSE` | Apache 2.0 software licence. |

## Installation

The evaluation scripts run on Python 3.14. Install the required Python packages:

```bash
pip install -r requirements.txt
```

Training and generation require GPU support and PyTorch. Install PyTorch separately to match your CUDA version:

```bash
pip install 'torch>=2.0' --index-url https://download.pytorch.org/whl/cu121
```

If you plan to run annotation scripts in `scripts/annotation/`, install `google-genai` and set your Gemini API key in the environment variable `GOOGLE_API_KEY_MSA`.

## Reproduce the Paper on CPU

You can reproduce all numbers, tables, and figures from the released evaluation CSVs, annotations, and anonymised ratings. This reproduction runs on CPU and requires no score downloads or GPU resources.

Run the following command from the repository root:

```bash
bash scripts/eval/run_all.sh --analyses
```

This command runs the analysis scripts against the released score tables in `outputs/eval/` and rewrites their outputs there. Some analyses read the generated MIDI or the section arrays, which are not released; `--analyses` skips them and says so.

| Reproduces on CPU (`--analyses`) | Needs the full pipeline |
|---|---|
| Effect sizes and the label comparison table (`eval_effect_sizes`, `make_label_comparison_table`) | Scoring the generated MIDI (`eval_robustness_table`, `eval_controls`, `eval_ablations`, `eval_copy_rate`, `eval_mismatched_context`, `eval_window_10s`, `eval_listening_metrics`) |
| The `sim:mid` analysis (`eval_mid_label`) and the numbers quoted in the text (`paper_numbers`) | Copying and gap analyses that re-read the sections (`eval_copy_matched`, `eval_train_overlap`, `eval_real_gap_cut`, `eval_gap_without_copies`, `eval_chroma_overlap`, `eval_composite_vs_single`) |
| Ablations by piece and the paired real gap (`eval_ablations_piece`, `eval_real_gap_paired`) | Listening-test models that score the clips from MIDI (`eval_listening_slopes`, `eval_listening_joint`, `eval_listening_round2`) and the per-trial figure (`plot_listening_trials`) |
| Listening-test scoring and tables (`score_listening_ratings`, `make_listening_tables`), the control tables (`make_controls_table`) and the forest plot (`plot_forest`) | |

The released score tables are the output of the scoring step, so every number in the paper can be traced to a file in `outputs/eval/`. The listening-test models that score clips from MIDI are not refitted under `--analyses`; their tables use the coefficients stored in `outputs/eval/`. The model is described in [MODEL_CARD.md](MODEL_CARD.md).

## Full Pipeline (GPU)

The full pipeline reproduces data preparation, model training, generation, and scoring. This path reflects the exact scripts used for the paper. It was not re-run end to end from a fresh download.

Set the following environment variables on your system:
- `DATA_ROOT`: Root folder containing downloaded scores and intermediate data.
- `MUSESCORE`: Path to the MuseScore 4 executable.
- `SCRATCH`: Scratch directory on your high-performance computing cluster.
- `MID_TO_AUDIO`: Path to the MIDI-to-audio renderer used for the listening-test clips (default: `scripts/eval/mid_to_audio.py`).

Follow these numbered steps:

1. **Score Acquisition**: Download the scores listed in `metadata.csv` of the dataset repository, `supermario-structure-dataset`, from https://www.ninsheetmusic.org. Neither repository redistributes these files.
2. **Unrolling**: Apply the bar maps of the dataset repository with `scripts/data/bar_maps.py apply`, so each score has the bar order the paper used.
3. **Data Preparation**: Put the scores, the annotations and a `metadata.xlsx` under `$DATA_ROOT` (default `data/ninsheetmusic/`): `processed/mxl/` for the unrolled scores, `processed/annotation/annotations/` for the annotation JSON files of the dataset repository, and `metadata.xlsx` from the dataset's `metadata.csv` (the scripts read the Excel file; save the CSV as `.xlsx`). Then run `scripts/data/run_pipeline.py`: it converts the scores to MIDI with MuseScore 4 (`mxl_to_mid.py`), builds the conditioning table and the token arrays, and checks them for anomalies.
4. **Sections and Pairs**: Run `scripts/data/extract_sections.py` to cut the annotated sections, then `scripts/data/build_pairs.py` to build `pairs.csv` with the similarity-bucket thresholds 0.60 and 0.78.
5. **Cluster setup**: The Slurm scripts expect this layout on the cluster:

   | Path | Content |
   |---|---|
   | `$SCRATCH/simllama/` | a clone of this repository; jobs are submitted from here |
   | `$SCRATCH/data/sections/` | the section arrays (`.npy`) from step 4, and `pairs.csv` |
   | `$SCRATCH/models/moonbeam_839M.pt` | the Moonbeam base weights, from https://huggingface.co/guozixunnicolas/moonbeam-midi-foundation-model |
   | `$SCRATCH/models/simllama_m1.pt` | the released M1 checkpoint (see [Checkpoint](#checkpoint)), if you do not train M1 |

   ```bash
   git clone https://github.com/ShxLuo-Saxon/simllama $SCRATCH/simllama
   mkdir -p $SCRATCH/data/sections $SCRATCH/models
   bash $SCRATCH/simllama/scripts/slurm/setup_env.sh     # once, on a login node: conda env moonbeam_train
   cd $SCRATCH/simllama && mkdir -p logs                 # Slurm writes job logs to logs/
   ```

   The scripts were written for one cluster. Before submitting, edit in each `scripts/slurm/*.sh`: the `#SBATCH --account` and `--partition` lines and the GPU resources, and the conda path (`$HOME/miniforge3`) if your conda is elsewhere.
6. **Training**: `sbatch scripts/slurm/train.sh` trains M1 into `$SCRATCH/runs/m1/` (`best.pt` is the checkpoint with the lowest validation loss). `CMODE=m2 sbatch scripts/slurm/train.sh` and `CMODE=m3 ...` train the controls without a label and with a wrong label into `runs/m2/` and `runs/m3/`. `train_controls.sh` retrains a control and samples it several times in one job: `CMODE=m2 sbatch scripts/slurm/train_controls.sh` (no label), `CMODE=m3 ...` (wrong label), `CMODE=m4 ...` (context vector, no label); each writes `runs/<model>_rerun/` and `outputs/<model>_rerun/`.
7. **Generation**: `sbatch scripts/slurm/infer_triplet.sh` generates the matched triplets of M1, M2 and M3 into `$SCRATCH/outputs/m1_triplet_raw/`. It reads `runs/m1/`, `runs/m2/` and `runs/m3/`; set `CKPT_M1`, `CKPT_M2` or `CKPT_M3` to use other files, e.g. `CKPT_M1=$SCRATCH/models/simllama_m1.pt sbatch scripts/slurm/infer_triplet.sh`. `sbatch scripts/slurm/infer_resample.sh` draws five more outputs per context and label from M1 into `outputs/m1_resample/`; it reads `$SCRATCH/models/simllama_m1.pt` unless `CKPT_M1` is set.
8. **Scoring**: Copy each output folder from `$SCRATCH/outputs/` to `outputs/generated/` in your local clone. Correct each with `scripts/eval/correct_generated_output.py`, which writes a `*_corrected` copy, then run `bash scripts/eval/run_all.sh` without `--analyses` to score the corrected outputs on the twelve measures.

A failed step fails its Slurm job: the scripts stop at the first error.

## Checkpoint

The trained M1 checkpoint is released separately under the Apache-2.0 licence. Download `best.pt` from https://huggingface.co/shangxuan-saxon/SimLlama (sha256 `e7542e1c1923cef4b6e1073d54ad15f10656689add328f9c6a0bccb9057834a8`) and save it as `$SCRATCH/models/simllama_m1.pt`. It holds `model_state_dict` with the weights for `configs/simllama_839M.json`, and the training `epoch`, losses and `condition_mode`. M1 alone generates with `infer_resample.sh`; `infer_triplet.sh` also needs the controls M2 and M3, which are not released and are trained in step 6.

## Annotation Prompt

The repository includes the Gemini annotation pipeline under `scripts/annotation/`. The prompt template is located at `scripts/annotation/prompt_v1.md`.

Annotations were created using Gemini 3.1 Pro with high thinking level and default sampling parameters. The API client script is `scripts/annotation/single_api.py`.

## Licence

The code and model weights are licensed under the Apache License, Version 2.0. In `docs/audio/`, the generated clips come from the dataset repository and are licensed under CC BY 4.0; the reference clips (`*_ref.mp3`) are short excerpts of NinSheetMusic transcriptions, shown for illustration only and not covered by either licence.

Parts of `scripts/model/` derive from the Moonbeam foundation model and Hugging Face Transformers. See the `NOTICE` file for detailed attributions.

## Citation

Version 1.0 of this repository is archived on Zenodo: [doi:10.5281/zenodo.23915905](https://doi.org/10.5281/zenodo.23915905) (all versions: [10.5281/zenodo.23915904](https://doi.org/10.5281/zenodo.23915904)). Please cite the paper:

```bibtex
@misc{luo2026simllama,
  title={How Close to the Theme? Similarity-Label Conditioning for Game Music Generation},
  author={Luo, Shangxuan and Guo, Zixun and Shi, Jinjie and Reiss, Joshua},
  year={2026},
  eprint={2611.01047},
  archivePrefix={arXiv},
  note={Submitted to Computer Music Journal}
}
```

## Contact

Shangxuan Luo, Queen Mary University of London: s.luo@qmul.ac.uk. For bugs and questions about the code, please open an issue in this repository.

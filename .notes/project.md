# Project: Mozilla Spontaneous Speech ASR Challenge

## Timeline
- **Week 1 (Days 1-7):** Setup, Baseline Replication, Data Sourcing & Augmentation.
- **Week 2 (Days 8-14):** Main Fine-Tuning Experiments for all Tasks.
- **Week 3 (Days 15-18):** Final Model Selection, Test Set Inference, and Report Writing.

## Folder Structure

/project/
|-- /data/
|   |-- /mozilla_speech_data/  # Provided competition data
|   |   |-- /train/
|   |   |-- /dev/
|   |-- /external_data/        # Any other data you find
|-- /scripts/
|   |-- 01_setup_env.sh
|   |-- 02_prepare_data.py
|   |-- 03_run_baseline.py     # MMS fine-tuning
|   |-- 04_run_whisper_large.py  # For Tasks 1, 2, 4
|   |-- 05_run_whisper_small.py  # For Task 3
|-- /notebooks/
|   |-- Data_Exploration.ipynb
|   |-- Baseline_Evaluation.ipynb
|   |-- Results_Analysis.ipynb
|-- /models/
|   |-- /mms_baseline/
|   |-- /whisper_large_multilingual/
|   |-- /whisper_small_multilingual/
|-- /submissions/
|   |-- /task1_submission/
|   |-- /task2_submission/
|   |-- etc...
|-- /.notes/
|   |-- project.md  # This file
|-- report.pdf  # Your final system description paper

---

## Step-by-Step Plan

### Phase 1: Setup & Data (Days 1-3)

1.  **[ ] Environment Setup (`scripts/01_setup_env.sh`):**
    *   Install PyTorch, Hugging Face Transformers, Datasets, Accelerate, `jiwer`, `libsndfile`, `ffmpeg`, `bitsandbytes`.
    *   Log in to Hugging Face Hub: `huggingface-cli login`.

2.  **[ ] Download Data:**
    *   Get the 21 language datasets from the Mozilla Data Collective link.
    *   Organize them into `/data/mozilla_speech_data/`.

3.  **[ ] Data Preparation (`scripts/02_prepare_data.py`):**
    *   Write a script to load all 21 language datasets using Hugging Face `datasets`.
    *   Create and save a combined `train` and `dev` dataset for multilingual training.
    *   Ensure all audio is resampled to 16kHz, as required by Whisper and MMS.
    *   Normalize the transcription text (lowercase, remove punctuation) to create a consistent vocabulary.

4.  **[ ] Data Sourcing (Parallel Task, Days 2-7):**
    *   **ACTION:** Dedicate several hours to searching for openly-licensed audio data for the 21 languages AND the 5 unseen languages (or related languages).
    *   Track findings, licenses, and URLs in a spreadsheet.

### Phase 2: Baseline & Initial Models (Days 3-7)

1.  **[ ] Replicate Baseline (`scripts/03_run_baseline.py`):**
    *   Adapt a Hugging Face MMS adapter-tuning script.
    *   Train on a single language (e.g., `aln` - Gheg Albanian) and verify your dev WER is close to the `0.548` in the PDF.
    *   This validates your entire pipeline.

2.  **[ ] First Whisper Fine-tune (`scripts/05_run_whisper_small.py`):**
    *   Start fine-tuning `openai/whisper-small` on the combined 21-language dataset. This is for **Task 3**.
    *   Use a Colab Pro high-RAM instance.
    *   Incorporate SpecAugment directly in the data processing function.
    *   This will be your first real model and will likely already beat the baseline on many languages.

### Phase 3: Advanced Models & Winning (Days 8-14)

1.  **[ ] Full-Scale Large Model Training (`scripts/04_run_whisper_large.py`):**
    *   Fine-tune `openai/whisper-large-v3` on the combined dataset (Mozilla data + any external data you found). This is your primary model for **Tasks 1, 2, and 4**.
    *   This will take significant time. Use the `Accelerate` library for efficient multi-GPU/TPU training if available, or just let it run for a day or two on your Colab Pro GPU.
    *   Push intermediate checkpoints to the Hugging Face Hub so you don't lose progress.

2.  **[ ] Iterate and Analyze (`notebooks/Results_Analysis.ipynb`):**
    *   As models finish, evaluate their WER on the dev set for *every* language.
    *   Identify which languages are performing poorly. Maybe they need more specific data or augmentation.
    *   Decide on a single language to target for **Task 2** (Best Improvement). A good candidate is one where the baseline is very high (e.g., `tob` or `top`), as you have more room for improvement.

### Phase 4: Submission (Days 15-18)

1.  **[ ] Test Data Release (Dec 1st):**
    *   The test audio (unlabeled) is released. Download it immediately.

2.  **[ ] Final Inference (Deadline is Dec 8th):**
    *   Load your best `whisper-large-v3` model and transcribe the test sets for Tasks 1, 2, and 4.
    *   Load your best `whisper-small` model and transcribe the test set for Task 3.
    *   Format the output into the required `.tsv` files and zip them according to the submission guidelines.

3.  **[ ] Write System Description Paper (`report.pdf`):**
    *   **DO NOT leave this to the last minute.** Start writing as soon as you have your first baseline results.
    *   Use the provided ACL template.
    *   Describe your approach: foundation model used (Whisper), data preparation, augmentation techniques, external data sources (be transparent!), and fine-tuning parameters.
    *   Include a table comparing your dev set results against the official baseline. This shows the effectiveness of your method.
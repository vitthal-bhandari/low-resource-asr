# Project: ASR for Endangered Languages
## Course Term Project (6 Weeks)

### Project Overview
This term project explores automatic speech recognition (ASR) for low-resource and endangered languages using Mozilla Common Voice's spontaneous speech datasets. The project evaluates baseline and improved ASR architectures across 21 underrepresented languages from Africa, Asia, Europe, and the Americas.

### Research Objectives
1. **Establish Baselines**: Evaluate baseline ASR models (MMS, Whisper) on all 21 languages
2. **Literature Review**: Research state-of-the-art ASR techniques for low-resource languages
3. **Architecture Improvement**: Implement and evaluate improved architectures on 21 languages
4. **Linguistic Error Analysis**: Conduct detailed error analysis on 2 languages with available linguistic resources

---

## Languages

| Region | Language | ISO 639 |
|--------|----------|---------|
| _Africa_ | Bukusu | bxk |
| | Chiga | cgg |
| | Nubi | kcn |
| | Konzo | koo |
| | Lendu | led |
| | Kenyi | lke |
| | Thur | lth |
| | Ruuli | ruc |
| | Amba | rwm |
| | Rutoro | ttj |
| | Kuku | ukv |
| _Americas_ | Wixárika | hch |
| | Southwestern Tlaxiaco Mixtec | meh |
| | Michoacán Mazahua | mmc |
| | Papantla Totonac | top |
| | Toba Qom | tob |
| _Europe_ | Gheg Albanian | aln |
| | Cypriot Greek | el-CY |
| | Scots | sco |
| _Asia_ | Betawi | bew |
| | Western Penan | pne |

---

## MVP Roadmap

Each MVP corresponds to a feature branch (`mvp1`, `mvp2`, etc.) and represents a concrete, testable milestone.

---

### MVP1: Foundation + Data Pipeline + MMS Baseline
**Branch**: `mvp1`
**Timeline**: Week 1-2
**Goal**: Complete infrastructure, data pipeline, and first baseline (MMS)

#### Phase 1.1: Project Infrastructure
- [ ] Set up uv environment with all dependencies
- [ ] Create project folder structure (`src/`, `scripts/`, `notebooks/`, `tests/`)
- [ ] Configure linting (ruff) and formatting
- [ ] Create configuration management (config files for paths, hyperparams)
- [ ] Set up logging infrastructure
- [ ] Set up pytest infrastructure

#### Phase 1.2: Data Pipeline
- [ ] Download all 21 language datasets from Mozilla Data Collective
- [ ] Implement dataset downloader script with progress tracking
- [ ] Create unified data loader for all languages
- [ ] Implement preprocessing: 16kHz resampling, audio normalization
- [ ] Implement text normalization (lowercase, punctuation handling)
- [ ] Create train/dev split utilities
- [ ] Write data exploration notebook (`notebooks/01_data_exploration.ipynb`)

#### Phase 1.3: MMS Baseline
- [ ] Implement MMS inference pipeline using `facebook/mms-1b-all`
- [ ] Create evaluation script with WER/CER calculation
- [ ] Run zero-shot evaluation on all 21 languages
- [ ] Implement results logging and saving (JSON/CSV)
- [ ] Create baseline results visualization

#### Phase 1.4: Unit Tests
- [ ] Tests for config loading and validation
- [ ] Tests for data loading (single language, batch loading)
- [ ] Tests for audio preprocessing (resampling, normalization)
- [ ] Tests for text normalization
- [ ] Tests for MMS inference pipeline
- [ ] Tests for WER/CER metric calculation

**Definition of Done**:
- `uv sync` runs successfully; all imports work
- Can load any of the 21 languages with a single function call
- Audio correctly resampled to 16kHz; text normalized consistently
- MMS WER scores recorded for all 21 languages
- Results saved to `results/baseline/mms_results.json`
- All unit tests pass (`uv run pytest tests/`)

---

### MVP2: Whisper Baseline
**Branch**: `mvp2`
**Timeline**: Week 2
**Goal**: Whisper models evaluated on all 21 languages; complete baseline comparison

#### Phase 2.1: Whisper Implementation
- [ ] Implement Whisper inference pipeline (`whisper-small`, `whisper-large-v3`)
- [ ] Handle language code mapping (Whisper language tokens vs ISO codes)
- [ ] Implement batch inference for efficiency
- [ ] Run zero-shot evaluation on all 21 languages for both model sizes

#### Phase 2.2: Baseline Consolidation
- [ ] Compare Whisper vs MMS baseline results
- [ ] Create consolidated baseline results table (all models × all languages)
- [ ] Generate baseline comparison visualizations
- [ ] Write baseline evaluation notebook (`notebooks/02_baseline_evaluation.ipynb`)

#### Phase 2.3: Unit Tests
- [ ] Tests for Whisper inference pipeline
- [ ] Tests for language code mapping
- [ ] Tests for batch inference
- [ ] Tests for results aggregation and comparison

**Definition of Done**:
- WER scores for `whisper-small` and `whisper-large-v3` on all 21 languages
- Results saved to `results/baseline/whisper_results.json`
- Comparison table: MMS vs Whisper-small vs Whisper-large
- Baseline analysis notebook complete
- All unit tests pass

**Depends on**: MVP1

---

### MVP3: Literature Review & Technique Selection
**Branch**: `mvp3`
**Timeline**: Week 3
**Goal**: Complete literature review; select improvement techniques

#### Phase 3.1: Literature Review
- [ ] Review papers on low-resource ASR (minimum 10-15 papers)
- [ ] Research adaptation methods: adapter tuning, LoRA, prompt tuning
- [ ] Research data augmentation: SpecAugment, speed perturbation, noise injection
- [ ] Research cross-lingual transfer strategies
- [ ] Create annotated bibliography (`literature/annotated_bibliography.md`)

#### Phase 3.2: Technique Selection & Planning
- [ ] Analyze baseline results to identify improvement opportunities
- [ ] Select 2-3 techniques to implement based on feasibility and potential
- [ ] Identify 2 languages for linguistic error analysis (based on resource availability)
- [ ] Document technique selection with justification

#### Phase 3.3: Report Drafting
- [ ] Draft literature review section of report
- [ ] Outline methodology section

**Definition of Done**:
- Annotated bibliography with 10+ papers
- Technique selection documented with justification
- 2 error analysis languages selected with rationale
- Literature review draft complete

**Depends on**: MVP2

---

### MVP4: Fine-tuning Pipeline
**Branch**: `mvp4`
**Timeline**: Week 4
**Goal**: Working fine-tuning pipeline with selected techniques

#### Phase 4.1: Training Infrastructure
- [ ] Implement fine-tuning script for Whisper (using HF Trainer)
- [ ] Create training configuration system (hyperparameters, checkpointing)
- [ ] Implement early stopping and best model selection
- [ ] Set up experiment tracking (W&B or tensorboard)

#### Phase 4.2: Improvement Techniques
- [ ] Implement LoRA/adapter integration (using PEFT library)
- [ ] Implement SpecAugment data augmentation
- [ ] Implement any additional selected techniques

#### Phase 4.3: Pilot Experiments
- [ ] Test fine-tuning on 1-2 pilot languages
- [ ] Validate WER improvement over baseline
- [ ] Tune hyperparameters based on pilot results

#### Phase 4.4: Unit Tests
- [ ] Tests for training configuration loading
- [ ] Tests for LoRA/adapter setup
- [ ] Tests for SpecAugment augmentation
- [ ] Tests for checkpoint saving/loading
- [ ] Tests for early stopping logic

**Definition of Done**:
- Fine-tuning pipeline runs end-to-end
- Can fine-tune with LoRA adapters
- SpecAugment augmentation working
- Pilot language shows WER improvement over baseline
- All unit tests pass

**Depends on**: MVP3

---

### MVP5: Full Fine-tuning Experiments
**Branch**: `mvp5`
**Timeline**: Week 4-5
**Goal**: Fine-tuned models for all 21 languages

#### Phase 5.1: Individual Language Fine-tuning
- [ ] Fine-tune Whisper-small on each language individually (21 runs)
- [ ] Track training metrics (loss curves, validation WER)
- [ ] Save best checkpoints for each language

#### Phase 5.2: Multilingual Training
- [ ] Fine-tune multilingual model on combined dataset
- [ ] Experiment with language sampling strategies
- [ ] Compare multilingual vs individual fine-tuning

#### Phase 5.3: Documentation
- [ ] Document hyperparameters and training details for each run
- [ ] Log all experiment configurations
- [ ] Create training summary report

**Definition of Done**:
- Fine-tuned models saved for all 21 languages
- Multilingual model trained and saved
- Training logs and metrics recorded
- `results/finetuned/` populated with all results

**Depends on**: MVP4

---

### MVP6: Comprehensive Evaluation
**Branch**: `mvp6`
**Timeline**: Week 5
**Goal**: Complete evaluation comparing all models

#### Phase 6.1: Model Evaluation
- [ ] Evaluate all fine-tuned models on dev sets
- [ ] Calculate WER/CER improvement over baselines for each language
- [ ] Statistical significance testing (bootstrap or paired tests)

#### Phase 6.2: Analysis
- [ ] Identify best-performing and worst-performing languages
- [ ] Analyze correlation between dataset size and performance
- [ ] Compare individual vs multilingual fine-tuning results

#### Phase 6.3: Visualization
- [ ] Create results visualizations (bar charts, heatmaps)
- [ ] Generate comparison tables
- [ ] Write results analysis notebook (`notebooks/03_results_analysis.ipynb`)

**Definition of Done**:
- Complete results table: baseline vs fine-tuned for all 21 languages
- Visualizations saved to `results/figures/`
- Key findings documented
- Statistical analysis complete

**Depends on**: MVP5

---

### MVP7: Linguistic Error Analysis
**Branch**: `mvp7`
**Timeline**: Week 5
**Goal**: Detailed error analysis on 2 selected languages

#### Phase 7.1: Resource Gathering
- [ ] Gather linguistic resources (phoneme inventory, grammar sketches)
- [ ] Compile word frequency lists (if available)
- [ ] Document phonological/morphological features of target languages

#### Phase 7.2: Error Analysis Implementation
- [ ] Implement error categorization script (substitution/deletion/insertion)
- [ ] Implement alignment-based error extraction
- [ ] Create error pattern aggregation utilities

#### Phase 7.3: Analysis
- [ ] Analyze error patterns by:
  - Phonological features (consonants vs vowels, specific phonemes)
  - Word position (initial, medial, final)
  - Word frequency (common vs rare words)
  - Morphological complexity
- [ ] Compare error patterns between baseline and fine-tuned models
- [ ] Identify language-specific challenges

#### Phase 7.4: Visualization & Reporting
- [ ] Create error analysis visualizations
- [ ] Write error analysis notebook (`notebooks/04_error_analysis.ipynb`)
- [ ] Write linguistic analysis section of report

#### Phase 7.5: Unit Tests
- [ ] Tests for error categorization logic
- [ ] Tests for alignment-based extraction
- [ ] Tests for pattern aggregation

**Definition of Done**:
- Error analysis complete for 2 languages
- Categorized error tables generated
- Visualizations showing error distributions
- Written analysis with linguistic insights
- All unit tests pass

**Depends on**: MVP6

---

### MVP8: Final Report & Presentation
**Branch**: `mvp8`
**Timeline**: Week 6
**Goal**: Complete, polished deliverables

#### Phase 8.1: Report Writing
- [ ] Write/finalize methodology section (data, models, training procedure)
- [ ] Write/finalize results section with tables and figures
- [ ] Write discussion (findings, limitations, future work)
- [ ] Write abstract and conclusion
- [ ] Compile references

#### Phase 8.2: Code Quality
- [ ] Code cleanup: docstrings, type hints
- [ ] README updates with full reproduction instructions
- [ ] Create reproducibility documentation
- [ ] Ensure all tests pass

#### Phase 8.3: Final Deliverables
- [ ] Final proofreading and formatting of report
- [ ] Prepare presentation slides (if required)
- [ ] Package code for submission

**Definition of Done**:
- Complete report (PDF) ready for submission
- All code documented and runnable
- README provides clear reproduction instructions
- All tests pass
- Presentation ready (if applicable)

**Depends on**: MVP7

---

## MVP Summary Table

| MVP | Branch | Week | Goal | Key Deliverable |
|-----|--------|------|------|-----------------|
| MVP1 | `mvp1` | 1-2 | Infrastructure + Data + MMS | MMS WER scores (21 langs) + tests |
| MVP2 | `mvp2` | 2 | Whisper Baseline | Complete baseline comparison |
| MVP3 | `mvp3` | 3 | Literature Review | Annotated bibliography + technique selection |
| MVP4 | `mvp4` | 4 | Fine-tuning Pipeline | Working training pipeline + tests |
| MVP5 | `mvp5` | 4-5 | Full Experiments | Fine-tuned models (21 langs) |
| MVP6 | `mvp6` | 5 | Evaluation | Complete results comparison |
| MVP7 | `mvp7` | 5 | Error Analysis | Linguistic analysis (2 langs) + tests |
| MVP8 | `mvp8` | 6 | Final Report | Submitted deliverables |

---

## Folder Structure

```
low-resource-asr/
├── data/
│   ├── mozilla_speech_data/       # Mozilla Common Voice spontaneous speech
│   │   ├── raw/                   # Original downloaded data
│   │   └── processed/             # Preprocessed data
│   └── linguistic_resources/      # Grammars, phoneme inventories
├── src/
│   ├── data/                      # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py             # Dataset classes
│   │   ├── preprocessing.py       # Audio/text preprocessing
│   │   └── augmentation.py        # Data augmentation (SpecAugment)
│   ├── models/                    # Model definitions
│   │   ├── __init__.py
│   │   ├── mms.py                 # MMS model wrapper
│   │   └── whisper.py             # Whisper model wrapper
│   ├── training/                  # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py             # Training loop
│   │   └── callbacks.py           # Custom callbacks
│   └── evaluation/                # Evaluation utilities
│       ├── __init__.py
│       ├── metrics.py             # WER, CER calculation
│       └── error_analysis.py      # Linguistic error analysis
├── scripts/
│   ├── download_data.py
│   ├── run_baseline.py
│   ├── finetune.py
│   └── evaluate.py
├── tests/                         # Unit tests
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_metrics.py
│   └── test_error_analysis.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_evaluation.ipynb
│   ├── 03_results_analysis.ipynb
│   └── 04_error_analysis.ipynb
├── results/
│   ├── baseline/
│   ├── finetuned/
│   ├── figures/
│   └── error_analysis/
├── literature/
│   ├── annotated_bibliography.md
│   └── papers/
├── configs/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── training_config.yaml
├── .notes/
│   └── project.md                 # This file
├── pyproject.toml
├── README.md
└── report.pdf
```

---

## Deliverables

1. **Baseline Results Table**: WER scores for MMS and Whisper on all 21 languages
2. **Improved Model Results**: WER comparison showing improvement over baselines
3. **Linguistic Error Analysis Report**: Detailed analysis for 2 languages
4. **Literature Review**: Annotated bibliography and synthesis of ASR SOTA
5. **Final Report**: Complete term paper documenting methodology, results, and analysis
6. **Codebase**: Documented scripts with unit tests for reproducibility

---

## Baseline Models

### MMS (Massively Multilingual Speech)
- Model: `facebook/mms-1b-all`
- Approach: Zero-shot evaluation, adapter-based fine-tuning
- Reference: Pratap et al. (2023) "Scaling Speech Technology to 1,000+ Languages"

### Whisper
- Models: `openai/whisper-small` (~244M params), `openai/whisper-large-v3` (~1.5B params)
- Approach: Zero-shot evaluation, full/LoRA fine-tuning
- Reference: Radford et al. (2022) "Robust Speech Recognition via Large-Scale Weak Supervision"

---

## Improvement Techniques (to explore in MVP3)

1. **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning
2. **Adapter Tuning**: Language-specific adapters
3. **SpecAugment**: Time/frequency masking for data augmentation
4. **Speed Perturbation**: Audio speed variation
5. **Cross-lingual Transfer**: Leverage related high-resource languages
6. **Curriculum Learning**: Train on easier examples first

---

## Error Analysis Languages (Candidates)

Select 2 languages based on:
- Availability of grammatical descriptions/phoneme inventories
- Typological diversity (different language families/regions)
- Sufficient dev set size for meaningful analysis

**Candidates**:
- Gheg Albanian (aln) - Indo-European, good documentation
- Scots (sco) - Germanic, well-documented
- Cypriot Greek (el-CY) - Indo-European, Greek resources applicable
- Wixárika (hch) - Uto-Aztecan, growing documentation

---

## Key Metrics

- **Primary**: Word Error Rate (WER)
- **Secondary**: Character Error Rate (CER)
- **Analysis**: Error type distribution (substitution/deletion/insertion rates)

---

## Resources

- [Mozilla Data Collective Datasets](https://community.mozilladatacollective.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT Library (LoRA)](https://huggingface.co/docs/peft)
- [MMS Paper](https://arxiv.org/abs/2305.13516)
- [Whisper Paper](https://arxiv.org/abs/2212.04356)

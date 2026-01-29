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

## Timeline (6 Weeks)

### Week 1: Environment Setup & Data Preparation
- [ ] Set up development environment (PyTorch, Transformers, Datasets, jiwer)
- [ ] Download 21 language datasets from Mozilla Data Collective
- [ ] Organize data into standardized folder structure
- [ ] Create data loading and preprocessing scripts (16kHz resampling, text normalization)
- [ ] Begin literature search for ASR SOTA papers

### Week 2: Baseline Establishment
- [ ] Implement MMS (Massively Multilingual Speech) baseline
- [ ] Implement Whisper baseline (whisper-small and whisper-large-v3)
- [ ] Run baseline evaluation on all 21 languages
- [ ] Document baseline WER scores for each language
- [ ] Continue literature review; create annotated bibliography

### Week 3: Literature Review & Architecture Research
- [ ] Complete literature review on low-resource ASR techniques
- [ ] Research adaptation methods: adapter tuning, LoRA, prompt tuning
- [ ] Research data augmentation: SpecAugment, speed perturbation, noise injection
- [ ] Research cross-lingual transfer and multilingual training strategies
- [ ] Identify 2 languages for linguistic error analysis (based on resource availability)
- [ ] Write literature review section of report

### Week 4: Improved Architecture Implementation
- [ ] Implement selected improvement techniques
- [ ] Fine-tune models with chosen methods (e.g., adapter tuning, data augmentation)
- [ ] Train multilingual models on combined dataset
- [ ] Begin evaluation on dev sets
- [ ] Gather linguistic resources for error analysis languages

### Week 5: Evaluation & Error Analysis
- [ ] Complete evaluation of improved models on all 21 languages
- [ ] Compare improved models against baselines (WER improvement analysis)
- [ ] Conduct linguistic error analysis on 2 selected languages
  - Analyze error types: substitutions, deletions, insertions
  - Categorize errors by phonological/morphological features
  - Identify language-specific challenges
- [ ] Create visualizations and result tables

### Week 6: Report Writing & Finalization
- [ ] Write methodology section
- [ ] Write results and analysis section
- [ ] Write linguistic error analysis section
- [ ] Complete discussion and conclusions
- [ ] Prepare final presentation/slides
- [ ] Code cleanup and documentation
- [ ] Submit final report

---

## Folder Structure

```
/project/
├── /data/
│   ├── /mozilla_speech_data/       # Mozilla Common Voice spontaneous speech
│   │   ├── /train/
│   │   └── /dev/
│   └── /linguistic_resources/      # Grammars, phoneme inventories, etc.
├── /scripts/
│   ├── 01_setup_env.sh
│   ├── 02_prepare_data.py
│   ├── 03_run_mms_baseline.py
│   ├── 04_run_whisper_baseline.py
│   ├── 05_finetune_improved.py
│   └── 06_error_analysis.py
├── /notebooks/
│   ├── Data_Exploration.ipynb
│   ├── Baseline_Evaluation.ipynb
│   ├── Results_Analysis.ipynb
│   └── Linguistic_Error_Analysis.ipynb
├── /models/
│   ├── /mms_baseline/
│   ├── /whisper_baseline/
│   └── /improved_models/
├── /results/
│   ├── /baseline_results/
│   ├── /improved_results/
│   └── /error_analysis/
├── /literature/
│   ├── annotated_bibliography.md
│   └── /papers/
├── /.notes/
│   └── project.md                  # This file
└── report.pdf                      # Final term paper
```

---

## Deliverables

1. **Baseline Results Table**: WER scores for MMS and Whisper on all 21 languages
2. **Improved Model Results**: WER comparison showing improvement over baselines
3. **Linguistic Error Analysis Report**: Detailed analysis for 2 languages
4. **Literature Review**: Annotated bibliography and synthesis of ASR SOTA
5. **Final Report**: Complete term paper documenting methodology, results, and analysis
6. **Codebase**: Documented scripts for reproducibility

---

## Baseline Models to Evaluate

### MMS (Massively Multilingual Speech)
- Model: `facebook/mms-1b-all`
- Approach: Adapter-based fine-tuning for each language
- Reference: Pratap et al. (2023) "Scaling Speech Technology to 1,000+ Languages"

### Whisper
- Models: `openai/whisper-small` (~242 MB), `openai/whisper-large-v3` (~1.5 GB)
- Approach: Multilingual fine-tuning on combined dataset
- Reference: Radford et al. (2022) "Robust Speech Recognition via Large-Scale Weak Supervision"

---

## Potential Improvement Techniques

1. **Adapter Tuning**: Efficient fine-tuning with language-specific adapters
2. **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning
3. **SpecAugment**: Time/frequency masking for data augmentation
4. **Cross-lingual Transfer**: Leverage related high-resource languages
5. **Curriculum Learning**: Train on easier examples first
6. **Self-Training**: Use pseudo-labels from confident predictions

---

## Languages for Linguistic Error Analysis (Candidates)

Select 2 languages based on:
- Availability of grammatical descriptions/phoneme inventories
- Typological diversity (different language families/regions)
- Sufficient dev set size for meaningful analysis

**Potential candidates**:
- Gheg Albanian (aln) - Indo-European, good linguistic documentation
- Scots (sco) - Germanic, well-documented
- Cypriot Greek (el-CY) - Indo-European, related to well-documented Greek
- Wixárika (hch) - Uto-Aztecan, growing linguistic documentation

---

## Key Metrics

- **Primary**: Word Error Rate (WER)
- **Secondary**: Character Error Rate (CER)
- **Analysis**: Error type distribution (substitution/deletion/insertion rates)

---

## Resources

- [Mozilla Data Collective Datasets](https://community.mozilladatacollective.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [MMS Paper](https://arxiv.org/abs/2305.13516)
- [Whisper Paper](https://arxiv.org/abs/2212.04356)

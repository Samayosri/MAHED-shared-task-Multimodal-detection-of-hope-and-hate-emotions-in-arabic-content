# MAHED-shared-task-Multimodal-detection-of-hope-and-hate-emotions-in-arabic-content
# Overview

This project jointly predicts three labels for Arabic tweets:

Emotion — 12-way classification (anger, anticipation, confidence, disgust, fear, joy, love, neutral, optimism, pessimism, sadness, surprise)
Offensive — binary (yes / no)
Hate — binary, defined only when a tweet is Offensive = yes

The training set has 5,960 tweets. Hate is annotated only for the 1,744 offensive tweets, making it the smallest and most imbalanced of the three tasks. All approaches share the same preprocessing pipeline (URL/@mention stripping, repeated-character collapsing, emoji removal, AraBERT/Farasa normalization) and the same stratified train/validation/test split.

Approach Summary

Development proceeded across three phases and eight modeling approaches:

#	Approach	Phase	Avg Macro-F1
1	Frozen AraBERT-twitter + Logistic Regression	1	0.606
2	Frozen MARBERT + Logistic Regression	1	0.643
3	Fully fine-tuned multi-head MARBERT	1	0.638
4	MARBERT + BiLSTM + Attention (Variant A)	2	0.637
5	MARBERT + Task Attention Pooling + Fusion (Variant B)	2	0.627
6	Fixed multi-task MARBERT (corrected label masking)	3	0.676
7	3-seed ensemble with LoRA fine-tuning	3	0.682

Approach 2 established MARBERT as the stronger backbone over AraBERT-twitter. Approach 6 fixed a bug where missing Hate labels were treated as a real class, and added effective-number class weighting, improved attention pooling, a robust masked multi-task loss, and early stopping. Approach 7 (the final submitted system) trains the fixed model with 3 random seeds and averages predictions at inference.

# Key Findings
Embedding choice mattered more than data augmentation. A sweep of under-sampling / back-translation / dialect-translation oversampling all underperformed the unaugmented baseline.
Correct NaN/label masking was the single biggest lever in Phase 3 — fixing how missing Hate labels were handled lifted macro-F1 from ~0.638 to 0.676.
Error analysis surfaced two recurring failure modes: (1) ground-truth annotation contradictions/label noise (e.g. urgent requests labeled neutral instead of anticipation), and (2) sarcasm and implicit hate speech that the model's Hate head misses because it's conditioned on the Offensive head.
Ensembling smoothed over these inconsistent, seed-sensitive errors better than adding architectural complexity — both Phase 2 variants (BiLSTM+attention, task-attention fusion) overfit more without a corresponding macro-F1 gain.
Scoring Reconciliation vs. Official Baseline

The official leaderboard lists a baseline of 0.50 average macro-F1, but running the organizers' own baseline code as-is gave 0.66 — a large mismatch traced to how missing Hate labels are handled during scoring. Re-scoring using only test rows with an actual Hate label brought the baseline to ≈0.55, much closer to the published number. Applying the same rule to our model gave ≈0.65.


# Final Submitted System

3-seed ensemble of the fixed multi-task MARBERT model (Approach 7): a single shared MARBERT encoder, fine-tuned with correctly masked and class-weighted multi-task loss, trained three times with different seeds, with predictions averaged at inference time.

Best average macro-F1 across all eight approaches (0.682), driven mainly by a stronger Hate score
Validated to genuinely outperform the reconciled official baseline (0.65 vs. 0.55)
Outperformed the more architecturally complex Phase 2 variants, which overfit without a corresponding macro-F1 gain
# Tech Stack

PyTorch · HuggingFace Transformers · MARBERT (UBC-NLP/MARBERTv2) · AraBERT · PEFT / LoRA · scikit-learn · Farasa · arabic-reshape

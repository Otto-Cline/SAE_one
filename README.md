///////// STRUCTURE /////////

sae-interpretability/

├── data/
│ ├── activations/ # saved activation tensors
│ └── text_samples/ # raw text you'll use

├── models/
│ ├── checkpoints/ # saved SAE weights
│ └── base_model/ # cache for GPT-2 if needed

├── src/
│ ├── extract_activations.py
│ ├── train_sae.py
│ ├── sae_model.py # SAE architecture
│ ├── analyze_features.py # interpretability analysis
│ └── utils.py # helper functions

├── notebooks/
│ └── exploration.ipynb # for visualization/analysis

├── results/
│ ├── figures/
│ └── feature_analysis/ # saved interpretations

├── requirements.txt
├── README.md
└── config.yaml # experiment w/ hyperparameters w/o changing code

Rationale:

- Keep data extraction separate from training (you'll rerun training many times)
- config.yaml lets you experiment with hyperparams without code changes
- Notebooks for exploratory work, scripts for reproducible runs
- Split architecture from training logic


//////// PLAN //////////

Steps:

1. Pick a small model - GPT-2 small (124M params) or TinyLlama. Use HuggingFace transformers.

2. Extract activations - Run a bunch of text through the model, hook into a specific layer (try residual stream after layer 6), save the activation vectors. Need ~10M+ activation vectors.

3. Build SAE - Simple: Linear encoder (d_model → k*d_model, usually k=4-8x), ReLU, Linear decoder back. That's it.

4. Training - Loss = MSE(reconstructed, original) + lambda * L1(encoded). Tune lambda to get ~5% sparsity (95% of features are zero).

5. Interpret - For each learned feature, find text that maximally activates it. See if it's interpretable.# SAE_one

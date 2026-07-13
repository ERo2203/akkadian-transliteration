# Training Run Results

This folder stores lightweight JSON artifacts copied from local training runs.
Model checkpoint weights (`*.pt`, `*.pth`, `*.ckpt`, etc.) are intentionally not
tracked in Git.

Runs included:

- `unicode_to_translit`
- `unicode_to_translit_10k_cpu`
- `unicode_to_translit_eos_weighted`
- `unicode_to_translit_eos_weighted_30k`
- `unicode_to_translit_full_152k_d256`

Each run includes `history.json` with epoch-level loss and evaluation metrics.
Where available, the folder also includes config, split summary, vocabulary,
test metrics, eval-only metrics, and prediction samples.

The 152k model training implementation is in:

- `scripts/train_unicode_to_translit.py`

Dashboard generation is in:

- `scripts/build_training_dashboard.py`

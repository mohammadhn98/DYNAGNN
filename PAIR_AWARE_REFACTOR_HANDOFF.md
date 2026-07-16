# Pair-aware-only Optuna refactor handoff

## Implemented

- Removed the runtime choice between GAT-CORAL and pair-aware GINE.
- Removed legacy GAT-CORAL training, decoding, checkpoint loading, and exports.
- Consolidated the duplicated Voltage/Spower pair-aware models into one shared
  `PairAwareGINE` implementation.
- Preserved the normal repository flow:
  `main.py` → `src/training.py` → task training modules → model checkpoints.
- Kept separate task entry points:
  `modules/voltage_training.py` and `modules/spower_training.py`.
- Added independent Optuna studies for Voltage and Spower.
- Moved all active model/optimizer hyperparameters into `optuna.hparams`:
  `hidden_dim`, `node_id_dim`, `contingency_id_dim`, `type_dim`, `pair_dim`,
  `num_gnn_layers`, `decoder_hidden_dim`, `dropout`, `lr`, and `weight_decay`.
- Kept loss construction and output decoding parameters fixed under
  `training.pair_aware`.
- Changed deployment checkpoint names to:
  `model/voltage_best_model.pt` and `model/spower_best_model.pt`.
- Kept the external inference output unchanged: one final class per component.

## Deliberate decision

`op_context_embedding_dim` was not added to Optuna because the final model does
not use a separate operating-point context encoder. Tuning a disabled dimension
would have no effect. Operating-point information remains available through
node/edge electrical features and graph mean/max pooling.

## Validation performed

- Compiled every Python file with `py_compile`.
- Constructed the generic model for both bus and generator target masks and
  verified forward-output dimensions.
- Sampled all ten hyperparameters through Optuna and reloaded a generated
  deployment checkpoint.
- Ran a synthetic one-trial end-to-end Optuna flow for both Voltage and Spower,
  including training, validation selection, test evaluation, and deployment
  checkpoint creation.

A full Nordic training run was not performed in the packaging environment.

## Left for repository maintainers

As requested, the general README/module documentation and
`Nordic_test_setup.py` still need to be aligned with the new configuration and
checkpoint names.

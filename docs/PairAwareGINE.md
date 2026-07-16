# Pair-aware six-class GINE

DYNAGNN now uses one model family: the direct pair-aware residual GINE model.
There is no architecture selector or legacy GAT-CORAL fallback.

The model predicts classes 0–5 directly and uses:

- residual edge-aware GINE message passing;
- concatenation of the initial node representation and all GINE-layer outputs;
- target-component and contingency identity embeddings;
- event encoding and explicit target–contingency interactions;
- graph mean/max context;
- a six-class head, class-0 gate, and auxiliary log-KPI regression head.

## Training

Use the normal pipeline:

```bash
python3 main.py --from-step training
```

The standard graph construction, operating-point split, electrical-distance
feature, and training-only feature scaling are retained. Voltage and Spower
are tuned independently with Optuna.

Deployment checkpoints are written to:

```text
<data.path>/model/voltage_best_model.pt
<data.path>/model/spower_best_model.pt
```

## Configuration

Loss construction and decoding settings remain fixed under
`training.pair_aware`. Model capacity, dropout, learning rate, and weight decay
are defined only in `optuna.hparams`.

The separate operating-point context encoder is not used. Operating-condition
information enters through the graph electrical features and graph-level
mean/max pooling.

## Inference

The external interface remains unchanged:

```bash
python3 DYNAGNN.py --case-dir /path/to/operating_point --events-csv /path/to/events.csv
```

The output remains one activity class per target component in
`prediction_voltage.csv` and `prediction_spower.csv`.

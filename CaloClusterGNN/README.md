# CaloClusterGNN

## Introduction

A Graph Neural Network for calorimeter hit clustering, intended as a
drop-in (parallel) replacement for the existing seed+BFS
`CaloClusterMaker` in Mu2e Offline. The deployed recipe is
**CaloClusterNet + BFS-style traversal at ExpandCut = 10 MeV**
("CCN+BFS10"). On the MDC2025 mixed-pileup test set
(276,688 events / 481,543 disk-graphs) it beats BFS on every
downstream-relevant cluster-physics metric:

| Metric (E_reco >= 50 MeV) | BFS   | CCN+BFS10 | Change |
|---------------------------|-------|-----------|--------|
| Mean abs(dE) / MeV        | 0.839 |   0.616   | -27%   |
| 95th-pct abs(dE) / MeV    | 3.520 |   2.338   | -34%   |
| Mean centroid dr / mm     | 1.589 |   1.292   | -19%   |
| 95th-pct dr / mm          | 3.606 |   2.294   | -36%   |

Two model classes are trained from this directory:
* `SimpleEdgeNet`  -- 215 K params, 3 message-passing rounds, sum aggregation.
* `CaloClusterNet` -- 676 K params, 4 EdgeAwareResBlocks with gated
  aggregation + global context, optional node-saliency head.
  This is the production model.

Both use the same input graph (one per calorimeter disk per event,
6 node features + 8 edge features), same z-score normalisation, and
the same pipeline -- so swapping models in deployment is config-only
(see [`docs/onnx_deployment.md`](#deployment) cross-link below).

This README covers, in order:
* where an analyzer can find things they might like to know;
* instructions for retraining (full pipeline, end to end);
* a table of commits for each training version;
* deployment cross-links.

## For the Interested Analyzer

Two short pointers:

* **Per-graph definition** -- one PyG `Data` object per calorimeter
  disk per event. Node features: `log(1+E)`, time, x, y, radial r,
  per-graph relative energy `E/E_max`. Edge features: dx, dy, distance,
  dt, dlog_e, energy asymmetry, log summed energy, dr.
  See `src/data/graph_builder.py`.
* **Truth labelling** -- "calo-entrant ancestor" rule: each `CaloHit`
  is grouped under the highest Geant4 ancestor that also deposited
  energy in the same disk. This recovers true shower membership for
  hits split across crystals during showering. Requires the
  `calomcsim.ancestorSimIds` branch added in
  [Mu2e/EventNtuple #](https://github.com/Mu2e/EventNtuple/pulls).
  See `src/data/truth_labels_primary.py`.

## For the Interested (Re)Trainer

Two stages: (a) train + freeze a model and export it to ONNX;
(b) the C++ inference module that consumes the `.onnx` lives in
`Mu2e/Offline:Offline/CaloCluster/` and is **not** part of MLTrain.
This subdirectory is training only.

### General Setup

Fork this repo and clone:

```bash
cd /path/to/your/work/area/

# only needs to be done once
git clone https://www.github.com/YourGitHubUsername/MLTrain.git
cd MLTrain/
git remote add -f mu2e https://www.github.com/Mu2e/MLTrain.git

# whenever you start a new development cycle
git fetch mu2e main
git checkout --no-track -b your-new-branchname mu2e/main
cd CaloClusterGNN/
```

Activate the Mu2e Python environment (PyTorch 2.5.1, PyG 2.7.0,
uproot 5.7.2 are available in `ana 2.6.1`). The included
`setup_env.sh` wraps `setupmu2e-art.sh` + `pyenv ana 2.6.1` and
extends `PYTHONPATH` so the unit tests find the `src/` package:

```bash
source setup_env.sh        # works in interactive and batch shells
python3 scripts/smoke_test_env.py
python3 -m unittest discover -s tests -p "test_*.py" -v
```

### Training Pipeline

Five steps, end to end. All paths are relative to `CaloClusterGNN/`.

1. **Lock a dataset split.** The split is frozen for v2 at 35/7/8
   (train/val/test) -- if you are extending the dataset, regenerate:

    ```bash
    python3 scripts/make_splits.py        # writes splits/{train,val,test}_files.txt
    ```

2. **Build per-disk graphs from EventNtuple ROOT files.** Reads the
   v2 NTS files (which carry `calomcsim.ancestorSimIds`), applies the
   calo-entrant truth rule, and writes one `.pt` graph per disk per
   event under `data/processed/`. About 10 min for 41,656 graphs on a
   CPU node:

    ```bash
    bash scripts/build_all_graphs.sh
    python3 scripts/pack_graphs.py        # packs into train.pt / val.pt / test.pt
    ```

3. **Train.** Two model families, each via a config file under
   `configs/`. CaloClusterNet (production model):

    ```bash
    python3 scripts/train_gnn.py \
        --config configs/calo_cluster_net.yaml \
        --device cuda --run-name calo_cluster_net_v2_stage1
    ```

    SimpleEdgeNet:

    ```bash
    python3 scripts/train_gnn.py \
        --config configs/default.yaml \
        --device cuda --epochs 100 --batch-size 64 \
        --run-name simple_edge_net_v2
    ```

    Per-run outputs land under `outputs/runs/<run-name>/`.

4. **Tune the edge threshold on val** (model-agnostic):

    ```bash
    python3 scripts/tune_threshold.py \
        --config configs/calo_cluster_net.yaml \
        --checkpoint outputs/runs/calo_cluster_net_v2_stage1/checkpoints/best_model.pt
    ```

    The frozen v2 thresholds are baked into the configs:
    CaloClusterNet `tau_edge=0.20`, SimpleEdgeNet `tau_edge=0.26`.

5. **Evaluate once on test, run failure audits + cluster-physics
   evaluation:**

    ```bash
    OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 python3 -u scripts/evaluate_test.py
    OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 python3 -u scripts/evaluate_cluster_physics.py
    python3 scripts/failure_audit.py
    ```

### Frozen Recipe Values

These match the deployment defaults (see Deployment below). If you
change them you'll want to re-tune and re-evaluate:

| Hyperparameter   | Value | Where it's set                     |
|------------------|-------|------------------------------------|
| `r_max`          | 210 mm| `configs/*.yaml -> graph.r_max`    |
| `dt_max`         | 25 ns | `configs/*.yaml -> graph.dt_max`   |
| `k_min`          | 3     | `configs/*.yaml -> graph.k_min`    |
| `k_max`          | 20    | `configs/*.yaml -> graph.k_max`    |
| `tau_edge` (CCN) | 0.20  | `configs/calo_cluster_net.yaml`    |
| `tau_edge` (SEN) | 0.26  | `configs/default.yaml`             |
| `bfs_expand_cut` | 10 MeV| `configs/calo_cluster_net.yaml`    |
| `min_hits`       | 2     | `configs/calo_cluster_net.yaml`    |
| `min_energy_mev` | 10.0  | `configs/calo_cluster_net.yaml`    |

## Versions and Provenance

| Version                       | Commit | EventNtuple dataset                                                              |
|-------------------------------|--------|----------------------------------------------------------------------------------|
| `calo-cluster-net-v2-stage1`  | TBD    | `FlateMinusMix1BBTriggered/MDC2025-002` (50 files, MDC2025af, with ancestorSimIds)|
| `simple-edge-net-v2`          | TBD    | same                                                                              |

The EventNtuples for v2 require the `calomcsim.ancestorSimIds`
branch added in
[Mu2e/EventNtuple#TBD](https://github.com/Mu2e/EventNtuple/pulls).

## Deployment

The trained models ship to Mu2e/Offline as ONNX artifacts; the C++
inference modules live in `Offline/CaloCluster/`. See:

* `Mu2e/Offline:Offline/CaloCluster/src/CaloHitGraphMaker_module.cc`
  (per-disk graph construction, port of `src/data/graph_builder.py`)
* `Mu2e/Offline:Offline/CaloCluster/src/CaloClusterMakerGNN_module.cc`
  (ONNX inference + cluster assembly, model-agnostic; one C++ class
  swaps SimpleEdgeNet vs CaloClusterNet via FHiCL)

Trained `.onnx` artifacts live in the Mu2e data area and are picked
up by `art::ConfigFileLookupPolicy` at job start. The deployment-side
parity gate (Python pipeline vs C++ Offline pipeline, byte-exact on
cluster labels) lives in the Mu2e/Offline PR for this work.

## License

This subdirectory inherits the MLTrain repository LICENSE.

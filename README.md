# SIMQ-Implement

This repository contains the CPU implementation used for the SIMQ main
experiments. SIMQ is a multi-vector retrieval index for ColBERT-style token
embeddings. It partitions document token vectors into data-aware clusters,
builds a graph over cluster representatives, retrieves candidate clusters for
each query token, and applies exact MaxSim reranking to the candidate
documents.

This repository is the implementation companion to the SIMQ paper and the
local reproduction experiments under `/home/yifan/multivector_experiment_p3`.
The paths below are examples for the current experiment machine; replace them
with paths appropriate to another machine.

## Repository layout

```text
cpp/
  simq_index_builder.cpp   Cluster construction and representative index build
  simq_query.cpp           Query, candidate scoring, reranking, and evaluation
  npy_reader.h             Minimal NumPy-array reader
  parallel_documents.h     OpenMP document-level parallel loop
  progress_reporter.h      Progress reporting helper
hnswlib/                   Vendored HNSW implementation and documentation
```

The repository does not include datasets, token embeddings, qrels, or large
index artifacts. Those files must be prepared separately.

## Requirements

- Linux with a C++17 compiler
- CMake 3.14 or newer
- OpenMP for parallel index construction
- POSIX threads
- `libaio` for the asynchronous raw-vector reranking path

The implementation expects ColBERT-style 64-dimensional token embeddings in
NumPy format, together with metadata describing the document-vector range.
The exact data preparation used in the paper experiments is maintained in the
local experiment workspace rather than downloaded by this repository.

## Build

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces:

```text
build/simq_index_builder
build/simq_query
```

The release flags enable `-O3 -march=native -DNDEBUG`; compile on the target
machine when using these flags.

## Expected embedding layout

For an embedding directory `<EMB_DIR>`, the builder expects:

```text
<EMB_DIR>/colbert_embeddings.npy
<EMB_DIR>/colbert_embeddings_meta.json
```

The metadata JSON must contain `doc_vector_start` and `doc_vector_end`. The
builder uses the half-open range `[doc_vector_start, doc_vector_end)` as the
document token vectors and ignores query vectors during index construction.

A completed clustering directory contains the files needed by the query
program, including:

```text
cluster_members.json
cluster_centroids.npy
cluster_representatives.txt
centroid_mapping.json
clustering_meta.json
centroid_hnsw_index.hnsw
```

The query path also expects the experiment-specific query and reorder files,
including the query embeddings, document/token metadata, and the raw-vector
reorder directory. Keep all IDs and row orders consistent across these files.

## Building a SIMQ index

The builder defaults are intended for a small local run. For the paper-style
multi-threaded construction, explicitly set all important parameters:

```bash
./build/simq_index_builder \
  --embeddings-dir /path/to/colbert_64d \
  --output-dir /path/to/colbert_64d/SIMQ_index \
  --init-ratio 0.1 \
  --max-size 160 \
  --split-idx 80 \
  --build-threads 64 \
  --batch-size 10000 \
  --seed 42 \
  --deferred-kmeans-split
```

The relevant construction choices are:

- `init-ratio` (`alpha`): sampling ratio used for the initial representatives;
- `max-size`: maximum target cluster size before splitting;
- `split-idx`: split threshold used by the clustering implementation;
- `build-threads`: OpenMP construction workers;
- `batch-size`: assignment batch size;
- `deferred-kmeans-split`: assigns vectors against a fixed initial set and
  recursively splits oversized clusters with local `K=2` k-means after the
  assignment phase.

The deferred strategy is the multi-threaded construction mode used in the
main experiments. The alternative online split-insertion behavior remains the
default when `--deferred-kmeans-split` is omitted. The builder also constructs
and saves the representative HNSW index unless
`--skip-representative-hnsw` is supplied.

Use `./build/simq_index_builder --help` for the complete option list.

## Querying and evaluation

The query executable can run interactively or in a non-interactive evaluation
mode. A representative main-experiment invocation is:

```bash
./build/simq_query \
  --embeddings-dir /path/to/colbert_64d \
  --cluster-dir /path/to/colbert_64d/SIMQ_index \
  --reorder-dir /path/to/colbert_64d/reorder_embedding \
  --qrels-file /path/to/search_results_top100.tsv \
  --m 100 \
  --ef 100 \
  --coarse-k 1000 \
  --final-k 100 \
  --query-threads 32
```

Here `m` is the number of representative clusters searched per query token.
`ef` is the graph-search budget. `coarse-k=1000` retains the top 1,000
document candidates for exact MaxSim reranking, and `final-k=100` reports the
top 100 documents after reranking. Thus, in the paper experiment label
`rerank=1000`, the 1,000 candidates are reranked and truncated to 100 final
results; it does not mean that only 100 candidates are reranked.

Useful interactive commands are:

```text
info
set-ef 100
set-m 100
eval
eval both
eval_lazy
eval_lazy_LoTTE
retrieve 0 1
quit
```

The evaluation prints throughput and precision/recall statistics, including
`Recall@10@100`, `Recall@20@100`, `Recall@50@100`, and the corresponding
`Recall@100@100` metric used for the main QPS--recall plots. The evaluator can
use HNSW or the optional center-index path; select the latter explicitly with
`--center-index cagra` only when the required CAGRA files exist.

## Paper experiment defaults

The main experiments use five datasets: Quora, SciDocs, NQ, HotpotQA, and
LoTTE-pooled. All use 64-dimensional token vectors and the same evaluation
qrels. The current paper workspace uses the following broad protocol:

| Component | Main setting |
| --- | --- |
| Initial sampling ratio | `alpha=0.1` |
| Cluster size/splitting | `max-size=160`, split around `80`, deferred `K=2` |
| Representative graph | HNSW over cluster representatives |
| Query threads | 32 in the recorded main sweeps |
| Graph search budget | `ef=100` in the standard SIMQ sweeps |
| Candidate documents | 1,000 |
| Final output | top 100 documents |
| Reported metric | `Recall@100@100` and QPS |

The exact dataset-specific cluster counts and index paths should be read from
the build logs and `clustering_meta.json`, rather than inferred from a generic
rule. Build time includes clustering, cluster splitting, representative
selection, and representative-graph construction; query time is measured only
after the index has been loaded.

## Reproducibility notes

1. Record the repository commit, compiler, CPU thread count, dataset revision,
   embedding row order, and qrels path.
2. Build the index once and reuse the same saved index for all `m` sweeps.
3. Keep `coarse-k` and `final-k` fixed when comparing only the `m` parameter.
4. Separate index-build time from query time; loading or rebuilding an index
   must not be included in steady-state QPS.
5. Report whether raw-vector reranking uses the asynchronous I/O path or an
   in-memory/reused representation, because the two modes measure different
   system behavior.
6. Preserve the complete stdout log for every build and query sweep.

## End-to-end reproduction commands

The following commands describe the experiment workflow used for the paper
workload. The standalone binaries in this repository are useful for building
and evaluating one index. The complete five-dataset sweep is driven by the
companion scripts in `/home/yifan/multivector_experiment_p3`; those scripts
also validate the dataset paths and append the full stdout to a timestamped
log.

### 1. Build the experiment evaluator

```bash
cd /home/yifan/multivector_experiment_p3
export CPP_DIR="$PWD/vector_test/cpp"
export BUILD_DIR="$CPP_DIR/build-server"

cmake -S "$CPP_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=TRUE
cmake --build "$BUILD_DIR" \
  --target random_clustering_updated0 cluster_eval_disk_v4 -j4
```

For the standalone copy in this repository, use the build command in the
previous section and run `build/simq_index_builder` plus `build/simq_query`.
The two entry points have the same high-level pipeline, but the companion
experiment evaluator contains the additional production measurement paths
used for the paper, including the exact log format and dataset conventions.

### 2. Build one paper-style index

```bash
export DATASET=quora
export EMB_DIR="/home/yifan/Dataset/$DATASET/colbert_64d"
export INDEX_DIR="$EMB_DIR/SIMQ_alpha0p10_deferredK2_max160_mt64"
mkdir -p /home/yifan/multivector_experiment_p3/server_logs

"$BUILD_DIR/random_clustering_updated0" \
  --embeddings-dir "$EMB_DIR" \
  --output-dir "$INDEX_DIR" \
  --init-ratio 0.1 \
  --max-size 160 \
  --split-idx 80 \
  --seed 42 \
  --build-threads 64 \
  --batch-size 10000 \
  --deferred-kmeans-split \
  2>&1 | tee "/home/yifan/multivector_experiment_p3/server_logs/simq_${DATASET}_build.log"
```

Do not reuse an output directory from a different parameter setting. Check
that the output contains `centroid_hnsw_index.hnsw` before querying it. To
build another dataset, change only `DATASET` and `INDEX_DIR` while keeping the
parameters explicit.

### 3. Run one fixed-parameter evaluation

```bash
DATASET=quora M=100 RERANK_DOCS=1000 QUERY_THREADS=32 \
  DATASETS=quora \
  bash /home/yifan/multivector_experiment_p3/simq_parameter_runner.txt
```

This runs with `ef=100` and `final-k=100`, checks the embedding/index/reorder/
qrels paths, and writes a log under
`/home/yifan/multivector_experiment_p3/server_logs/simq_parameter_eval/`.
For example, `RERANK_DOCS=5000` changes only the candidate count; it does not
change the final output size unless `final-k` is changed in the script.

### 4. Sweep `m` on selected datasets

The standard five-dataset sweep reuses already-built indexes and does not
rebuild them:

```bash
M_VALUES="2 5 10 20 30 40 50 60 70 80 90 100" \
DATASETS="quora nq" \
QUERY_THREADS=32 \
SIMQ_RERANK_DOCS=1000 \
SIMQ_EF=100 \
SIMQ_FINAL_K=100 \
bash /home/yifan/multivector_experiment_p3/simq_m_sweep32.txt
```

For the larger sweep used in the recorded alpha experiments:

```bash
bash /home/yifan/multivector_experiment_p3/run_alpha_query_existing32.sh
```

That script uses `query_threads=32`, `ef=100`, `coarse_k=1000`,
`final_k=100`, and the explicit `m` list in the script. It only queries
existing alpha indexes and stops a parameter series when an index check or
query fails.

### 5. Run all five standard datasets

```bash
DATASETS="scidocs quora hotpotqa lotte_beir nq" \
M_VALUES="2 5 15 20 25 30 35 40 45 50" \
bash /home/yifan/multivector_experiment_p3/simq_m_sweep32.txt
```

The valid dataset names are `scidocs`, `quora`, `hotpotqa`, `lotte_beir`, and
`nq`. The script maps each name to its `colbert_64d` directory, the standard
`HNSW_cluster_efc100` index, `reorder_embedding`, and
`search_results_top100.tsv`.

### 6. Inspect and extract results

```bash
rg -n "QPS=|P100@100|R100@100|Recall|Dataset=" \
  /home/yifan/multivector_experiment_p3/server_logs/
```

Keep the raw log together with the exported table. The reported main-plot
point is the QPS and `Recall@100@100` printed for the same run; do not combine
QPS from one parameter setting with recall from another.

## License and attribution

See `hnswlib/LICENSE` for the vendored HNSW library license and the repository
license for this implementation. Cite the SIMQ paper and the upstream HNSW
project when using this code in published work.

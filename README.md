# SIMQ-Implement

This repository contains the CPU implementation used for the SIMQ main
experiments. SIMQ is a multi-vector retrieval index for ColBERT-style token
embeddings. It partitions document token vectors into data-aware clusters,
builds a graph over cluster representatives, retrieves candidate clusters for
each query token, and applies exact MaxSim reranking to the candidate
documents.

This repository is the implementation companion to the SIMQ paper. The
paper's larger multi-dataset reproduction scripts may be kept in a separate
experiment workspace; this README documents the required interfaces and the
commands needed to run them. All commands use shell variables so that the
instructions remain portable across machines.

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
  --embeddings-dir /path/to/colbert \
  --output-dir /path/to/colbert/SIMQ_index \
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

The query executable runs one complete evaluation directly from the command
line and exits after printing the results. A representative main-experiment
invocation is:

```bash
./build/simq_query \
  --embeddings-dir /path/to/colbert \
  --cluster-dir /path/to/colbert/SIMQ_index \
  --reorder-dir /path/to/colbert/reorder_embedding \
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

Parameter sweeps are implemented by starting the program once per parameter
setting with the corresponding command-line options.

The evaluation prints throughput together with the precision and recall values
computed by the current evaluator: `P10@10`, `P20@20`, `P50@50`, `P100@100`,
`R10@10`, `R20@20`, `R50@50`, and `R100@100`. The `Rk@k` values use the top-k
qrels as ground truth for the same k.

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
workload. They use only the binaries and files in this repository plus an
external dataset directory. Set `SIMQ_ROOT` and `DATA_BASE` once, then reuse
them for all commands.

### 1. Build the executables

```bash
export SIMQ_ROOT=/path/to/SIMQ-Implement
export DATA_BASE=/path/to/Dataset
cd "$SIMQ_ROOT"
export CPP_DIR="$SIMQ_ROOT/cpp"
export BUILD_DIR="$SIMQ_ROOT/build"

cmake -S "$CPP_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=TRUE
cmake --build "$BUILD_DIR" -j4
```

The build produces `build/simq_index_builder` and `build/simq_query`.

### 2. Build one paper-style index

```bash
export DATASET=quora
export DATA_BASE=/path/to/Dataset
export EMB_DIR="$DATA_BASE/$DATASET/colbert"
export INDEX_DIR="$EMB_DIR/SIMQ_alpha0p10_deferredK2_max160_mt64"
mkdir -p "$SIMQ_ROOT/logs"

"$BUILD_DIR/simq_index_builder" \
  --embeddings-dir "$EMB_DIR" \
  --output-dir "$INDEX_DIR" \
  --init-ratio 0.1 \
  --max-size 160 \
  --split-idx 80 \
  --seed 42 \
  --build-threads 64 \
  --batch-size 10000 \
  --deferred-kmeans-split \
  2>&1 | tee "$SIMQ_ROOT/logs/simq_${DATASET}_build.log"
```

Do not reuse an output directory from a different parameter setting. Check
that the output contains `centroid_hnsw_index.hnsw` before querying it. To
build another dataset, change only `DATASET` and `INDEX_DIR` while keeping the
parameters explicit.

### 3. Run one fixed-parameter evaluation

```bash
export EMB_DIR="$DATA_BASE/quora/colbert"
export INDEX_DIR="$EMB_DIR/SIMQ_alpha0p10_deferredK2_max160_mt64"
export REORDER_DIR="$EMB_DIR/reorder_embedding"
export QRELS="$EMB_DIR/search_results_top100.tsv"

"$BUILD_DIR/simq_query" \
  --embeddings-dir "$EMB_DIR" \
  --cluster-dir "$INDEX_DIR" \
  --reorder-dir "$REORDER_DIR" \
  --qrels-file "$QRELS" \
  --m 100 --ef 100 --coarse-k 1000 --final-k 100 --query-threads 32 \
  2>&1 | tee "$SIMQ_ROOT/logs/simq_quora_m100.log"
```

This runs with `ef=100` and `final-k=100`, checks the embedding/index/reorder/
qrels paths, and writes a log under
`$SIMQ_ROOT/logs/`.
For example, use `--coarse-k 5000` to rerank 5,000 candidates; the final output
remains top-100 because `--final-k 100` is unchanged.

### 4. Sweep `m` on selected datasets

The standard five-dataset sweep reuses already-built indexes and does not
rebuild them:

```bash
for M in 2 5 10 20 30 40 50 60 70 80 90 100; do
  "$BUILD_DIR/simq_query" \
    --embeddings-dir "$DATA_BASE/quora/colbert" \
    --cluster-dir "$DATA_BASE/quora/colbert/SIMQ_alpha0p10_deferredK2_max160_mt64" \
    --reorder-dir "$DATA_BASE/quora/colbert/reorder_embedding" \
    --qrels-file "$DATA_BASE/quora/search_results_top100.tsv" \
    --m "$M" --ef 100 --coarse-k 1000 --final-k 100 --query-threads 32 \
    2>&1 | tee "$SIMQ_ROOT/logs/simq_quora_m${M}.log"
done
```

To compare several saved alpha indexes, repeat the same command and change
only `INDEX_DIR`:

```bash
for ALPHA in 0p05 0p08 0p15 0p20; do
  export INDEX_DIR="$EMB_DIR/SIMQ_alpha${ALPHA}_deferredK2_max160_mt64"
  "$BUILD_DIR/simq_query" \
    --embeddings-dir "$EMB_DIR" \
    --cluster-dir "$INDEX_DIR" \
    --reorder-dir "$REORDER_DIR" \
    --qrels-file "$QRELS" \
    --m 100 --ef 100 --coarse-k 1000 --final-k 100 --query-threads 32 \
    2>&1 | tee "$SIMQ_ROOT/logs/simq_quora_alpha${ALPHA}_m100.log"
done
```

This uses `query_threads=32`, `ef=100`, `coarse_k=1000`, and `final_k=100`.
It queries existing alpha indexes and does not rebuild them.

### 5. Run all five standard datasets

```bash
for DATASET in scidocs quora hotpotqa lotte_beir nq; do
  EMB_DIR="$DATA_BASE/$DATASET/colbert"
  INDEX_DIR="$EMB_DIR/SIMQ_alpha0p10_deferredK2_max160_mt64"
  "$BUILD_DIR/simq_query" \
    --embeddings-dir "$EMB_DIR" \
    --cluster-dir "$INDEX_DIR" \
    --reorder-dir "$EMB_DIR/reorder_embedding" \
    --qrels-file "$EMB_DIR/search_results_top100.tsv" \
    --m 100 --ef 100 --coarse-k 1000 --final-k 100 --query-threads 32 \
    2>&1 | tee "$SIMQ_ROOT/logs/simq_${DATASET}_m100.log"
done
```

The valid dataset names are `scidocs`, `quora`, `hotpotqa`, `lotte_beir`, and
`nq`. Each dataset uses a `colbert` directory, a saved SIMQ index,
`reorder_embedding`, and `search_results_top100.tsv`.

### 6. Inspect and extract results

```bash
rg -n "QPS=|P100@100|R100@100|Recall|Dataset=" \
  "$SIMQ_ROOT/logs/"
```

Keep the raw log together with the exported table. The reported main-plot
point is the QPS and `Recall@100@100` printed for the same run; do not combine
QPS from one parameter setting with recall from another.

## License and attribution

See `hnswlib/LICENSE` for the vendored HNSW library license and the repository
license for this implementation. Cite the SIMQ paper and the upstream HNSW
project when using this code in published work.

// SIMQ index construction.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "npy_reader.h"
#include "parallel_documents.h"
#include "progress_reporter.h"
#include "../hnswlib/hnswlib/hnswlib.h"

struct ClusterConfig {
    float  init_cluster_ratio = 0.2f;
    int    max_cluster_size   = 82;
    int    split_start_idx    = 42;
    int    random_seed        = 42;
    int    build_threads      = 8;
    int    batch_size         = 10000;
    double split_delay_seconds = 1.0;
    bool   build_representative_hnsw = true;
    // When enabled, defer all overflow handling until assignment completes and
    // recursively split oversized clusters with the NSG-style local K=2
    // cosine K-means routine.
    bool   deferred_kmeans_split = false;
};

struct ClusterMember {
    int   vector_id;
    float distance;
};

class HNSWDynamicClustering {
public:
    explicit HNSWDynamicClustering(const ClusterConfig& cfg = {}) : config_(cfg) {}

    void fit(const NpyArray& vectors);
    void save(const std::string& output_dir) const;
    size_t get_representative_count() const { return representative_vids_.size(); }

private:
    void build_hnsw_index(const std::vector<int>& center_ids, int dim);
    int  find_nearest_cluster(int vector_id);
    float compute_distance(int v1, int v2) const;
    void add_to_cluster(int vector_id, int center_id, float distance);
    void split_cluster(int old_center_id, int dim);
    void flush_pending_splits(const std::unordered_set<int>* eligible = nullptr);
    void deferred_kmeans_split_oversized();

    // Key change: instead of computing a synthetic centroid, pick the
    // cluster member that is closest to the unnormalized mean.
    void find_cluster_representatives();
    void build_representative_hnsw_index();

    ClusterConfig config_;

    // data
    const NpyArray* vectors_ = nullptr;
    int dim_ = 0;

    // clustering state
    std::vector<int> cluster_centers_;
    std::unordered_map<int, std::vector<ClusterMember>> clusters_;
    std::unordered_map<int, int> vector_to_cluster_;
    std::unordered_map<int, std::chrono::steady_clock::time_point> pending_split_since_;

    // HNSW used during the clustering process (on cluster seeds)
    hnswlib::InnerProductSpace* space_ = nullptr;
    hnswlib::HierarchicalNSW<float>* hnsw_ = nullptr;
    std::unordered_map<size_t, int> hnsw_id_to_vec_;
    std::unordered_map<int, size_t> vec_to_hnsw_id_;

    // medoid result: representative_vids_[new_cluster_idx] = vec_id
    std::vector<int> representative_vids_;
    std::unordered_map<int, int> old_to_new_mapping_;  // old center id → new cluster idx

    // HNSW built on representative actual vectors (replaces centroid HNSW)
    hnswlib::InnerProductSpace* rep_space_ = nullptr;
    hnswlib::HierarchicalNSW<float>* rep_hnsw_ = nullptr;
};

void HNSWDynamicClustering::build_hnsw_index(
        const std::vector<int>& center_ids, int dim) {
    delete hnsw_;
    delete space_;

    space_ = new hnswlib::InnerProductSpace(dim);
    size_t max_capacity = vectors_->rows;
    hnsw_  = new hnswlib::HierarchicalNSW<float>(
                    space_, max_capacity, /*M=*/16, /*efC=*/100);

    hnsw_id_to_vec_.clear();
    vec_to_hnsw_id_.clear();

    const int total = (int)center_ids.size();
    for (int i = 0; i < total; ++i) {
        hnsw_id_to_vec_[i] = center_ids[i];
        vec_to_hnsw_id_[center_ids[i]] = i;
    }
    ProgressReporter progress("Initial center HNSW", total);
    if (total) {
        hnsw_->addPoint(vectors_->row_ptr(center_ids[0]), 0);
        progress.advance();
        parallel_documents(total - 1, config_.build_threads, [&](int j) {
            int i = j + 1;
            hnsw_->addPoint(vectors_->row_ptr(center_ids[i]), i);
            progress.advance();
        });
    }
    progress.finish();
    hnsw_->setEf(50);
}

int HNSWDynamicClustering::find_nearest_cluster(int vector_id) {
    auto result = hnsw_->searchKnn(vectors_->row_ptr(vector_id), 1);
    size_t hnsw_id = result.top().second;
    return hnsw_id_to_vec_.at(hnsw_id);
}

float HNSWDynamicClustering::compute_distance(int v1, int v2) const {
    const float* a = vectors_->row_ptr(v1);
    const float* b = vectors_->row_ptr(v2);
    float dot = 0.0f;
    for (int i = 0; i < dim_; i++) dot += a[i] * b[i];
    return 1.0f - dot;  // cosine distance (vectors are L2-normalized)
}

void HNSWDynamicClustering::add_to_cluster(
        int vector_id, int center_id, float distance) {
    auto& cluster = clusters_[center_id];
    // binary insertion to keep sorted by distance
    int lo = 0, hi = (int)cluster.size();
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cluster[mid].distance < distance) lo = mid + 1;
        else hi = mid;
    }
    cluster.insert(cluster.begin() + lo, {vector_id, distance});
    vector_to_cluster_[vector_id] = center_id;
}

void HNSWDynamicClustering::split_cluster(int old_center_id, int dim) {
    auto& cluster = clusters_[old_center_id];

    int new_center_id = cluster[config_.max_cluster_size - 1].vector_id;

    std::vector<ClusterMember> to_reassign(
        cluster.begin() + (config_.split_start_idx - 1),
        cluster.begin() + config_.max_cluster_size);

    cluster.erase(cluster.begin() + (config_.split_start_idx - 1),
                  cluster.begin() + config_.max_cluster_size);

    std::vector<ClusterMember> new_cluster;
    new_cluster.push_back({new_center_id, 0.0f});

    for (auto& m : to_reassign) {
        if (m.vector_id == new_center_id) continue;
        float d = compute_distance(m.vector_id, new_center_id);
        int lo = 0, hi = (int)new_cluster.size();
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (new_cluster[mid].distance < d) lo = mid + 1;
            else hi = mid;
        }
        new_cluster.insert(new_cluster.begin() + lo, {m.vector_id, d});
        vector_to_cluster_[m.vector_id] = new_center_id;
    }

    vector_to_cluster_[new_center_id] = new_center_id;
    clusters_[new_center_id] = std::move(new_cluster);
    cluster_centers_.push_back(new_center_id);

    size_t new_hnsw_id = cluster_centers_.size() - 1;
    hnsw_->addPoint(vectors_->row_ptr(new_center_id), new_hnsw_id);
    hnsw_id_to_vec_[new_hnsw_id] = new_center_id;
    vec_to_hnsw_id_[new_center_id] = new_hnsw_id;
}

void HNSWDynamicClustering::flush_pending_splits(const std::unordered_set<int>* eligible) {
    struct SplitPlan {
        int old_center = -1;
        int new_center = -1;
        std::vector<ClusterMember> moved;
        std::vector<float> distances;
        std::vector<ClusterMember> new_members;
    };
    std::vector<SplitPlan> plans;
    for (const auto& [old_center, since] : pending_split_since_) {
        if (eligible && !eligible->count(old_center)) continue;
        auto it = clusters_.find(old_center);
        if (it == clusters_.end() || (int)it->second.size() < config_.max_cluster_size)
            continue;
        auto& cluster = it->second;
        SplitPlan plan;
        plan.old_center = old_center;
        plan.new_center = cluster[config_.max_cluster_size - 1].vector_id;
        plan.moved.assign(cluster.begin() + (config_.split_start_idx - 1),
                          cluster.end());
        plan.distances.resize(plan.moved.size(), 0.0f);
        plans.push_back(std::move(plan));
    }
    if (plans.empty()) return;

    // Prepare all independent split calculations before changing any cluster
    parallel_documents((int)plans.size(), config_.build_threads, [&](int i) {
        auto& plan = plans[(size_t)i];
        for (size_t j = 0; j < plan.moved.size(); ++j)
            plan.distances[j] = compute_distance(plan.moved[j].vector_id, plan.new_center);
        plan.new_members.push_back({plan.new_center, 0.0f});
        for (size_t j = 0; j < plan.moved.size(); ++j) {
            int vid = plan.moved[j].vector_id;
            if (vid == plan.new_center) continue;
            plan.new_members.push_back({vid, plan.distances[j]});
        }
        std::sort(plan.new_members.begin(), plan.new_members.end(),
                  [](const ClusterMember& a, const ClusterMember& b) {
                      return a.distance < b.distance;
                  });
    });

    // Commit serially: HNSW insertion and cluster maps are shared mutable state.
    for (auto& plan : plans) {
        auto& old_cluster = clusters_.at(plan.old_center);
        old_cluster.erase(old_cluster.begin() + (config_.split_start_idx - 1),
                          old_cluster.end());
        for (const auto& member : plan.new_members)
            vector_to_cluster_[member.vector_id] = plan.new_center;
        vector_to_cluster_[plan.old_center] = plan.old_center;
        clusters_[plan.new_center] = std::move(plan.new_members);
        cluster_centers_.push_back(plan.new_center);
        size_t hnsw_id = cluster_centers_.size() - 1;
        hnsw_->addPoint(vectors_->row_ptr(plan.new_center), hnsw_id);
        hnsw_id_to_vec_[hnsw_id] = plan.new_center;
        vec_to_hnsw_id_[plan.new_center] = hnsw_id;
    }
    for (const auto& plan : plans) pending_split_since_.erase(plan.old_center);
}

void HNSWDynamicClustering::deferred_kmeans_split_oversized() {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<int> pending = cluster_centers_;
    std::vector<int> refined;
    refined.reserve(cluster_centers_.size());
    size_t split_count = 0;

    while (!pending.empty()) {
        int old_center = pending.back();
        pending.pop_back();
        auto it = clusters_.find(old_center);
        if (it == clusters_.end()) continue;
        auto members = std::move(it->second);
        clusters_.erase(it);

        if ((int)members.size() <= config_.max_cluster_size) {
            clusters_[old_center] = std::move(members);
            refined.push_back(old_center);
            for (const auto& m : clusters_[old_center]) vector_to_cluster_[m.vector_id] = old_center;
            continue;
        }

        const size_t n = members.size();
        std::vector<float> centers((size_t)2 * dim_, 0.0f);
        for (size_t c = 0; c < 2; ++c) {
            int vid = members[(c * n) / 2].vector_id;
            std::copy_n(vectors_->row_ptr(vid), dim_, centers.data() + c * (size_t)dim_);
        }
        std::vector<uint8_t> assignment(n, 0);
        for (int iter = 0; iter < 8; ++iter) {
            uint64_t counts[2] = {0, 0};
            std::vector<double> sums((size_t)2 * dim_, 0.0);
            for (size_t i = 0; i < n; ++i) {
                const float* v = vectors_->row_ptr(members[i].vector_id);
                float best_score = -std::numeric_limits<float>::infinity();
                uint8_t best = 0;
                for (uint8_t c = 0; c < 2; ++c) {
                    const float* center = centers.data() + (size_t)c * dim_;
                    float score = 0.0f;
                    for (int d = 0; d < dim_; ++d) score += v[d] * center[d];
                    if (score > best_score) { best_score = score; best = c; }
                }
                assignment[i] = best;
                ++counts[best];
                for (int d = 0; d < dim_; ++d) sums[(size_t)best * dim_ + d] += v[d];
            }
            // Keep both children non-empty, matching the NSG implementation.
            for (uint8_t empty = 0; empty < 2; ++empty) {
                if (counts[empty] != 0) continue;
                uint8_t donor = counts[0] >= counts[1] ? 0 : 1;
                if (donor == empty || counts[donor] <= 1) {
                    donor = (empty == 0 ? 1 : 0);
                }
                size_t moved = n;
                for (size_t i = 0; i < n; ++i) {
                    if (assignment[i] == donor) { moved = i; break; }
                }
                if (moved == n) throw std::runtime_error("deferred K-means could not create two non-empty children");
                assignment[moved] = empty;
            }
            std::fill(std::begin(counts), std::end(counts), 0);
            std::fill(sums.begin(), sums.end(), 0.0);
            for (size_t i = 0; i < n; ++i) {
                uint8_t c = assignment[i];
                const float* v = vectors_->row_ptr(members[i].vector_id);
                ++counts[c];
                for (int d = 0; d < dim_; ++d) sums[(size_t)c * dim_ + d] += v[d];
            }
            for (uint8_t c = 0; c < 2; ++c) {
                float norm = 0.0f;
                for (int d = 0; d < dim_; ++d) {
                    float x = (float)(sums[(size_t)c * dim_ + d] / (double)counts[c]);
                    centers[(size_t)c * dim_ + d] = x; norm += x * x;
                }
                if (norm > 0.0f) {
                    float inv = 1.0f / std::sqrt(norm);
                    for (int d = 0; d < dim_; ++d) centers[(size_t)c * dim_ + d] *= inv;
                }
            }
        }

        std::vector<ClusterMember> parts[2];
        for (size_t i = 0; i < n; ++i) parts[assignment[i]].push_back(members[i]);
        ++split_count;
        for (int c = 0; c < 2; ++c) {
            if (parts[c].empty()) throw std::runtime_error("deferred K-means produced an empty child");
            int child_center = parts[c].front().vector_id;
            std::vector<ClusterMember> sorted;
            sorted.reserve(parts[c].size());
            sorted.push_back({child_center, 0.0f});
            for (const auto& m : parts[c]) {
                if (m.vector_id == child_center) continue;
                sorted.push_back({m.vector_id, compute_distance(m.vector_id, child_center)});
            }
            std::sort(sorted.begin(), sorted.end(), [](const ClusterMember& a, const ClusterMember& b) {
                return a.distance < b.distance;
            });
            clusters_[child_center] = std::move(sorted);
            for (const auto& m : clusters_[child_center]) vector_to_cluster_[m.vector_id] = child_center;
            pending.push_back(child_center);
        }
    }
    cluster_centers_.swap(refined);
    std::cout << "  Deferred local K-means: max_size=" << config_.max_cluster_size
              << " refined_clusters=" << cluster_centers_.size()
              << " split_source_clusters=" << split_count
              << " seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << "\n";
}

//
// For each cluster:
//   1. Compute the unnormalized mean of all member vectors.
//   2. Find the member v* that maximises  dot(v*, mean).
//      (Equivalent to finding the actual vector closest to the centroid
//       under inner-product similarity when vectors are L2-normalized.)
//   3. Use v* as the cluster representative.
void HNSWDynamicClustering::find_cluster_representatives() {
    representative_vids_.clear();
    old_to_new_mapping_.clear();

    auto t0 = std::chrono::steady_clock::now();
    size_t total = cluster_centers_.size();
    size_t step = std::max((size_t)1, total / 100);  
    size_t count = 0;

    for (int cid : cluster_centers_) {
        auto& members = clusters_[cid];

        // Step 1: unnormalized mean
        std::vector<float> mean(dim_, 0.0f);
        for (auto& m : members) {
            const float* v = vectors_->row_ptr(m.vector_id);
            for (int d = 0; d < dim_; d++) mean[d] += v[d];
        }
        // (no normalization — we only need it for ranking, not for a unit vector)

        // Step 2: find member with highest dot product to mean
        float best_dot = -1e30f;
        int   best_vid = members[0].vector_id;
        for (auto& m : members) {
            const float* v = vectors_->row_ptr(m.vector_id);
            float dot = 0.0f;
            for (int d = 0; d < dim_; d++) dot += v[d] * mean[d];
            if (dot > best_dot) {
                best_dot = dot;
                best_vid  = m.vector_id;
            }
        }

        representative_vids_.push_back(best_vid);
        old_to_new_mapping_[cid] = count++;

        if (count % step == 0 || count == total) {
            auto now = std::chrono::steady_clock::now();
            double sec = std::chrono::duration<double>(now - t0).count();
            double pct = 100.0 * count / total;
            double speed = count / sec;
            std::cout << "  [Medoid] " << count << "/" << total
                      << " (" << std::fixed << std::setprecision(1) << pct << "%) | "
                      << std::setprecision(0) << speed << " cluster/s\r" << std::flush;
        }
    }
    std::cout << std::endl;
    std::cout << "  Found " << representative_vids_.size()
              << " cluster representatives (medoids)\n";
}

void HNSWDynamicClustering::build_representative_hnsw_index() {
    delete rep_hnsw_;
    delete rep_space_;

    rep_space_ = new hnswlib::InnerProductSpace(dim_);
    rep_hnsw_  = new hnswlib::HierarchicalNSW<float>(
                     rep_space_, representative_vids_.size(),
                     /*M=*/16, /*efC=*/100);

    const int total = (int)representative_vids_.size();
    ProgressReporter progress("Representative HNSW", total);
    if (total) {
        rep_hnsw_->addPoint(vectors_->row_ptr(representative_vids_[0]), 0);
        progress.advance();
        parallel_documents(total - 1, config_.build_threads, [&](int j) {
            int i = j + 1;
            rep_hnsw_->addPoint(vectors_->row_ptr(representative_vids_[i]), i);
            progress.advance();
        });
    }
    progress.finish();
    rep_hnsw_->setEf(50);
    std::cout << "  Representative HNSW index built ("
              << representative_vids_.size() << " nodes)\n";
}

void HNSWDynamicClustering::fit(const NpyArray& vectors) {
    if (!vectors.rows || !vectors.cols || vectors.rows > (size_t)std::numeric_limits<int>::max())
        throw std::runtime_error("Clustering requires nonempty vectors and at most INT_MAX tokens");
    vectors_ = &vectors;
    dim_ = (int)vectors.cols;
    int num_vectors = (int)vectors.rows;

    std::cout << "Clustering: " << num_vectors
              << " vectors, dim=" << dim_ << "\n";

    std::mt19937 rng(config_.random_seed);

    // Step 1: select initial cluster centers
    int num_init = std::max(1, (int)(num_vectors * config_.init_cluster_ratio));
    std::vector<int> all_indices(num_vectors);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::shuffle(all_indices.begin(), all_indices.end(), rng);

    std::vector<int> init_centers(all_indices.begin(),
                                  all_indices.begin() + num_init);
    std::vector<int> remaining(all_indices.begin() + num_init,
                                all_indices.end());

    cluster_centers_ = init_centers;
    for (int cid : init_centers) {
        clusters_[cid] = {{cid, 0.0f}};
        vector_to_cluster_[cid] = cid;
    }
    std::cout << "[1/6] Initial centers: " << num_init << "\n";

    // Step 2: build HNSW on initial centers
    std::cout << "[2/6] Building HNSW index on initial centers...\n";
    build_hnsw_index(init_centers, dim_);

    // Step 3: assign remaining vectors
    std::cout << "[3/6] Assigning " << remaining.size()
              << " remaining vectors...\n";
    std::cout << "  Build threads=" << config_.build_threads
              << " batch_size=" << config_.batch_size << "\n";
    ProgressReporter progress("Assigned vectors", remaining.size());
    for (size_t begin = 0; begin < remaining.size(); begin += config_.batch_size) {
        int count = (int)std::min<size_t>(config_.batch_size, remaining.size() - begin);
        std::vector<int> nearest(count);
        std::vector<float> distances(count);
        // Read-only center graph for the whole batch. The implicit barrier
        // finishes every search before any split/addPoint can modify the graph.
        parallel_documents(count, config_.build_threads, [&](int j) {
            int vid = remaining[begin + j];
            nearest[j] = find_nearest_cluster(vid);
            distances[j] = compute_distance(vid, nearest[j]);
        });
        // Preserve input order for membership updates. Overflow is queued and
        // handled after a delay so independent clusters can split together.
        for (int j = 0; j < count; ++j) {
            int vid = remaining[begin + j];
            add_to_cluster(vid, nearest[j], distances[j]);
            progress.advance();
        }
        if (!config_.deferred_kmeans_split) {
            auto now = std::chrono::steady_clock::now();
            std::unordered_set<int> current_overflow;
            for (const auto& [cid, members] : clusters_)
                if ((int)members.size() >= config_.max_cluster_size)
                    current_overflow.insert(cid);
            for (int cid : current_overflow)
                if (!pending_split_since_.count(cid)) pending_split_since_[cid] = now;
            for (auto it = pending_split_since_.begin(); it != pending_split_since_.end(); ) {
                if (!current_overflow.count(it->first)) it = pending_split_since_.erase(it);
                else ++it;
            }
            std::unordered_set<int> ready;
            for (const auto& [cid, since] : pending_split_since_)
                if (std::chrono::duration<double>(now - since).count() >= config_.split_delay_seconds)
                    ready.insert(cid);
            if (!ready.empty()) {
                std::cout << "  flushing " << ready.size()
                          << " time-delayed split tasks\n";
                flush_pending_splits(&ready);
            }
        }
    }
    progress.finish();
    if (!config_.deferred_kmeans_split && !pending_split_since_.empty()) {
        std::cout << "  flushing final delayed split tasks\n";
        flush_pending_splits();
    }

    if (config_.deferred_kmeans_split) {
        std::cout << "[4/6] All assignments complete; applying deferred local K=2 K-means...\n";
        deferred_kmeans_split_oversized();
    }

    std::cout << "[4/6] Cluster member order validated (binary insert)\n";

    // Step 5: find medoid representatives (replaces centroid computation)
    std::cout << "[5/6] Finding cluster representatives (medoids)...\n";
    find_cluster_representatives();

    // Step 6: build HNSW on representative actual vectors
    if (config_.build_representative_hnsw) {
        std::cout << "[6/6] Building representative HNSW index...\n";
        build_representative_hnsw_index();
    } else {
        std::cout << "[6/6] Skipping representative HNSW; CAGRA center graph is built separately.\n";
    }

    // statistics
    std::vector<int> sizes;
    for (auto& [cid, members] : clusters_)
        sizes.push_back((int)members.size());
    std::sort(sizes.begin(), sizes.end());
    double avg =
        std::accumulate(sizes.begin(), sizes.end(), 0.0) / sizes.size();
    std::cout << "\nClustering statistics:\n"
              << "  Total vectors: " << num_vectors << "\n"
              << "  Clusters:      " << cluster_centers_.size() << "\n"
              << "  Avg size:      " << avg << "\n"
              << "  Min size:      " << sizes.front() << "\n"
              << "  Max size:      " << sizes.back() << "\n"
              << "  Median:        " << sizes[sizes.size() / 2] << "\n";
}

void HNSWDynamicClustering::save(const std::string& output_dir) const {
    std::string cmd = "mkdir -p " + output_dir;
    system(cmd.c_str());

    // 1. cluster_centers.txt
    {
        std::ofstream f(output_dir + "/cluster_centers.txt");
        f << "# cluster_index\tvector_id\n";
        for (size_t i = 0; i < cluster_centers_.size(); i++)
            f << i << "\t" << cluster_centers_[i] << "\n";
        std::cout << "  Saved cluster_centers.txt\n";
    }

    // 2. cluster_assignments.txt
    {
        std::ofstream f(output_dir + "/cluster_assignments.txt");
        f << "# vector_id\tcluster_index\n";
        std::vector<std::pair<int, int>> assignments;
        for (auto& [vid, cid] : vector_to_cluster_) {
            auto it = old_to_new_mapping_.find(cid);
            if (it != old_to_new_mapping_.end())
                assignments.push_back({vid, it->second});
        }
        std::sort(assignments.begin(), assignments.end());
        for (auto& [vid, cidx] : assignments)
            f << vid << "\t" << cidx << "\n";
        std::cout << "  Saved cluster_assignments.txt\n";
    }

    // 3. cluster_members.json
    {
        std::ofstream f(output_dir + "/cluster_members.json");
        f << "{\n";
        bool first_cluster = true;
        for (auto& [old_cid, members] : clusters_) {
            auto it = old_to_new_mapping_.find(old_cid);
            if (it == old_to_new_mapping_.end()) continue;
            int new_idx = it->second;

            if (!first_cluster) f << ",\n";
            first_cluster = false;

            f << "  \"" << new_idx << "\": [";
            for (size_t i = 0; i < members.size(); i++) {
                if (i > 0) f << ",";
                f << members[i].vector_id;
            }
            f << "]";
        }
        f << "\n}\n";
        std::cout << "  Saved cluster_members.json\n";
    }

    // 4. cluster_representatives.txt  (replaces cluster_centroids.npy)
    //    Format: cluster_index <TAB> representative_vec_id
    {
        std::ofstream f(output_dir + "/cluster_representatives.txt");
        f << "# cluster_index\trepresentative_vec_id\n";
        for (size_t i = 0; i < representative_vids_.size(); i++)
            f << i << "\t" << representative_vids_[i] << "\n";
        std::cout << "  Saved cluster_representatives.txt\n";
    }

    // 5. Optional centroid_hnsw_index.hnsw for the HNSW-based evaluator.
    if (rep_hnsw_) {
        std::string path = output_dir + "/centroid_hnsw_index.hnsw";
        rep_hnsw_->saveIndex(path);
        std::cout << "  Saved centroid_hnsw_index.hnsw"
                     " (representative vectors)\n";
    }

    // 6. centroid_mapping.json
    {
        std::ofstream f(output_dir + "/centroid_mapping.json");
        f << "{\n";
        bool first = true;
        for (auto& [old_id, new_idx] : old_to_new_mapping_) {
            if (!first) f << ",\n";
            first = false;
            f << "  \"" << old_id << "\": " << new_idx;
        }
        f << "\n}\n";
        std::cout << "  Saved centroid_mapping.json\n";
    }

    // 7. clustering_meta.json
    {
        std::ofstream f(output_dir + "/clustering_meta.json");
        std::vector<int> sizes;
        for (auto& [cid, members] : clusters_)
            sizes.push_back((int)members.size());
        double avg =
            std::accumulate(sizes.begin(), sizes.end(), 0.0) / sizes.size();
        int mx = *std::max_element(sizes.begin(), sizes.end());
        int mn = *std::min_element(sizes.begin(), sizes.end());

        f << "{\n"
          << "  \"num_vectors\": " << vectors_->rows << ",\n"
          << "  \"num_clusters\": " << cluster_centers_.size() << ",\n"
          << "  \"vector_dim\": " << dim_ << ",\n"
          << "  \"representative_mode\": \"medoid\",\n"
          << "  \"config\": {\n"
          << "    \"init_cluster_ratio\": " << config_.init_cluster_ratio << ",\n"
          << "    \"max_cluster_size\": "   << config_.max_cluster_size   << ",\n"
          << "    \"split_start_idx\": "    << config_.split_start_idx    << ",\n"
          << "    \"build_threads\": " << config_.build_threads << ",\n"
          << "    \"batch_size\": " << config_.batch_size << ",\n"
          << "    \"split_delay_seconds\": " << config_.split_delay_seconds << ",\n"
          << "    \"deferred_kmeans_split\": " << (config_.deferred_kmeans_split ? "true" : "false") << ",\n"
          << "    \"random_seed\": "        << config_.random_seed        << "\n"
          << "  },\n"
          << "  \"statistics\": {\n"
          << "    \"avg_cluster_size\": " << avg << ",\n"
          << "    \"max_cluster_size\": " << mx  << ",\n"
          << "    \"min_cluster_size\": " << mn  << "\n"
          << "  }\n"
          << "}\n";
        std::cout << "  Saved clustering_meta.json\n";
    }

    std::cout << "All results saved to " << output_dir << "\n";
}

int main(int argc, char** argv) {
    std::string embeddings_dir = "Jie_Guo/colbert_64d";
    std::string output_dir     = "Jie_Guo/colbert_64d/HNSW_cluster_rep";
    float  init_ratio = 0.2f;
    int    max_size   = 82;
    int    split_idx  = 42;
    int    seed       = 42;
    int build_threads = 8, batch_size = 10000;
    double split_delay_seconds = 1.0;
    bool skip_representative_hnsw = false;
    bool deferred_kmeans_split = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--embeddings-dir" && i+1 < argc) embeddings_dir = argv[++i];
        else if (arg == "--output-dir"     && i+1 < argc) output_dir     = argv[++i];
        else if (arg == "--init-ratio"     && i+1 < argc) init_ratio     = std::stof(argv[++i]);
        else if (arg == "--max-size"        && i+1 < argc) max_size      = std::stoi(argv[++i]);
        else if (arg == "--split-idx"       && i+1 < argc) split_idx     = std::stoi(argv[++i]);
        else if (arg == "--build-threads" && i+1 < argc) build_threads = std::stoi(argv[++i]);
        else if (arg == "--batch-size" && i+1 < argc) batch_size = std::stoi(argv[++i]);
        else if (arg == "--split-delay-seconds" && i+1 < argc) split_delay_seconds = std::stod(argv[++i]);
        else if (arg == "--seed"            && i+1 < argc) seed          = std::stoi(argv[++i]);
        else if (arg == "--skip-representative-hnsw") skip_representative_hnsw = true;
        else if (arg == "--deferred-kmeans-split") deferred_kmeans_split = true;
        else if (arg == "--help" || arg == "-h") {
            std::cout <<
                "Usage: " << argv[0] << " [options]\n"
                "  --embeddings-dir DIR   Embeddings dir (default: Jie_Guo/colbert_64d)\n"
                "  --output-dir DIR       Output dir    (default: Jie_Guo/colbert_64d/HNSW_cluster_rep)\n"
                "  --init-ratio FLOAT     Initial center ratio (default: 0.2)\n"
                "  --max-size INT         Max cluster size     (default: 82)\n"
                "  --split-idx INT        Split start index    (default: 42)\n"
                "  --build-threads INT    Build workers (default: 8)\n"
                "  --batch-size INT       Assignment batch (default: 10000; 1 for immediate assignment)\n"
                "  --split-delay-seconds FLOAT  Delay overflow before grouped split (default: 1.0)\n"
                "  --deferred-kmeans-split  Defer overflow handling; recursively split with local K=2 K-means\n"
                "  --skip-representative-hnsw  Do not build/save the optional final HNSW index\n"
                "  --seed INT             Random seed          (default: 42)\n";
            return 0;
        }
    }

    if (build_threads < 1 || batch_size < 1 || split_delay_seconds < 0 || !(init_ratio > 0 && init_ratio <= 1) ||
        max_size < 3 || split_idx < 2 || split_idx >= max_size) {
        std::cerr << "Invalid clustering parameters\n";
        return 1;
    }
    ClusterConfig config;
    config.build_threads = build_threads;
    config.batch_size = batch_size;
    config.split_delay_seconds = split_delay_seconds;
    config.init_cluster_ratio = init_ratio;
    config.max_cluster_size   = max_size;
    config.split_start_idx    = split_idx;
    config.random_seed        = seed;
    config.build_representative_hnsw = !skip_representative_hnsw;
    config.deferred_kmeans_split = deferred_kmeans_split;

    std::string emb_path  = embeddings_dir + "/colbert_embeddings.npy";
    std::string meta_path = embeddings_dir + "/colbert_embeddings_meta.json";

    std::cout << "Loading embeddings: " << emb_path << "\n";
    NpyArray all_embeddings = load_npy_float32(emb_path);
    std::cout << "  Shape: " << all_embeddings.rows
              << " x " << all_embeddings.cols << "\n";

    std::string meta_text    = read_file_text(meta_path);
    int doc_vector_start = (int)json_int(meta_text, "doc_vector_start");
    int doc_vector_end   = (int)json_int(meta_text, "doc_vector_end");

    std::cout << "Document vectors: " << doc_vector_start
              << " - " << (doc_vector_end - 1) << "\n";
    int num_doc_vectors = doc_vector_end - doc_vector_start;

    NpyArray doc_embeddings;
    doc_embeddings.cols = all_embeddings.cols;
    doc_embeddings.rows = num_doc_vectors;
    doc_embeddings.data.assign(
        all_embeddings.data.begin() + doc_vector_start * all_embeddings.cols,
        all_embeddings.data.begin() + doc_vector_end   * all_embeddings.cols);

    all_embeddings.data.clear();
    all_embeddings.data.shrink_to_fit();

    std::cout << "Document embeddings: " << doc_embeddings.rows
              << " x " << doc_embeddings.cols << "\n";

    HNSWDynamicClustering clusterer(config);

    auto build_start = std::chrono::steady_clock::now();

    clusterer.fit(doc_embeddings);
    clusterer.save(output_dir);

    auto build_end = std::chrono::steady_clock::now();
    double build_time_s = std::chrono::duration<double>(build_end - build_start).count();

    int64_t total_bytes = 0;
    for (const auto& entry : std::filesystem::directory_iterator(output_dir))
        if (entry.is_regular_file())
            total_bytes += (int64_t)entry.file_size();
    double total_mb = total_bytes / (1024.0 * 1024.0);

    size_t rss_kb = get_rss_kb();
    double rss_mb = rss_kb / 1024.0;

    // Each rep vector: dim * 4 bytes
    // HNSW index overhead: roughly 2-3x the raw vector size (for links)
    int num_reps = (int)clusterer.get_representative_count();
    int dim = (int)doc_embeddings.cols;
    size_t rep_vec_bytes = (size_t)num_reps * dim * 4;
    size_t hnsw_overhead = rep_vec_bytes * 2;
    double in_memory_mb = (rep_vec_bytes + hnsw_overhead) / (1024.0 * 1024.0);

    std::cout << "\n" << std::string(72, '=') << "\n";
    std::cout << "[Index Build Summary]\n";
    std::cout << std::string(72, '-') << "\n";
    std::cout << "  Build time:        " << std::fixed << std::setprecision(2)
              << build_time_s << " s\n";
    std::cout << "  Index size (disk): " << total_mb << " MB\n";
    std::cout << "  Num clusters:      " << num_reps << "\n";
    std::cout << "  Vector dim:        " << dim << "\n";
    std::cout << "  Est. in-memory:    " << std::setprecision(1) << in_memory_mb
              << " MB (rep vectors + HNSW)\n";
    std::cout << "  Process RSS:       " << rss_mb << " MB\n";
    std::cout << std::string(72, '=') << "\n";

    std::cout << "\nDone!\n";
    return 0;
}

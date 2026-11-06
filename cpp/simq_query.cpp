// SIMQ query and evaluation.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iomanip>
#include <map>
#include <mutex>
#include <functional>
#include <queue>
#include <unistd.h>
#include <libaio.h>

#include "npy_reader.h"
#include "../hnswlib/hnswlib/hnswlib.h"

// Direct I/O aligned-read helper
static constexpr int64_t DIO_ALIGN = 4096;

struct ThreadAioContext {
    io_context_t ctx = 0;
    ThreadAioContext() {
        if (io_setup(4096, &ctx) != 0)
            throw std::runtime_error("io_setup failed");
    }
    ~ThreadAioContext() { if (ctx) io_destroy(ctx); }
};

static void parallel_for_queries(int count, int query_threads,
                                 const std::function<void(int)>& fn) {
    const int nth = std::max(1, std::min(query_threads, count));
    if (nth == 1) {
        for (int i = 0; i < count; ++i) fn(i);
        return;
    }
    std::atomic<int> next{0};
    std::vector<std::thread> workers;
    workers.reserve(nth);
    for (int t = 0; t < nth; ++t) {
        workers.emplace_back([&] {
            for (;;) {
                int i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= count) break;
                fn(i);
            }
        });
    }
    for (auto& worker : workers) worker.join();
}

// Read data_size bytes from fd at byte offset into dst.
// Uses O_DIRECT-compatible aligned intermediate buffer.
static bool dio_pread(int fd, void* dst, size_t data_size, int64_t offset) {
    int64_t aligned_off = (offset / DIO_ALIGN) * DIO_ALIGN;
    size_t  prefix      = (size_t)(offset - aligned_off);
    size_t  read_size   = ((prefix + data_size + DIO_ALIGN - 1) / DIO_ALIGN) * DIO_ALIGN;
    void*   buf         = nullptr;
    if (::posix_memalign(&buf, (size_t)DIO_ALIGN, read_size) != 0) return false;
    ssize_t got = ::pread(fd, buf, read_size, aligned_off);
    bool ok = (got >= 0 && (size_t)got >= prefix + data_size);
    if (ok) std::memcpy(dst, static_cast<char*>(buf) + prefix, data_size);
    std::free(buf);
    return ok;
}

struct RetrievalResult {
    std::string doc_id;
    double      coarse_score;
    double      final_score;
};

// parse_npy_header
static std::tuple<size_t, size_t, size_t>
parse_npy_header(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open npy: " + path);

    char magic[6];
    f.read(magic, 6);
    if (std::string(magic + 1, 5) != "NUMPY")
        throw std::runtime_error("Not a .npy file: " + path);

    uint8_t major, minor;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t header_len = 0; int len_bytes;
    if (major == 1) {
        uint16_t hl16; f.read(reinterpret_cast<char*>(&hl16), 2);
        header_len = hl16; len_bytes = 2;
    } else {
        f.read(reinterpret_cast<char*>(&header_len), 4); len_bytes = 4;
    }
    std::string header(header_len, '\0');
    f.read(&header[0], header_len);
    size_t data_offset = 6 + 2 + len_bytes + header_len;

    auto pos = header.find("'shape'");
    if (pos == std::string::npos)
        throw std::runtime_error("No shape in npy header: " + path);
    auto lp = header.find('(', pos), rp = header.find(')', lp);
    std::string shape_str = header.substr(lp + 1, rp - lp - 1);
    auto comma = shape_str.find(',');
    size_t rows = std::stoull(shape_str.substr(0, comma));
    std::string rest = shape_str.substr(comma + 1);
    while (!rest.empty() && (rest.front() == ' ' || rest.front() == ',')) rest.erase(rest.begin());
    while (!rest.empty() && (rest.back()  == ' ' || rest.back()  == ',')) rest.pop_back();
    size_t cols = rest.empty() ? 1 : std::stoull(rest);
    return {data_offset, rows, cols};
}

// ClusterDiskRetrievalV2
class ClusterDiskRetrievalV2 {
public:
    ClusterDiskRetrievalV2(const std::string& emb_dir,
                           const std::string& cluster_dir,
                           const std::string& reorder_dir,
                           const std::string& qrels_path,
                           int m, int ef, int coarse_k, int final_k,
                           int query_threads,
                           bool in_memory = false,
                           bool use_cagra = false)
        : emb_dir_(emb_dir), cluster_dir_(cluster_dir),
          reorder_dir_(reorder_dir), qrels_path_(qrels_path),
          m_(m), ef_(ef), coarse_k_(coarse_k), final_k_(final_k),
          query_threads_(std::max(1, query_threads)),
          in_memory_(in_memory), use_cagra_(use_cagra) {
        load_npy_info();
        load_clustering();
        load_query_vecs();
        load_reorder_data();
        load_metadata();
        build_vid_to_doc_local();
        if (use_cagra_) load_cagra_index();
        else load_representative_hnsw_index();
    }

    ~ClusterDiskRetrievalV2() {
        if (emb_mmap_ && emb_mmap_ != MAP_FAILED)
            munmap(emb_mmap_, emb_mmap_size_);
        if (emb_fd_ >= 0) ::close(emb_fd_);
        delete chnsw_; delete cspace_;
    }

    void retrieve_query(int query_index, bool use_rerank = true);
    void evaluate(bool use_rerank = true);
    void evaluate_both();
    void eval_lazy();
    void eval_m_10_100();
    void eval_lazy_LoTTE();
    void print_info() const;

    void set_hnsw_ef(int ef) {
        ef_ = ef;
        if (chnsw_) { chnsw_->setEf(ef); std::cout << "HNSW ef=" << ef << "\n"; }
    }
    void set_m(int m) { m_ = m; std::cout << "m set to " << m_ << "\n"; }

private:
    void load_npy_info();
    void load_clustering();
    void load_query_vecs();
    void load_reorder_data();
    void load_metadata();
    void build_vid_to_doc_local();
    void load_representative_hnsw_index();
    void load_cagra_index();
    std::vector<std::pair<float, int>> search_cagra(const float* query, int top_k) const;

    std::vector<const float*> get_query_vecs(int qi) const;

    std::vector<std::pair<int, double>>
    coarse_search(const std::vector<const float*>& q_ptrs,
                  double& hnsw_ms, double& scoring_ms) const;

    std::vector<RetrievalResult>
    rerank_maxsim_disk(const std::vector<const float*>& q_ptrs,
                       const std::vector<std::pair<int, double>>& coarse_top,
                       double& pread_ms, double& compute_ms) const;

    std::vector<RetrievalResult>
    search_query(int qi, double& hnsw_ms, double& scoring_ms,
                 double& rerank_ms, double& pread_ms, double& compute_ms,
                 bool use_rerank = true) const;

    std::map<int, double> compute_precision_at_k(
        const std::vector<std::vector<RetrievalResult>>&,
        const std::vector<int>&) const;
    std::map<int, double> compute_recall_k_at_100(
        const std::vector<std::vector<RetrievalResult>>&,
        const std::vector<int>&) const;
    std::map<int, double> compute_recall_k_at_k(
        const std::vector<std::vector<RetrievalResult>>&,
        const std::vector<int>&) const;

    std::string emb_dir_, cluster_dir_, reorder_dir_, qrels_path_;
    int m_, ef_, coarse_k_, final_k_, query_threads_;

    int    dim_             = 0;
    size_t npy_data_offset_ = 0;
    int    doc_vec_start_   = 0;
    int    doc_vec_end_     = 0;
    int    query_vec_start_ = 0;
    int    query_vec_end_   = 0;
    std::vector<int> query_lengths_;

    std::vector<float> query_vecs_;
    int n_query_vecs_ = 0;

    std::vector<std::vector<int>> cluster_members_;  // indexed by cluster id
    std::unordered_map<int, int>              cluster_representatives_;

    int emb_fd_ = -1;
    float* emb_mmap_      = nullptr;     
    size_t emb_mmap_size_ = 0;
    bool   in_memory_     = false;
    bool   use_cagra_     = false;
    std::vector<int64_t> doc_offsets_;
    std::vector<int32_t> doc_token_counts_;
    std::unordered_map<std::string, int> doc_id_to_local_;

    std::unordered_map<int, VectorDocInfo> vec_to_doc_;
    std::vector<std::string> query_ids_;
    std::unordered_map<std::string, std::vector<std::string>> qrels_;
    std::vector<int> qrel_qindices_;   // indices into query_ids_ that have qrels

    std::vector<int>         vid_to_doc_local_;  // local_vid → doc local idx (-1=non-doc)
    std::vector<std::string> local_to_doc_id_;   // doc local idx → doc_id string
    hnswlib::InnerProductSpace*      cspace_ = nullptr;
    hnswlib::HierarchicalNSW<float>* chnsw_  = nullptr;
    std::vector<float> cagra_centers_;
    std::vector<int32_t> cagra_graph_;
    int cagra_n_ = 0;
    int cagra_degree_ = 0;
    int cagra_entry_ = 0;
};

void ClusterDiskRetrievalV2::load_npy_info() {
    std::cout << "[1/6] Parsing npy header + meta JSON...\n";
    auto [data_off, rows, cols] = parse_npy_header(emb_dir_ + "/colbert_embeddings.npy");
    npy_data_offset_ = data_off; dim_ = (int)cols;
    std::cout << "  npy shape: " << rows << " x " << cols << "\n";
    std::string meta = read_file_text(emb_dir_ + "/colbert_embeddings_meta.json");
    doc_vec_start_   = (int)json_int(meta, "doc_vector_start");
    doc_vec_end_     = (int)json_int(meta, "doc_vector_end");
    query_vec_start_ = (int)json_int(meta, "query_vector_start");
    query_vec_end_   = (int)json_int(meta, "query_vector_end");
    query_lengths_   = json_int_array(meta, "query_lengths");
    std::cout << "  Doc vecs: [" << doc_vec_start_ << "," << doc_vec_end_ << ")"
              << "  Query vecs: [" << query_vec_start_ << "," << query_vec_end_ << ")\n";
}

void ClusterDiskRetrievalV2::load_clustering() {
    std::cout << "[2/6] Loading clustering...\n";
    auto tmp_members = load_cluster_members_json(cluster_dir_ + "/cluster_members.json");
    // Convert unordered_map → vector<vector> for direct array access (no hashing)
    int max_cidx = -1;
    for (auto& [k, _] : tmp_members) if (k > max_cidx) max_cidx = k;
    cluster_members_.assign(max_cidx + 1, {});
    for (auto& [k, v] : tmp_members) cluster_members_[k] = std::move(v);
    std::cout << "  Clusters: " << tmp_members.size() << "\n";
    std::ifstream rf(cluster_dir_ + "/cluster_representatives.txt");
    if (!rf) throw std::runtime_error("Cannot open cluster_representatives.txt");
    std::string line;
    while (std::getline(rf, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line); int cidx, vid;
        if (iss >> cidx >> vid) cluster_representatives_[cidx] = vid;
    }
    std::cout << "  Representatives: " << cluster_representatives_.size() << "\n";
}

void ClusterDiskRetrievalV2::load_query_vecs() {
    std::cout << "[3/6] Loading query vectors...\n";
    n_query_vecs_ = query_vec_end_ - query_vec_start_;
    query_vecs_.resize((size_t)n_query_vecs_ * dim_);
    int fd = ::open((emb_dir_ + "/colbert_embeddings.npy").c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("Cannot open npy for query read");
    int64_t off = (int64_t)npy_data_offset_ + (int64_t)query_vec_start_ * dim_ * sizeof(float);
    size_t nb = (size_t)n_query_vecs_ * dim_ * sizeof(float);
    ssize_t got = ::pread(fd, query_vecs_.data(), nb, off);
    ::close(fd);
    if (got != (ssize_t)nb) throw std::runtime_error("Partial read query vecs");
    std::cout << "  Loaded " << n_query_vecs_ << " query token vectors\n";
}

void ClusterDiskRetrievalV2::load_reorder_data() {
    std::cout << "[4/6] Loading reorder data...\n";
    std::string mt = read_file_text(reorder_dir_ + "/reorder_meta.json");
    int n_docs = (int)json_int(mt, "n_docs");
    std::cout << "  n_docs=" << n_docs << "\n";
    { std::ifstream f(reorder_dir_ + "/doc_offsets.bin", std::ios::binary);
      if (!f) throw std::runtime_error("Cannot open doc_offsets.bin");
      doc_offsets_.resize(n_docs + 1);
      f.read(reinterpret_cast<char*>(doc_offsets_.data()), (n_docs+1)*sizeof(int64_t));
      if (!f) throw std::runtime_error("Truncated doc_offsets.bin"); }
    { std::ifstream f(reorder_dir_ + "/doc_token_counts.bin", std::ios::binary);
      if (!f) throw std::runtime_error("Cannot open doc_token_counts.bin");
      doc_token_counts_.resize(n_docs);
      f.read(reinterpret_cast<char*>(doc_token_counts_.data()), n_docs*sizeof(int32_t));
      if (!f) throw std::runtime_error("Truncated doc_token_counts.bin"); }
    { std::ifstream f(reorder_dir_ + "/doc_ids.txt");
      if (!f) throw std::runtime_error("Cannot open doc_ids.txt");
      std::string line; int idx = 0;
      while (std::getline(f, line)) if (!line.empty()) doc_id_to_local_[line] = idx++;
      if (idx != n_docs) std::cerr << "  Warning: doc_ids mismatch\n"; }
    emb_fd_ = ::open((reorder_dir_ + "/doc_embeddings_docorder.bin").c_str(),
                     O_RDONLY | (in_memory_ ? 0 : O_DIRECT));
    if (emb_fd_ < 0) throw std::runtime_error("Cannot open doc_embeddings_docorder.bin");

    if (in_memory_) {
        struct stat st;
        if (::fstat(emb_fd_, &st) != 0)
            throw std::runtime_error("fstat failed on reorder file");
        emb_mmap_size_ = (size_t)st.st_size;
        emb_mmap_ = reinterpret_cast<float*>(
            ::mmap(nullptr, emb_mmap_size_, PROT_READ, MAP_PRIVATE | MAP_POPULATE,
                   emb_fd_, 0));
        if (emb_mmap_ == MAP_FAILED)
            throw std::runtime_error("mmap failed for reorder file");
        ::madvise(emb_mmap_, emb_mmap_size_, MADV_RANDOM);
        std::cout << "  [in-memory] mmap 完成: "
                  << emb_mmap_size_ / 1024.0 / 1024 / 1024
                  << " GB (MAP_POPULATE 预热中...)\n";
        volatile char sink = 0;
        for (size_t i = 0; i < emb_mmap_size_; i += 4096)
            sink += reinterpret_cast<volatile char*>(emb_mmap_)[i];
        (void)sink;
        std::cout << "  [in-memory] 预热完成\n";
    }
    std::cout << "  " << doc_id_to_local_.size() << " docs\n";
}

void ClusterDiskRetrievalV2::load_metadata() {
    std::cout << "[5/6] Loading metadata...\n";
    if (std::ifstream probe(emb_dir_ + "/vector_to_doc_mapping.txt"); probe)
        vec_to_doc_ = load_vector_to_doc_mapping(emb_dir_ + "/vector_to_doc_mapping.txt");
    else
        std::cout << "  vector_to_doc_mapping.txt 不存在，将从 doc_lengths 直接计算映射\n";
    query_ids_  = load_query_ids(emb_dir_ + "/query_ids.txt");
    std::ifstream f(qrels_path_);
    if (!f) throw std::runtime_error("Cannot open qrels: " + qrels_path_);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line.find("query-id") != std::string::npos ||
            line.find("query_id") != std::string::npos) continue;
        std::istringstream iss(line); std::string qid, did; float score;
        if (iss >> qid >> did >> score && score > 0) qrels_[qid].push_back(did);
    }
    for (int qi = 0; qi < (int)query_ids_.size(); qi++)
        if (qrels_.count(query_ids_[qi])) qrel_qindices_.push_back(qi);
    std::cout << "  Queries: " << query_ids_.size()
              << "  Qrels: " << qrels_.size()
              << "  Eval subset: " << qrel_qindices_.size() << "\n";
}

void ClusterDiskRetrievalV2::build_vid_to_doc_local() {
    std::cout << "[6/6] Building integer vid→doc mapping...\n";
    int n_local = doc_vec_end_ - doc_vec_start_;
    int n_docs  = (int)doc_id_to_local_.size();
    vid_to_doc_local_.assign(n_local, -1);
    local_to_doc_id_.resize(n_docs);
    for (auto& [doc_id, li] : doc_id_to_local_) local_to_doc_id_[li] = doc_id;
    if (!vec_to_doc_.empty()) {
        for (auto& [gvid, info] : vec_to_doc_) {
            if (info.doc_type != "doc") continue;
            int lv = gvid - doc_vec_start_;
            if (lv < 0 || lv >= n_local) continue;
            auto it = doc_id_to_local_.find(info.doc_id);
            if (it != doc_id_to_local_.end()) vid_to_doc_local_[lv] = it->second;
        }
    } else {
        std::string meta_text = read_file_text(emb_dir_ + "/colbert_embeddings_meta.json");
        std::vector<int> doc_lengths = json_int_array(meta_text, "doc_lengths");
        int lv = 0;
        for (int di = 0; di < n_docs && lv < n_local; di++)
            for (int t = 0; t < doc_lengths[di] && lv < n_local; t++, lv++)
                vid_to_doc_local_[lv] = di;
        std::cout << "  (fallback from doc_lengths)\n";
    }
    std::cout << "  Mapped " << n_local << " vecs → " << n_docs << " docs\n";
}

void ClusterDiskRetrievalV2::load_representative_hnsw_index() {
    std::cout << "Loading HNSW index...\n";
    cspace_ = new hnswlib::InnerProductSpace(dim_);
    chnsw_  = new hnswlib::HierarchicalNSW<float>(
                  cspace_, cluster_dir_ + "/centroid_hnsw_index.hnsw");
    chnsw_->setEf(ef_);

    size_t rss_kb = get_rss_kb();
    double rss_mb = rss_kb / 1024.0;

    std::cout << "  HNSW loaded (ef=" << ef_ << ")\n"
              << "\n✓ Ready. m=" << m_ << " ef=" << ef_
              << " coarse_k=" << coarse_k_ << " final_k=" << final_k_
              << " query_threads=" << query_threads_ << "\n";
    std::cout << "[Memory after load] " << std::fixed << std::setprecision(1)
              << rss_mb << " MB (RSS)\n\n";
}

// get_query_vecs
void ClusterDiskRetrievalV2::load_cagra_index() {
    std::ifstream cf(cluster_dir_ + "/cagra_centers.bin", std::ios::binary);
    if (!cf) throw std::runtime_error("Cannot open CAGRA center table");
    int32_t n = 0, d = 0;
    cf.read(reinterpret_cast<char*>(&n), 4); cf.read(reinterpret_cast<char*>(&d), 4);
    if (!cf || n <= 0 || d != dim_) throw std::runtime_error("Invalid CAGRA center table header");
    if (n != (int)cluster_members_.size()) throw std::runtime_error("CAGRA graph and cluster counts differ");
    cagra_n_ = n; cagra_centers_.resize((size_t)n * d);
    cf.read(reinterpret_cast<char*>(cagra_centers_.data()), (std::streamsize)cagra_centers_.size() * sizeof(float));
    if (!cf) throw std::runtime_error("Truncated CAGRA center table");
    std::ifstream gf(cluster_dir_ + "/cagra_centers.nsg", std::ios::binary);
    if (!gf) throw std::runtime_error("Cannot open CAGRA graph");
    uint32_t degree = 0, entry = 0;
    gf.read(reinterpret_cast<char*>(&degree), 4); gf.read(reinterpret_cast<char*>(&entry), 4);
    if (!gf || degree == 0 || entry >= (uint32_t)n) throw std::runtime_error("Invalid CAGRA graph header");
    cagra_degree_ = (int)degree; cagra_entry_ = (int)entry;
    cagra_graph_.assign((size_t)n * degree, -1);
    for (int i = 0; i < n; ++i) {
        uint32_t count = 0; gf.read(reinterpret_cast<char*>(&count), 4);
        if (!gf || count > degree) throw std::runtime_error("Invalid CAGRA graph row");
        gf.read(reinterpret_cast<char*>(cagra_graph_.data() + (size_t)i * degree), count * sizeof(int32_t));
        if (!gf) throw std::runtime_error("Truncated CAGRA graph row");
    }
    std::cout << "  CAGRA loaded: centers=" << n << " degree=" << degree << " ef=" << ef_ << "\n";
}

std::vector<std::pair<float, int>>
ClusterDiskRetrievalV2::search_cagra(const float* query, int top_k) const {
    using Item = std::pair<float, int>;
    const int beam = std::max({1, ef_, top_k});
    std::vector<uint8_t> visited((size_t)cagra_n_, 0);
    std::priority_queue<Item> frontier;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> best;
    auto score = [&](int id) {
        float s = 0.0f; const float* v = cagra_centers_.data() + (size_t)id * dim_;
        for (int d = 0; d < dim_; ++d) s += query[d] * v[d];
        return s;
    };
    float first = score(cagra_entry_);
    frontier.push({first, cagra_entry_}); best.push({first, cagra_entry_}); visited[cagra_entry_] = 1;
    while (!frontier.empty()) {
        Item cur = frontier.top();
        if ((int)best.size() >= beam && cur.first < best.top().first) break;
        frontier.pop();
        const int32_t* edges = cagra_graph_.data() + (size_t)cur.second * cagra_degree_;
        for (int j = 0; j < cagra_degree_; ++j) {
            int id = edges[j];
            if (id < 0 || id >= cagra_n_ || visited[id]) continue;
            visited[id] = 1; float s = score(id);
            if ((int)best.size() < beam || s > best.top().first) {
                frontier.push({s, id}); best.push({s, id});
                if ((int)best.size() > beam) best.pop();
            }
        }
    }
    std::vector<Item> out; out.reserve(std::min(top_k, (int)best.size()));
    while (!best.empty()) { out.push_back(best.top()); best.pop(); }
    std::sort(out.begin(), out.end(), [](const Item& a, const Item& b) { return a.first > b.first; });
    if ((int)out.size() > top_k) out.resize(top_k);
    return out;
}

std::vector<const float*>
ClusterDiskRetrievalV2::get_query_vecs(int qi) const {
    int off = 0;
    for (int j = 0; j < qi; j++) off += query_lengths_[j];
    std::vector<const float*> ptrs;
    ptrs.reserve(query_lengths_[qi]);
    for (int v = 0; v < query_lengths_[qi]; v++)
        ptrs.push_back(query_vecs_.data() + (size_t)(off + v) * dim_);
    return ptrs;
}

// coarse_search  —  two optimisations vs cluster_eval_disk:
//
//   (a) cscore = 1.0f - result.top().first
//       InnerProductSpace stores dist = 1 - dot, so we reuse the value
//       HNSW already computed.  rep_vecs_ and the dim-loop are gone.
//
//   (b) Integer arrays replace string maps in the scoring hot path.
//       vid_to_doc_local_[mvid] → int di  (array lookup, ~2 ns)
//       score_buf_[di] += cscore           (array write,  ~2 ns)
//       seen_buf_[di]                      (array read,   ~2 ns)
//       vs unordered_map<string,…> ops     (~50-100 ns each)
std::vector<std::pair<int, double>>
ClusterDiskRetrievalV2::coarse_search(
        const std::vector<const float*>& q_ptrs,
        double& hnsw_ms, double& scoring_ms) const {
    int nq = (int)q_ptrs.size();
    hnsw_ms = 0.0; scoring_ms = 0.0;
    const int n_docs = (int)local_to_doc_id_.size();
    thread_local std::vector<float> score_buf;
    thread_local std::vector<uint8_t> seen_buf;
    if ((int)score_buf.size() != n_docs) score_buf.assign(n_docs, 0.0f);
    if ((int)seen_buf.size() != n_docs) seen_buf.assign(n_docs, 0);
    std::vector<int> dirty;
    std::vector<int> seen_dirty;
    dirty.reserve(std::min(n_docs, coarse_k_ * 4));
    seen_dirty.reserve(std::min(n_docs, coarse_k_ * 4));

    for (int ti = 0; ti < nq; ti++) {
        auto th0 = std::chrono::steady_clock::now();
        std::vector<std::pair<float, int>> cscores;
        decltype(chnsw_->searchKnn(q_ptrs[ti], m_)) result;
        if (use_cagra_) cscores = search_cagra(q_ptrs[ti], m_);
        else result = chnsw_->searchKnn(q_ptrs[ti], m_);
        hnsw_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - th0).count();

        auto ts0 = std::chrono::steady_clock::now();
        if (!use_cagra_) {
        cscores.reserve(result.size());
        while (!result.empty()) {
            // InnerProductSpace: dist = 1 - dot  →  dot = 1 - dist
            float cscore = 1.0f - result.top().first;
            cscores.push_back({cscore, (int)result.top().second});
            result.pop();
        }
        std::sort(cscores.begin(), cscores.end(),
                  [](const auto& a, const auto& b){ return a.first > b.first; });
        }

        seen_dirty.clear();
        for (auto& [cscore, cidx] : cscores) {
            if (cidx < 0 || cidx >= (int)cluster_members_.size()) continue;
            for (int mvid : cluster_members_[cidx]) {
                if (mvid >= (int)vid_to_doc_local_.size()) continue;
                int di = vid_to_doc_local_[mvid];
                if (di < 0 || seen_buf[di]) continue;  // non-doc or already scored this token
                seen_buf[di] = 1;
                seen_dirty.push_back(di);
                if (score_buf[di] == 0.0f) dirty.push_back(di);
                score_buf[di] += cscore;
            }
        }
        // reset per-token seen flags
        for (int di : seen_dirty) seen_buf[di] = 0;

        scoring_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - ts0).count();
    }

    std::vector<std::pair<int, double>> ranked;
    ranked.reserve(dirty.size());
    for (int di : dirty) {
        ranked.push_back({di, score_buf[di]});
        score_buf[di] = 0.0f;  // reset for next query
    }

    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b){ return a.second > b.second; });
    return ranked;
}

// rerank_maxsim_disk  (libaio async IO — v4)
std::vector<RetrievalResult>
ClusterDiskRetrievalV2::rerank_maxsim_disk(
        const std::vector<const float*>& q_ptrs,
        const std::vector<std::pair<int, double>>& coarse_top,
        double& pread_ms, double& compute_ms) const {
    int nq = (int)q_ptrs.size();
    int n_total_docs = (int)doc_offsets_.size() - 1;
    struct DocFetch {
        int li; double coarse_score;
        int64_t byte_offset; int32_t token_count;
    };
    std::vector<DocFetch> fetches;
    fetches.reserve(coarse_top.size());
    for (auto& [li, cs] : coarse_top) {
        if (li < 0 || li >= n_total_docs) continue;
        fetches.push_back({li, cs, doc_offsets_[li], doc_token_counts_[li]});
    }
    if (fetches.empty()) { pread_ms = 0.0; compute_ms = 0.0; return {}; }

    std::sort(fetches.begin(), fetches.end(),
              [](const DocFetch& a, const DocFetch& b){ return a.byte_offset < b.byte_offset; });

    int n_docs = (int)fetches.size();

    if (in_memory_ && emb_mmap_ && emb_mmap_ != MAP_FAILED) {
        auto tp0 = std::chrono::steady_clock::now();
        std::vector<double> scores(n_docs, -1e30);
        std::atomic<long long> compute_ns{0};

        auto worker = [&](int s, int e) {
            long long lc = 0;
            for (int i = s; i < e; i++) {
                auto tc0 = std::chrono::steady_clock::now();
                int tc = fetches[i].token_count;
                const float* data = reinterpret_cast<const float*>(
                    reinterpret_cast<const char*>(emb_mmap_) + fetches[i].byte_offset);
                double score = 0.0;
                for (int qi = 0; qi < nq; qi++) {
                    const float* qv = q_ptrs[qi]; float mx = -1e9f;
                    for (int dv = 0; dv < tc; dv++) {
                        const float* dp = data + (size_t)dv * dim_;
                        float dot = 0.0f;
                        for (int k = 0; k < dim_; k++) dot += qv[k] * dp[k];
                        if (dot > mx) mx = dot;
                    }
                    score += mx;
                }
                scores[i] = score;
                lc += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - tc0).count();
            }
            compute_ns += lc;
        };
        worker(0, n_docs);
        pread_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - tp0).count();
        compute_ms = compute_ns.load() / 1e6;

        std::vector<RetrievalResult> res; res.reserve(n_docs);
        for (int i = 0; i < n_docs; i++)
            if (scores[i] > -1e29)
                res.push_back({local_to_doc_id_[fetches[i].li], fetches[i].coarse_score, scores[i]});
        std::sort(res.begin(), res.end(),
                  [](const auto& a, const auto& b){ return a.final_score > b.final_score; });
        return res;
    }

    // Per-doc aligned buffers and iocbs
    struct AioBuf {
        void*  raw;       // aligned allocation
        size_t prefix;    // bytes before actual data within raw
        size_t data_size; // actual embedding bytes needed
    };
    std::vector<AioBuf>        abufs(n_docs);
    std::vector<struct iocb>   cbs(n_docs);
    std::vector<struct iocb*>  cbptrs(n_docs);

    for (int i = 0; i < n_docs; i++) {
        auto& f = fetches[i];
        size_t data_size    = (size_t)f.token_count * dim_ * sizeof(float);
        int64_t aligned_off = (f.byte_offset / DIO_ALIGN) * DIO_ALIGN;
        size_t  prefix      = (size_t)(f.byte_offset - aligned_off);
        size_t  read_size   = ((prefix + data_size + DIO_ALIGN - 1) / DIO_ALIGN) * DIO_ALIGN;

        void* raw = nullptr;
        if (::posix_memalign(&raw, (size_t)DIO_ALIGN, read_size) != 0) {
            // allocation failure: mark as skip
            abufs[i] = {nullptr, 0, data_size};
            io_prep_pread(&cbs[i], emb_fd_, nullptr, 0, 0); // dummy
        } else {
            abufs[i] = {raw, prefix, data_size};
            io_prep_pread(&cbs[i], emb_fd_, raw, read_size, aligned_off);
        }
        cbs[i].data = (void*)(uintptr_t)i;
        cbptrs[i]   = &cbs[i];
    }

    // Each query worker owns its AIO context, so completions cannot be mixed.
    thread_local ThreadAioContext thread_aio;
    auto tp0 = std::chrono::steady_clock::now();
    std::vector<struct iocb*> valid_cbs;
    valid_cbs.reserve(n_docs);
    for (int i = 0; i < n_docs; i++)
        if (abufs[i].raw) valid_cbs.push_back(cbptrs[i]);
    int n_submit = (int)valid_cbs.size();
    if (n_submit > 0)
        io_submit(thread_aio.ctx, (long)n_submit, valid_cbs.data());

    // Wait for all completions
    std::vector<struct io_event> events(n_submit);
    if (n_submit > 0)
        io_getevents(thread_aio.ctx, n_submit, n_submit, events.data(), nullptr);
    pread_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - tp0).count();

    std::vector<double> scores(n_docs, -1e30);
    std::atomic<long long> compute_ns{0};

    auto worker = [&](int s, int e) {
        long long lc = 0;
        for (int i = s; i < e; i++) {
            if (!abufs[i].raw) continue;
            auto tc0 = std::chrono::steady_clock::now();
            int tc = fetches[i].token_count;
            const float* data = static_cast<const float*>(abufs[i].raw) +
                                 abufs[i].prefix / sizeof(float);
            double score = 0.0;
            for (int qi = 0; qi < nq; qi++) {
                const float* qv = q_ptrs[qi]; float mx = -1e9f;
                for (int dv = 0; dv < tc; dv++) {
                    const float* dp = data + (size_t)dv * dim_;
                    float dot = 0.0f;
                    for (int k = 0; k < dim_; k++) dot += qv[k] * dp[k];
                    if (dot > mx) mx = dot;
                }
                score += mx;
            }
            scores[i] = score;
            lc += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - tc0).count();
        }
        compute_ns += lc;
    };

    // Single-threaded compute: IO parallelism comes from async IO (hardware),
    // not CPU threads, so baselines remain comparable on equal footing.
    worker(0, n_docs);
    compute_ms = compute_ns.load() / 1e6;

    // Cleanup aligned buffers (the worker-local AIO context is reused).
    for (auto& ab : abufs) if (ab.raw) std::free(ab.raw);

    std::vector<RetrievalResult> res; res.reserve(n_docs);
    for (int i = 0; i < n_docs; i++)
        if (scores[i] > -1e29)
            res.push_back({local_to_doc_id_[fetches[i].li], fetches[i].coarse_score, scores[i]});
    std::sort(res.begin(), res.end(),
              [](const auto& a, const auto& b){ return a.final_score > b.final_score; });
    return res;
}

std::vector<RetrievalResult>
ClusterDiskRetrievalV2::search_query(
        int qi, double& hnsw_ms, double& scoring_ms,
        double& rerank_ms, double& pread_ms, double& compute_ms,
        bool use_rerank) const {
    auto q_ptrs = get_query_vecs(qi);
    auto coarse = coarse_search(q_ptrs, hnsw_ms, scoring_ms);

    int trunc = use_rerank ? coarse_k_ : final_k_;
    if ((int)coarse.size() > trunc) coarse.resize(trunc);

    if (use_rerank) {
        auto t2 = std::chrono::steady_clock::now();
        auto res = rerank_maxsim_disk(q_ptrs, coarse, pread_ms, compute_ms);
        rerank_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t2).count();
        if ((int)res.size() > final_k_) res.resize(final_k_);
        return res;
    } else {
        rerank_ms = 0.0; pread_ms = 0.0; compute_ms = 0.0;
        std::vector<RetrievalResult> res; res.reserve(coarse.size());
        for (auto& [li, cs] : coarse) res.push_back({local_to_doc_id_[li], cs, cs});
        return res;
    }
}

std::map<int, double> ClusterDiskRetrievalV2::compute_precision_at_k(
        const std::vector<std::vector<RetrievalResult>>& all_results,
        const std::vector<int>& k_values) const {
    std::map<int, double> out;
    int nq = (int)qrel_qindices_.size();
    for (int k : k_values) {
        double total = 0.0;
        for (int li = 0; li < nq; li++) {
            int qi = qrel_qindices_[li];
            auto qit = qrels_.find(query_ids_[qi]);
            if (qit == qrels_.end() || qit->second.empty()) continue;
            std::unordered_set<std::string> gt;
            for (int i = 0; i < std::min(k, (int)qit->second.size()); i++) gt.insert(qit->second[i]);
            int hits = 0;
            for (int i = 0; i < std::min(k, (int)all_results[li].size()); i++)
                if (gt.count(all_results[li][i].doc_id)) hits++;
            total += (double)hits / k;
        }
        out[k] = nq > 0 ? total / nq : 0.0;
    }
    return out;
}

std::map<int, double> ClusterDiskRetrievalV2::compute_recall_k_at_100(
        const std::vector<std::vector<RetrievalResult>>& all_results,
        const std::vector<int>& k_values) const {
    std::map<int, double> out;
    int nq = (int)qrel_qindices_.size();
    for (int k : k_values) {
        double total = 0.0;
        for (int li = 0; li < nq; li++) {
            int qi = qrel_qindices_[li];
            auto qit = qrels_.find(query_ids_[qi]);
            if (qit == qrels_.end() || qit->second.empty()) continue;
            std::unordered_set<std::string> gt100;
            for (int i = 0; i < std::min(100, (int)qit->second.size()); i++) gt100.insert(qit->second[i]);
            int hits = 0;
            for (int i = 0; i < std::min(k, (int)all_results[li].size()); i++)
                if (gt100.count(all_results[li][i].doc_id)) hits++;
            total += (double)hits / k;
        }
        out[k] = nq > 0 ? total / nq : 0.0;
    }
    return out;
}

// R10@10, R20@20, R50@50, R100@100
std::map<int, double> ClusterDiskRetrievalV2::compute_recall_k_at_k(
        const std::vector<std::vector<RetrievalResult>>& all_results,
        const std::vector<int>& k_values) const {
    std::map<int, double> out;
    int nq = (int)qrel_qindices_.size();
    for (int k : k_values) {
        double total = 0.0;
        int valid_q = 0;
        for (int li = 0; li < nq; li++) {
            int qi = qrel_qindices_[li];
            auto qit = qrels_.find(query_ids_[qi]);
            if (qit == qrels_.end() || qit->second.empty()) continue;
            valid_q++;
            // Ground truth = top-k from qrels
            int gt_lim = std::min(k, (int)qit->second.size());
            std::unordered_set<std::string> gt_k;
            for (int i = 0; i < gt_lim; i++) gt_k.insert(qit->second[i]);
            // Count hits in top-k results
            int hits = 0;
            int res_lim = std::min(k, (int)all_results[li].size());
            for (int i = 0; i < res_lim; i++)
                if (gt_k.count(all_results[li][i].doc_id)) hits++;
            total += (double)hits / k;
        }
        out[k] = valid_q > 0 ? total / valid_q : 0.0;
    }
    return out;
}

void ClusterDiskRetrievalV2::retrieve_query(int qi, bool use_rerank) {
    if (qi < 0 || qi >= (int)qrel_qindices_.size()) {
        std::cerr << "query index out of range (0.." << qrel_qindices_.size()-1 << ")\n"; return;
    }
    qi = qrel_qindices_[qi];  // remap: user-facing index → global query index
    const std::string& qid = query_ids_[qi];
    std::cout << "\n" << std::string(80,'=') << "\nQuery [" << qi << "]: " << qid
              << "\nMode: " << (use_rerank ? "Coarse+DiskRerank" : "Coarse Only")
              << "\n" << std::string(80,'=') << "\n";
    auto q_ptrs = get_query_vecs(qi);
    auto qit = qrels_.find(qid);
    const std::vector<std::string>* gt = (qit != qrels_.end()) ? &qit->second : nullptr;
    if (gt) std::cout << "Ground truth: " << gt->size() << " docs\n";

    double hnsw_ms, scoring_ms, rerank_ms, pread_ms, compute_ms;
    auto res = search_query(qi, hnsw_ms, scoring_ms, rerank_ms, pread_ms, compute_ms, use_rerank);

    std::cout << "\nTop-10:\n" << std::string(80,'-') << "\n";
    for (int i = 0; i < std::min(10, (int)res.size()); i++) {
        bool in_gt = gt && std::find(gt->begin(), gt->end(), res[i].doc_id) != gt->end();
        std::cout << "  " << std::setw(2) << (i+1) << ". [" << (in_gt?"✓":" ") << "] "
                  << std::setw(10) << res[i].doc_id
                  << "  score=" << std::fixed << std::setprecision(4) << res[i].final_score << "\n";
    }
    double tot = hnsw_ms + scoring_ms + rerank_ms;
    std::cout << "\nCenterSearch=" << std::setprecision(3) << hnsw_ms
              << "ms  Scoring=" << scoring_ms << "ms";
    if (use_rerank) std::cout << "  Rerank=" << rerank_ms
                              << "ms  [pread=" << pread_ms << "ms compute=" << compute_ms << "ms]";
    std::cout << "  Total=" << tot << "ms  QPS=" << std::setprecision(1) << (1000.0/tot)
              << "\n" << std::string(80,'=') << "\n\n";
}

void ClusterDiskRetrievalV2::evaluate(bool use_rerank) {
    const auto& qidxs = qrel_qindices_;
    int nq = (int)qidxs.size();
    std::cout << "\n" << std::string(80,'=') << "\nEVALUATION - "
              << (use_rerank ? "COARSE+DISK-RERANK" : "COARSE ONLY") << "\n"
              << std::string(80,'=') << "\n"
              << "m=" << m_ << " ef=" << ef_;
    if (use_rerank) std::cout << " coarse_k=" << coarse_k_;
    std::cout << " final_k=" << final_k_ << " query_threads=" << query_threads_
              << "  Queries(with qrels): " << nq << "\n" << std::string(80,'=') << "\n";

    std::vector<std::vector<RetrievalResult>> all(nq);
    std::vector<double> ht(nq), st(nq), rt(nq), pt(nq), ct(nq);
    auto t0 = std::chrono::steady_clock::now();
    std::atomic<int> completed{0};
    std::mutex progress_mu;
    parallel_for_queries(nq, query_threads_, [&](int i) {
        int qi = qidxs[i];
        all[i] = search_query(qi, ht[i], st[i], rt[i], pt[i], ct[i], use_rerank);
        int done = completed.fetch_add(1) + 1;
        int step = std::max(1, nq/20);
        if (done % step == 0 || done == nq) {
            std::lock_guard<std::mutex> lock(progress_mu);
            double sec = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
            std::cout << "  " << done << "/" << nq
                      << " (" << std::fixed << std::setprecision(1) << 100.0*done/nq << "%)"
                      << "  " << std::setprecision(2) << (done/sec) << " q/s\n";
        }
    });
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    double ah = std::accumulate(ht.begin(),ht.end(),0.0)/nq;
    double as = std::accumulate(st.begin(),st.end(),0.0)/nq;
    double ar = std::accumulate(rt.begin(),rt.end(),0.0)/nq;
    double at = ah + as + ar;
    std::cout << "\nwall=" << std::fixed << std::setprecision(2) << wall << "s"
              << "  CenterSearch=" << std::setprecision(3) << ah
              << "ms  Scoring=" << as << "ms";
    if (use_rerank) std::cout << "  Rerank=" << ar << "ms";
    std::cout << "  Total=" << at << "ms  QPS=" << std::setprecision(1) << (nq / wall) << "\n"
              << std::string(80,'-') << "\n";
    for (auto& [k,p] : compute_precision_at_k(all, {10,20,50,100}))
        std::cout << "  P" << k << "@" << k << ": " << std::fixed << std::setprecision(4) << p << "\n";
    std::cout << std::string(80,'-') << "\n";
    // Output R10@10, R20@20, R50@50, R100@100 (recall@k where ground truth is top-k)
    for (auto& [k,r] : compute_recall_k_at_k(all, {10,20,50,100}))
        std::cout << "  R" << k << "@" << k << ": " << std::fixed << std::setprecision(4) << r << "\n";
    std::cout << std::string(80,'=') << "\n";
}

void ClusterDiskRetrievalV2::evaluate_both() {
    const auto& qidxs = qrel_qindices_;
    int nq = (int)qidxs.size();
    std::cout << "\n" << std::string(80,'=') << "\n"
              << "COMPARATIVE: COARSE-ONLY vs COARSE+DISK-RERANK\n"
              << "Queries(with qrels): " << nq << "\n" << std::string(80,'=') << "\n";
    std::vector<std::vector<RetrievalResult>> co(nq), rr(nq);
    std::vector<double> hc(nq),sc(nq),dc(nq),prc(nq),cmc(nq),
                        hr(nq),sr(nq),dr(nq),prr(nq),cmr(nq);
    auto t0 = std::chrono::steady_clock::now();
    std::cout << "Coarse-only...\n";
    parallel_for_queries(nq, query_threads_, [&](int i) {
        int qi=qidxs[i]; co[i]=search_query(qi,hc[i],sc[i],dc[i],prc[i],cmc[i],false);
    });
    std::cout << "Coarse+rerank...\n";
    parallel_for_queries(nq, query_threads_, [&](int i) {
        int qi=qidxs[i]; rr[i]=search_query(qi,hr[i],sr[i],dr[i],prr[i],cmr[i],true);
    });
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    double ahc=std::accumulate(hc.begin(),hc.end(),0.0)/nq,
           asc=std::accumulate(sc.begin(),sc.end(),0.0)/nq,
           ahr=std::accumulate(hr.begin(),hr.end(),0.0)/nq,
           asr=std::accumulate(sr.begin(),sr.end(),0.0)/nq,
           adr=std::accumulate(dr.begin(),dr.end(),0.0)/nq,
           apr=std::accumulate(prr.begin(),prr.end(),0.0)/nq,
           acr=std::accumulate(cmr.begin(),cmr.end(),0.0)/nq;
    double avg_co=ahc+asc, avg_rr=ahr+asr+adr;
    std::vector<int> kv={10,20,50,100};
    auto pc=compute_precision_at_k(co,kv), pr=compute_precision_at_k(rr,kv);
    auto rc=compute_recall_k_at_100(co,{10,20,50}), rrm=compute_recall_k_at_100(rr,{10,20,50});
    std::cout << "\nwall=" << std::fixed << std::setprecision(2) << wall << "s\n"
              << "Coarse-only:   CenterSearch=" << std::setprecision(3) << ahc << "ms Scoring=" << asc
              << "ms  Total=" << avg_co << "ms (" << std::setprecision(1) << 1000.0/avg_co << " q/s)\n"
              << "Coarse+rerank: CenterSearch=" << std::setprecision(3) << ahr << "ms Scoring=" << asr
              << "ms Rerank=" << adr << "ms  [pread=" << apr << "ms compute=" << acr << "ms]"
              << "  Total=" << avg_rr
              << "ms (" << std::setprecision(1) << 1000.0/avg_rr << " q/s)\n"
              << "Slowdown: " << std::setprecision(2) << avg_rr/avg_co << "x\n\n";
    for (int k : kv) { if (!pc.count(k)) continue;
        double d=pr[k]-pc[k], dp=pc[k]>0?d/pc[k]*100:0;
        std::cout << "P" << k << "@" << k << ": " << std::setprecision(4)
                  << pc[k] << " → " << pr[k] << "  " << std::showpos << std::setprecision(1)
                  << dp << std::noshowpos << "%\n"; }
    for (int k : {10,20,50}) { if (!rc.count(k)) continue;
        double d=rrm[k]-rc[k], dp=rc[k]>0?d/rc[k]*100:0;
        std::cout << "R" << k << "@100: " << std::setprecision(4)
                  << rc[k] << " → " << rrm[k] << "  " << std::showpos << std::setprecision(1)
                  << dp << std::noshowpos << "%\n"; }
    std::cout << std::string(80,'=') << "\n";
}

// eval_lazy  —  QPS sweep over a fixed set of m values
void ClusterDiskRetrievalV2::eval_lazy() {
    static const std::vector<int> m_vals =
        {2,5,10,15,20,25,30,35,40,45,50,60,70,80,90,100};
    const auto& qidxs = qrel_qindices_;
    int nq  = (int)qidxs.size();
    int orig_m = m_;

    std::cout << "\n" << std::string(72,'=') << "\n"
              << "EVAL_LAZY  coarse_k=" << coarse_k_
              << "  ef=" << ef_ << "  queries(qrel)=" << nq << "\n"
              << std::string(72,'=') << "\n"
              << std::setw(5)  << "m"
              << std::setw(9)  << "QPS"
              << std::setw(12) << "CenterSearch"
              << std::setw(9)  << "Score"
              << std::setw(9)  << "Rerank"
              << std::setw(9)  << "pread"
              << std::setw(9)  << "compute"
              << std::setw(10) << "P10@10"
              << std::setw(10) << "P20@20"
              << std::setw(10) << "P50@50"
              << std::setw(10) << "P100@100"
              << std::setw(10) << "P10@GT100"
              << std::setw(10) << "P20@GT100"
              << std::setw(10) << "P50@GT100"
              << "\n" << std::string(137,'-') << "\n";

    for (int mv : m_vals) {
        m_ = mv;
        std::vector<std::vector<RetrievalResult>> all(nq);
        std::vector<double> vh(nq),vs(nq),vr(nq),vp(nq),vc(nq);
        auto t0 = std::chrono::steady_clock::now();
        parallel_for_queries(nq, query_threads_, [&](int i) {
            int qi = qidxs[i];
            all[i] = search_query(qi, vh[i], vs[i], vr[i], vp[i], vc[i], true);
        });
        double wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        double qps  = nq / wall;
        double ah   = std::accumulate(vh.begin(),vh.end(),0.0)/nq;
        double as   = std::accumulate(vs.begin(),vs.end(),0.0)/nq;
        double ar   = std::accumulate(vr.begin(),vr.end(),0.0)/nq;
        double ap   = std::accumulate(vp.begin(),vp.end(),0.0)/nq;
        double ac   = std::accumulate(vc.begin(),vc.end(),0.0)/nq;
        auto pm = compute_precision_at_k(all, {10,20,50,100});
        auto pm100 = compute_recall_k_at_100(all, {10,20,50});
        std::cout << std::setw(5)  << mv
                  << std::setw(9)  << std::fixed << std::setprecision(1) << qps
                  << std::setw(9)  << std::setprecision(2) << ah
                  << std::setw(9)  << as
                  << std::setw(9)  << ar
                  << std::setw(9)  << ap
                  << std::setw(9)  << ac
                  << std::setw(10) << std::setprecision(4) << pm[10]
                  << std::setw(10) << pm[20]
                  << std::setw(10) << pm[50]
                  << std::setw(10) << pm[100]
                  << std::setw(10) << pm100[10]
                  << std::setw(10) << pm100[20]
                  << std::setw(10) << pm100[50]
                  << "\n";
        std::cout.flush();
        { std::vector<std::vector<RetrievalResult>> tmp; tmp.swap(all); }
    }

    m_ = orig_m;
    std::cout << std::string(72,'=') << "\n"
              << "m restored to " << m_ << "\n";
}

// Single-process sweep used by the small Quora/SciDocs-style experiments.
// The graph and postings remain resident while only m_ changes.
void ClusterDiskRetrievalV2::eval_m_10_100() {
    static const std::vector<int> m_vals = {10,20,30,40,50,60,70,80,90,100};
    const auto& qidxs = qrel_qindices_;
    int nq = (int)qidxs.size();
    int orig_m = m_;
    std::cout << "\n" << std::string(72,'=') << "\n"
              << "EVAL_M_10_100  coarse_k=" << coarse_k_
              << "  ef=" << ef_ << "  queries(qrel)=" << nq << "\n"
              << "Graph remains loaded for the full sweep\n"
              << std::string(72,'=') << "\n"
              << std::setw(5) << "m" << std::setw(9) << "QPS"
              << std::setw(12) << "CenterSearch" << std::setw(9) << "Score"
              << std::setw(9) << "Rerank" << std::setw(9) << "pread"
              << std::setw(9) << "compute" << std::setw(10) << "P10@10"
              << std::setw(10) << "P20@20" << std::setw(10) << "P50@50"
              << std::setw(10) << "P100@100" << std::setw(10) << "P10@GT100"
              << std::setw(10) << "P20@GT100" << std::setw(10) << "P50@GT100"
              << "\n" << std::string(137,'-') << "\n";
    for (int mv : m_vals) {
        m_ = mv;
        std::vector<std::vector<RetrievalResult>> all(nq);
        std::vector<double> vh(nq),vs(nq),vr(nq),vp(nq),vc(nq);
        auto t0 = std::chrono::steady_clock::now();
        parallel_for_queries(nq, query_threads_, [&](int i) {
            int qi = qidxs[i];
            all[i] = search_query(qi, vh[i], vs[i], vr[i], vp[i], vc[i], true);
        });
        double wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        double qps = nq / wall;
        double ah = std::accumulate(vh.begin(),vh.end(),0.0)/nq;
        double as = std::accumulate(vs.begin(),vs.end(),0.0)/nq;
        double ar = std::accumulate(vr.begin(),vr.end(),0.0)/nq;
        double ap = std::accumulate(vp.begin(),vp.end(),0.0)/nq;
        double ac = std::accumulate(vc.begin(),vc.end(),0.0)/nq;
        auto pm = compute_precision_at_k(all, {10,20,50,100});
        auto pm100 = compute_recall_k_at_100(all, {10,20,50});
        std::cout << std::setw(5) << mv
                  << std::setw(9) << std::fixed << std::setprecision(1) << qps
                  << std::setw(9) << std::setprecision(2) << ah
                  << std::setw(9) << as << std::setw(9) << ar
                  << std::setw(9) << ap << std::setw(9) << ac
                  << std::setw(10) << std::setprecision(4) << pm[10]
                  << std::setw(10) << pm[20] << std::setw(10) << pm[50]
                  << std::setw(10) << pm[100] << std::setw(10) << pm100[10]
                  << std::setw(10) << pm100[20] << std::setw(10) << pm100[50] << "\n";
        std::cout.flush();
    }
    m_ = orig_m;
    std::cout << std::string(72,'=') << "\n"
              << "m restored to " << m_ << "\n";
}

// eval_lazy_LoTTE  —  LoTTE-style parameter sweep
// ef<=100: test m=10,20,50,100
// ef>100: m=ef, test 200,300,400...1000,1200,1400...2000
void ClusterDiskRetrievalV2::eval_lazy_LoTTE() {
    const auto& qidxs = qrel_qindices_;
    int nq  = (int)qidxs.size();
    int orig_m = m_;
    int orig_ef = ef_;

    // Build parameter pairs: (ef, m)
    std::vector<std::pair<int,int>> params;

    // ef <= 100: test m=10,20,50,100
    static const std::vector<int> m_small = {10, 20, 50, 100};
    for (int m : m_small) {
        params.push_back({100, m});
    }

    // ef > 100: m = ef, from 200 to 1000 (step 100)
    for (int v = 200; v <= 1000; v += 100) {
        params.push_back({v, v});
    }
    // ef > 1000: m = ef, from 1200 to 2000 (step 200)
    for (int v = 1200; v <= 2000; v += 200) {
        params.push_back({v, v});
    }

    std::cout << "\n" << std::string(72,'=') << "\n"
              << "EVAL_LAZY_LoTTE  coarse_k=" << coarse_k_
              << "  queries(qrel)=" << nq << "\n"
              << "  Total parameter combinations: " << params.size() << "\n"
              << std::string(72,'=') << "\n"
              << std::setw(7)  << "ef"
              << std::setw(7)  << "m"
              << std::setw(9)  << "QPS"
              << std::setw(12) << "CenterSearch"
              << std::setw(9)  << "Score"
              << std::setw(9)  << "Rerank"
              << std::setw(9)  << "pread"
              << std::setw(9)  << "compute"
              << std::setw(10) << "P10@10"
              << std::setw(10) << "P20@20"
              << std::setw(10) << "P50@50"
              << std::setw(10) << "P100@100"
              << std::setw(10) << "P10@GT100"
              << std::setw(10) << "P20@GT100"
              << std::setw(10) << "P50@GT100"
              << "\n" << std::string(137,'-') << "\n";

    for (const auto& [ef_val, m_val] : params) {
        ef_ = ef_val;
        m_ = m_val;
        if (chnsw_) chnsw_->setEf(ef_val);

        std::vector<std::vector<RetrievalResult>> all(nq);
        std::vector<double> vh(nq),vs(nq),vr(nq),vp(nq),vc(nq);
        auto t0 = std::chrono::steady_clock::now();
        parallel_for_queries(nq, query_threads_, [&](int i) {
            int qi = qidxs[i];
            all[i] = search_query(qi, vh[i], vs[i], vr[i], vp[i], vc[i], true);
        });
        double wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        double qps  = nq / wall;
        double ah   = std::accumulate(vh.begin(),vh.end(),0.0)/nq;
        double as   = std::accumulate(vs.begin(),vs.end(),0.0)/nq;
        double ar   = std::accumulate(vr.begin(),vr.end(),0.0)/nq;
        double ap   = std::accumulate(vp.begin(),vp.end(),0.0)/nq;
        double ac   = std::accumulate(vc.begin(),vc.end(),0.0)/nq;
        auto pm = compute_precision_at_k(all, {10,20,50,100});
        auto pm100 = compute_recall_k_at_100(all, {10,20,50});
        std::cout << std::setw(7)  << ef_val
                  << std::setw(7)  << m_val
                  << std::setw(9)  << std::fixed << std::setprecision(1) << qps
                  << std::setw(9)  << std::setprecision(2) << ah
                  << std::setw(9)  << as
                  << std::setw(9)  << ar
                  << std::setw(9)  << ap
                  << std::setw(9)  << ac
                  << std::setw(10) << std::setprecision(4) << pm[10]
                  << std::setw(10) << pm[20]
                  << std::setw(10) << pm[50]
                  << std::setw(10) << pm[100]
                  << std::setw(10) << pm100[10]
                  << std::setw(10) << pm100[20]
                  << std::setw(10) << pm100[50]
                  << "\n";
        std::cout.flush();
        { std::vector<std::vector<RetrievalResult>> tmp; tmp.swap(all); }
    }

    m_ = orig_m;
    ef_ = orig_ef;
    if (chnsw_) chnsw_->setEf(orig_ef);
    std::cout << std::string(72,'=') << "\n"
              << "Parameters restored: m=" << m_ << " ef=" << ef_ << "\n";
}

void ClusterDiskRetrievalV2::print_info() const {
    std::cout << "\nQueries=" << query_ids_.size() << " Clusters=" << cluster_members_.size()
              << " Docs=" << doc_id_to_local_.size() << " dim=" << dim_
              << "\nm=" << m_ << " ef=" << ef_ << " coarse_k=" << coarse_k_
              << " final_k=" << final_k_ << " query_threads=" << query_threads_ << "\n";
}

static void print_help_v2() {
    std::cout <<
        "\nCommands:\n"
        "  retrieve <index> [0/1]  Retrieve (0=coarse-only, 1=coarse+rerank)\n"
        "  eval [0/1]              Evaluate all queries\n"
        "  eval both               Coarse-only vs coarse+rerank\n"
        "  eval_lazy               QPS sweep: m=2,5,10..100\n"
        "  eval_lazy_LoTTE         LoTTE sweep: ef=100(m=10,20,50,100), ef=200..2000\n"
        "  set-ef <value>          Set HNSW ef\n"
        "  set-m  <value>          Set coarse top-m clusters per token\n"
        "  info                    System info\n"
        "  help / quit\n\n";
}

static std::vector<std::string> split_tokens_v2(const std::string& line) {
    std::vector<std::string> t; std::istringstream iss(line); std::string tok;
    while (iss >> tok) t.push_back(tok); return t;
}

static void run_interactive_v2(ClusterDiskRetrievalV2& ret) {
    print_help_v2();
    std::string line;
    while (true) {
        std::cout << "> "; std::cout.flush();
        if (!std::getline(std::cin, line)) break;
        auto tok = split_tokens_v2(line);
        if (tok.empty()) continue;
        const std::string& cmd = tok[0];
        if (cmd=="quit"||cmd=="q") break;
        if (cmd=="help") { print_help_v2(); continue; }
        if (cmd=="info") { ret.print_info(); continue; }
        if (cmd=="set-ef" && tok.size()>=2) {
            try { ret.set_hnsw_ef(std::stoi(tok[1])); } catch(...) { std::cout<<"invalid\n"; }
            continue; }
        if (cmd=="eval" && tok.size()>=2 && tok[1]=="both") { ret.evaluate_both(); continue; }
        if (cmd=="eval_lazy") { ret.eval_lazy(); continue; }
        if (cmd=="eval_m_10_100") { ret.eval_m_10_100(); continue; }
        if (cmd=="eval_lazy_LoTTE") { ret.eval_lazy_LoTTE(); continue; }
        if (cmd=="eval") {
            bool ur=true;
            if (tok.size()>=2) try { ur=(std::stoi(tok[1])!=0); } catch(...) {}
            ret.evaluate(ur); continue; }
        if (cmd=="set-m" && tok.size()>=2) {
            try { ret.set_m(std::stoi(tok[1])); } catch(...) { std::cout<<"invalid\n"; }
            continue; }
        if (cmd=="retrieve" && tok.size()>=2) {
            try {
                int idx=std::stoi(tok[1]); bool ur=true;
                if (tok.size()>=3) ur=(std::stoi(tok[2])!=0);
                ret.retrieve_query(idx,ur);
            } catch(...) { std::cout<<"invalid\n"; }
            continue; }
        std::cout << "Unknown command. Type 'help'.\n";
    }
}

int main(int argc, char** argv) {
    std::string base_dir=".", jieguo="Jie_Guo", emb_subdir="colbert_64d";
    std::string emb_dir_override, cluster_dir_override,
                reorder_dir_override, quora_dir_override, qrels_file_override;
    std::string eval_split="search_results";
    std::string eval_mode;  // non-interactive mode: eval_once/eval_lazy/eval_m_10_100/eval_lazy_LoTTE
    int m=5, ef=100, coarse_k=1000, final_k=100, query_threads=1;
    bool in_memory = false;
    bool use_cagra = false;

    for (int i=1;i<argc;i++) {
        std::string a=argv[i];
        if      (a=="--base-dir"       &&i+1<argc) base_dir            =argv[++i];
        else if (a=="--jieguo"         &&i+1<argc) jieguo              =argv[++i];
        else if (a=="--emb-subdir"     &&i+1<argc) emb_subdir          =argv[++i];
        else if (a=="--embeddings-dir" &&i+1<argc) emb_dir_override    =argv[++i];
        else if (a=="--cluster-dir"    &&i+1<argc) cluster_dir_override=argv[++i];
        else if (a=="--reorder-dir"    &&i+1<argc) reorder_dir_override=argv[++i];
        else if (a=="--quora-dir"      &&i+1<argc) quora_dir_override  =argv[++i];
        else if (a=="--eval-split"     &&i+1<argc) eval_split          =argv[++i];
        else if (a=="--qrels-file"     &&i+1<argc) qrels_file_override =argv[++i];
        else if (a=="--eval-mode"      &&i+1<argc) eval_mode           =argv[++i];
        else if (a=="--m"        &&i+1<argc) m        =std::stoi(argv[++i]);
        else if (a=="--ef"       &&i+1<argc) ef       =std::stoi(argv[++i]);
        else if (a=="--coarse-k" &&i+1<argc) coarse_k =std::stoi(argv[++i]);
        else if (a=="--final-k"  &&i+1<argc) final_k  =std::stoi(argv[++i]);
        else if (a=="--query-threads" &&i+1<argc) query_threads=std::stoi(argv[++i]);
        else if (a=="--in-memory") in_memory = true;
        else if (a=="--center-index" &&i+1<argc) {
            std::string mode = argv[++i];
            if (mode == "cagra") use_cagra = true;
            else if (mode == "hnsw") use_cagra = false;
            else { std::cerr << "Invalid --center-index: " << mode << " (use hnsw or cagra)\n"; return 2; }
        }
        else if (a=="--help"||a=="-h") {
            std::cout << "Usage: " << argv[0] << " [same options as cluster_eval_disk]\n"
                         "Optimisations: HNSW/CAGRA center search + integer-array scoring + libaio async IO\n"
                         "\n--center-index hnsw|cagra   Select center graph (CAGRA requires cagra_centers.bin and cagra_centers.nsg)\n"
                         "\n--eval-mode <mode>  Non-interactive mode: eval_once, eval_lazy, eval_m_10_100, or eval_lazy_LoTTE\n"
                         "\n--query-threads N    Parallel queries (default: 1)\n"
                         "\n--in-memory         mmap 整个 reorder 文件到 RAM（首次预热后零 IO）\n";
            return 0; }
    }
    std::string emb_dir = emb_dir_override.empty()
        ? base_dir+"/"+jieguo+"/"+emb_subdir : emb_dir_override;
    std::string cluster_dir = cluster_dir_override.empty()
        ? base_dir+"/cpp/"+jieguo+"/"+emb_subdir+"/HNSW_cluster_rep_nc" : cluster_dir_override;
    std::string reorder_dir = reorder_dir_override.empty()
        ? emb_dir+"/reorder_embedding" : reorder_dir_override;
    std::string quora_dir = quora_dir_override.empty()
        ? base_dir+"/quora" : quora_dir_override;
    std::string qrels_path = qrels_file_override.empty()
        ? quora_dir+"/qrels/"+eval_split+".tsv" : qrels_file_override;

    std::cout << "emb=" << emb_dir << "\ncluster=" << cluster_dir
              << "\nreorder=" << reorder_dir << "\nqrels=" << qrels_path
              << "\nin-memory=" << (in_memory ? "YES" : "no") << "\n\n";

    ClusterDiskRetrievalV2 ret(emb_dir, cluster_dir, reorder_dir, qrels_path,
                               m, ef, coarse_k, final_k, query_threads, in_memory, use_cagra);

    // Non-interactive mode: run the specified eval and exit
    if (!eval_mode.empty()) {
        if (eval_mode == "eval_lazy") {
            ret.eval_lazy();
        } else if (eval_mode == "eval_lazy_LoTTE") {
            ret.eval_lazy_LoTTE();
        } else if (eval_mode == "eval_m_10_100") {
            ret.eval_m_10_100();
        } else if (eval_mode == "eval_once") {
            ret.evaluate();
        } else {
            std::cerr << "Unknown eval-mode: " << eval_mode << " (use eval_once, eval_lazy, eval_m_10_100, or eval_lazy_LoTTE)\n";
            return 1;
        }
        return 0;
    }

    // Interactive mode
    run_interactive_v2(ret);
    return 0;
}

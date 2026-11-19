#pragma once
/**
 * npy_reader.h  —  minimal NumPy .npy / metadata reader (header-only)
 */
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct NpyArray {
    std::vector<float> data;
    size_t rows = 0;
    size_t cols = 0;
    const float* row_ptr(size_t r) const { return data.data() + r * cols; }
    float* row_ptr(size_t r) { return data.data() + r * cols; }
};

inline NpyArray load_npy_float32(const std::string& path) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) throw std::runtime_error("Cannot open " + path);

    char magic[6];
    fin.read(magic, 6);
    if (std::string(magic + 1, 5) != "NUMPY")
        throw std::runtime_error("Not a standard .npy file: " + path);

    uint8_t major, minor;
    fin.read(reinterpret_cast<char*>(&major), 1);
    fin.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t hl16;
        fin.read(reinterpret_cast<char*>(&hl16), 2);
        header_len = hl16;
    } else {
        fin.read(reinterpret_cast<char*>(&header_len), 4);
    }

    std::string header(header_len, '\0');
    fin.read(&header[0], header_len);

    auto parse_shape = [&](size_t& r, size_t& c) {
        auto pos = header.find("'shape'");
        if (pos == std::string::npos) throw std::runtime_error("no shape in npy header");
        auto lp = header.find('(', pos);
        auto rp = header.find(')', lp);
        std::string shape_str = header.substr(lp + 1, rp - lp - 1);
        auto comma = shape_str.find(',');
        r = std::stoull(shape_str.substr(0, comma));
        if (comma != std::string::npos) {
            std::string rest = shape_str.substr(comma + 1);
            while (!rest.empty() && (rest[0] == ' ' || rest[0] == ',')) rest.erase(rest.begin());
            while (!rest.empty() && (rest.back() == ' ' || rest.back() == ',')) rest.pop_back();
            c = rest.empty() ? 1 : std::stoull(rest);
        } else {
            c = 1;
        }
    };

    NpyArray arr;
    parse_shape(arr.rows, arr.cols);
    arr.data.resize(arr.rows * arr.cols);
    fin.read(reinterpret_cast<char*>(arr.data.data()),
             arr.rows * arr.cols * sizeof(float));
    if (!fin)
        throw std::runtime_error("Truncated .npy file: " + path);
    return arr;
}

inline NpyArray load_raw_float32(const std::string& path,
                                  size_t rows, size_t cols) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) throw std::runtime_error("Cannot open " + path);

    NpyArray arr;
    arr.rows = rows;
    arr.cols = cols;
    arr.data.resize(rows * cols);
    fin.read(reinterpret_cast<char*>(arr.data.data()),
             rows * cols * sizeof(float));
    if (!fin)
        throw std::runtime_error("Truncated raw binary file: " + path);
    return arr;
}

inline NpyArray load_raw_float32_rows(const std::string& path,
                                       size_t total_rows, size_t cols,
                                       size_t row_start, size_t row_end) {
    if (row_end > total_rows) row_end = total_rows;
    std::ifstream fin(path, std::ios::binary);
    if (!fin) throw std::runtime_error("Cannot open " + path);

    fin.seekg(static_cast<std::streamoff>(row_start * cols * sizeof(float)));

    NpyArray arr;
    arr.rows = row_end - row_start;
    arr.cols = cols;
    arr.data.resize(arr.rows * cols);
    fin.read(reinterpret_cast<char*>(arr.data.data()),
             arr.rows * cols * sizeof(float));
    if (!fin)
        throw std::runtime_error("Read failed for raw binary file: " + path);
    return arr;
}

// 5.  JSON helpers

inline std::pair<size_t,size_t> npy_shape(const std::string& path) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) throw std::runtime_error("Cannot open " + path);
    char magic[6]; fin.read(magic, 6);
    uint8_t major, minor;
    fin.read(reinterpret_cast<char*>(&major), 1);
    fin.read(reinterpret_cast<char*>(&minor), 1);
    uint32_t header_len = 0;
    if (major == 1) { uint16_t h; fin.read(reinterpret_cast<char*>(&h), 2); header_len = h; }
    else              { fin.read(reinterpret_cast<char*>(&header_len), 4); }
    std::string header(header_len, '\0'); fin.read(&header[0], header_len);
    auto pos = header.find("'shape'");
    auto lp = header.find('(', pos), rp = header.find(')', lp);
    std::string s = header.substr(lp + 1, rp - lp - 1);
    auto comma = s.find(',');
    size_t rows = std::stoull(s.substr(0, comma));
    size_t cols = 0;
    if (comma != std::string::npos) {
        std::string rest = s.substr(comma + 1);
        while (!rest.empty() && (rest[0] == ' ' || rest[0] == ',')) rest.erase(rest.begin());
        if (!rest.empty()) cols = std::stoull(rest);
    }
    return {rows, cols};
}

inline int64_t npy_data_offset(const std::string& path) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) throw std::runtime_error("Cannot open " + path);
    char magic[6]; fin.read(magic, 6);
    if (magic[0] != '\x93' || std::string(magic + 1, 5) != "NUMPY")
        return 0;
    uint8_t major, minor;
    fin.read(reinterpret_cast<char*>(&major), 1);
    fin.read(reinterpret_cast<char*>(&minor), 1);
    uint32_t header_len = 0;
    if (major == 1) { uint16_t h; fin.read(reinterpret_cast<char*>(&h), 2); header_len = h; }
    else            { fin.read(reinterpret_cast<char*>(&header_len), 4); }
    int64_t field_bytes = (major == 1) ? 2 : 4;
    return 6 + 2 + field_bytes + (int64_t)header_len;
}

inline std::string read_file_text(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
}

inline int64_t json_int(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) throw std::runtime_error("key not found: " + key);
    pos = json.find(':', pos);
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    return std::stoll(json.substr(pos));
}

inline std::vector<int> json_int_array(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) throw std::runtime_error("key not found: " + key);
    auto lb = json.find('[', pos);
    auto rb = json.find(']', lb);
    std::string arr_str = json.substr(lb + 1, rb - lb - 1);
    std::vector<int> result;
    std::istringstream iss(arr_str);
    std::string token;
    while (std::getline(iss, token, ',')) {
        while (!token.empty() && (token[0] == ' ' || token[0] == '\n')) token.erase(token.begin());
        while (!token.empty() && (token.back() == ' ' || token.back() == '\n')) token.pop_back();
        if (!token.empty()) result.push_back(std::stoi(token));
    }
    return result;
}

// 6.  TSV loaders
struct VectorDocInfo {
    std::string doc_type;
    std::string doc_id;
    int token_idx;
};

inline std::unordered_map<int, VectorDocInfo>
load_vector_to_doc_mapping(const std::string& path) {
    std::unordered_map<int, VectorDocInfo> mapping;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        int vec_id;
        VectorDocInfo info;
        if (iss >> vec_id >> info.doc_type >> info.doc_id >> info.token_idx)
            mapping[vec_id] = info;
    }
    return mapping;
}

inline std::vector<std::string> load_ids_file(const std::string& path) {
    std::vector<std::string> ids;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        // Accept both common formats:
        //   integer-index<TAB>id
        //   id
        // The latter is used by some LOTTE exports.
        std::string first, second;
        if (!(iss >> first)) continue;
        if (iss >> second) ids.push_back(second);
        else ids.push_back(first);
    }
    return ids;
}

inline std::unordered_map<std::string, std::unordered_set<std::string>>
load_qrels(const std::string& path) {
    std::unordered_map<std::string, std::unordered_set<std::string>> qrels;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line.find("query-id") != std::string::npos ||
            line.find("query_id") != std::string::npos) continue;
        std::istringstream iss(line);
        std::string qid, did;
        int score;
        if (iss >> qid >> did >> score)
            if (score > 0) qrels[qid].insert(did);
    }
    return qrels;
}

inline std::unordered_map<int, std::vector<int>>
load_cluster_members_json(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);

    std::unordered_map<int, std::vector<int>> result;
    int current_cluster = -1;
    std::string line;

    while (std::getline(f, line)) {
        size_t s = 0;
        while (s < line.size() && (line[s] == ' ' || line[s] == '\t')) s++;
        if (s == line.size()) continue;
        const char* p = line.c_str() + s;

        if (*p == '"') {
            const char* q = p + 1;
            while (*q && *q != '"') q++;
            std::string key(p + 1, q);
            bool numeric = !key.empty();
            for (char c : key) if (!isdigit(c)) { numeric = false; break; }
            if (numeric) {
                current_cluster = std::stoi(key);
                const char* bracket = std::strchr(q + 1, '[');
                if (bracket) {
                    bracket++;
                    while (*bracket) {
                        while (*bracket == ' ' || *bracket == ',') bracket++;
                        if (*bracket == ']' || *bracket == '\0') break;
                        if (std::isdigit(*bracket) || *bracket == '-') {
                            result[current_cluster].push_back(std::atoi(bracket));
                            while (*bracket && *bracket != ',' && *bracket != ']')
                                bracket++;
                        } else break;
                    }
                }
            }
            continue;
        }

        if (*p == '{' && current_cluster >= 0) {
            const char* vid_key = "\"vector_id\"";
            const char* found = std::strstr(p, vid_key);
            if (found) {
                const char* colon = std::strchr(found + 11, ':');
                if (colon) {
                    colon++;
                    while (*colon == ' ' || *colon == '\t') colon++;
                    result[current_cluster].push_back(std::atoi(colon));
                }
            }
            continue;
        }
    }
    return result;
}

inline std::vector<std::string> load_query_ids(const std::string& path) {
    std::vector<std::string> ids;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
            line.pop_back();
        if (line.empty()) continue;

        auto last_ws = line.find_last_of(" \t");
        if (last_ws == std::string::npos) {
            ids.push_back(line);
        } else {
            ids.push_back(line.substr(last_ws + 1));
        }
    }
    return ids;
}

// Lightweight fingerprint for detecting an index built against a different
// doc_ids.txt ordering. The masked FNV-1a value fits the signed JSON integer
// reader used by the experiment programs.
inline uint64_t doc_ids_file_fingerprint(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);
    uint64_t hash = 14695981039346656037ull;
    char buffer[1 << 16];
    while (f) {
        f.read(buffer, sizeof(buffer));
        const auto n = f.gcount();
        for (std::streamsize i = 0; i < n; ++i) {
            hash ^= static_cast<unsigned char>(buffer[i]);
            hash *= 1099511628211ull;
        }
    }
    return hash & 0x7fffffffffffffffull;
}

inline size_t get_rss_kb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("VmRSS:", 0) == 0)
            return std::stoul(line.substr(6));
    return 0;
}

inline void print_memory(const std::string& label = "") {
    size_t rss = get_rss_kb();
    std::cout << "[Memory" << (label.empty() ? "" : " " + label) << "] "
              << rss / 1024 << " MB  (" << rss << " KB)\n";
}

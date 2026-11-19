#pragma once
#include <atomic>
#include <exception>
#include <stdexcept>

// Independent document tasks; exceptions cross the OpenMP boundary safely.
template<class Function>
void parallel_documents(int count, int threads, Function fn) {
    if (threads <= 0) throw std::invalid_argument("document threads must be positive");
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    #pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (int i = 0; i < count; ++i) {
        if (failed.load()) continue;
        try { fn(i); }
        catch (...) {
            failed.store(true);
            #pragma omp critical(muvera_document_error)
            { if (!error) error = std::current_exception(); }
        }
    }
    if (error) std::rethrow_exception(error);
}

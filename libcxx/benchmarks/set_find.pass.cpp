#include <unordered_set>
#include <vector>
#include <cstdint>

#include "benchmark/benchmark_api.h"

template <class IntT>
std::vector<IntT> getInputs(size_t N) {
    std::vector<IntT> inputs;
    for (size_t i=0; i < N; ++i) {
        inputs.push_back(i);
    }
    return inputs;
}

template <class Container, class Inputs>
void BM_SetLookup(benchmark::State& st, Container c, Inputs const& in) {
    c.insert(in.begin(), in.end());
    const auto end = in.end();
    while (st.KeepRunning()) {
        for (auto it = in.begin(); it != end; ++it) {
            benchmark::DoNotOptimize(c.find(*it++));
        }
    }
}
BENCHMARK_CAPTURE(BM_SetLookup, uint32_lookup,
    std::unordered_set<uint32_t>{}, getInputs<uint32_t>(1024));

BENCHMARK_MAIN()

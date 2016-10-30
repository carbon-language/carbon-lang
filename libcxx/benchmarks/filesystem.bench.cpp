#include <experimental/filesystem>

#include "benchmark/benchmark_api.h"
#include "GenerateInput.hpp"

namespace fs = std::experimental::filesystem;

static const size_t TestNumInputs = 1024;


template <class GenInputs>
void BM_PathConstructString(benchmark::State &st, GenInputs gen) {
  using namespace fs;
  const auto in = gen(st.range(0));
  path PP;
  for (auto& Part : in)
    PP /= Part;
  benchmark::DoNotOptimize(PP.native().data());
  while (st.KeepRunning()) {
    const path P(PP.native());
    benchmark::DoNotOptimize(P.native().data());
  }
}
BENCHMARK_CAPTURE(BM_PathConstructString, large_string,
  getRandomStringInputs)->Arg(TestNumInputs);


template <class GenInputs>
void BM_PathConstructCStr(benchmark::State &st, GenInputs gen) {
  using namespace fs;
  const auto in = gen(st.range(0));
  path PP;
  for (auto& Part : in)
    PP /= Part;
  benchmark::DoNotOptimize(PP.native().data());
  while (st.KeepRunning()) {
    const path P(PP.native().c_str());
    benchmark::DoNotOptimize(P.native().data());
  }
}
BENCHMARK_CAPTURE(BM_PathConstructCStr, large_string,
  getRandomStringInputs)->Arg(TestNumInputs);

template <class GenInputs>
void BM_PathIterateMultipleTimes(benchmark::State &st, GenInputs gen) {
  using namespace fs;
  const auto in = gen(st.range(0));
  path PP;
  for (auto& Part : in)
    PP /= Part;
  benchmark::DoNotOptimize(PP.native().data());
  while (st.KeepRunning()) {
    for (auto &E : PP) {
      benchmark::DoNotOptimize(E.native().data());
    }
    benchmark::ClobberMemory();
  }
}
BENCHMARK_CAPTURE(BM_PathIterateMultipleTimes, iterate_elements,
  getRandomStringInputs)->Arg(TestNumInputs);


template <class GenInputs>
void BM_PathIterateOnce(benchmark::State &st, GenInputs gen) {
  using namespace fs;
  const auto in = gen(st.range(0));
  path PP;
  for (auto& Part : in)
    PP /= Part;
  benchmark::DoNotOptimize(PP.native().data());
  while (st.KeepRunning()) {
    const path P = PP.native();
    for (auto &E : P) {
      benchmark::DoNotOptimize(E.native().data());
    }
    benchmark::ClobberMemory();
  }
}
BENCHMARK_CAPTURE(BM_PathIterateOnce, iterate_elements,
  getRandomStringInputs)->Arg(TestNumInputs);

template <class GenInputs>
void BM_PathIterateOnceBackwards(benchmark::State &st, GenInputs gen) {
  using namespace fs;
  const auto in = gen(st.range(0));
  path PP;
  for (auto& Part : in)
    PP /= Part;
  benchmark::DoNotOptimize(PP.native().data());
  while (st.KeepRunning()) {
    const path P = PP.native();
    const auto B = P.begin();
    auto I = P.end();
    while (I != B) {
      --I;
      benchmark::DoNotOptimize(*I);
    }
    benchmark::DoNotOptimize(*I);
  }
}
BENCHMARK_CAPTURE(BM_PathIterateOnceBackwards, iterate_elements,
  getRandomStringInputs)->Arg(TestNumInputs);

BENCHMARK_MAIN()

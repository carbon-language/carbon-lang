// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <cstdint>

#include "common/check.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "testing/base/global_exe_path.h"
#include "toolchain/base/install_paths.h"
#include "toolchain/base/install_paths_test_helpers.h"
#include "toolchain/base/mem_usage.h"
#include "toolchain/driver/driver.h"

namespace Carbon::Testing {
namespace {

// The name of the input file. Compiling it pulls in the entire `Core` prelude
// via the implicit prelude import. The prelude is lexed, parsed, and checked
// lazily as part of the check phase, so checking this file is dominated by the
// cost of compiling the prelude itself.
static constexpr llvm::StringLiteral InputFileName =
    "prelude_benchmark_input.carbon";

// The variations of input file we compile. The prelude is fully checked in all
// cases; the variations differ in how much of the prelude they then exercise,
// which drives additional instantiation in the main file.
enum class Import : uint8_t {
  // The minimal possible file: no use of the prelude at all. Measures the
  // baseline cost of compiling the prelude on its own.
  Nothing,
  // The minimal use of a single prelude entity (the `i32` type).
  SingleType,
  // A small amount of code that exercises many `impl`s across the prelude: the
  // arithmetic and comparison operators on several builtin numeric types.
  ManyTypesManyImpls,
  // The smallest code that pulls in a wide swath of the prelude: a `for` loop
  // over an array plus a string operation, touching iteration, indexing,
  // comparison, arithmetic, copy/destroy, and string support.
  NonTrivialAmount,
};

// Returns the source code for a given import variation.
static auto ImportSource(Import import) -> llvm::StringRef {
  switch (import) {
    case Import::Nothing:
      return "fn F() {}\n";
    case Import::SingleType:
      return "fn F(n: i32) -> i32 { return n; }\n";
    case Import::ManyTypesManyImpls:
      return "fn F(a: i32, b: i64, c: u32, d: f64) -> bool {\n"
             "  return ((a + a - a) == a) and ((b * b) > b) and\n"
             "         ((c / c) <= c) and (d < d);\n"
             "}\n";
    case Import::NonTrivialAmount:
      return "fn F(a: array(i32, 3), s: str) -> i32 {\n"
             "  var n: i32 = s.Size() as i32;\n"
             "  for (x: i32 in a) { n = (n + x) * x; }\n"
             "  return n;\n"
             "}\n";
  }
}

// The memory sizes of interest, in bytes, summed across all compiled files.
struct MemSizes {
  int64_t total = 0;
  int64_t sem_ir = 0;
  int64_t insts = 0;
  int64_t inst_locs = 0;
  int64_t constant_values = 0;
  int64_t inst_blocks = 0;
  int64_t allocator = 0;
};

// Sets up an in-memory filesystem containing the prelude and the input file,
// and drives repeated in-process check-phase compilations of that input.
//
// Because the prelude is implicitly imported and checking resolves imports
// eagerly, every compilation fully compiles the prelude.
class PreludeCompileBenchmark {
 public:
  explicit PreludeCompileBenchmark(llvm::StringRef source)
      : fs_(new llvm::vfs::InMemoryFileSystem),
        installation_(InstallPaths::MakeForBazelRunfiles(GetExePath())) {
    AddPreludeFilesToVfs(installation_, fs_);
    fs_->addFile(InputFileName, /*ModificationTime=*/0,
                 llvm::MemoryBuffer::getMemBuffer(source));
  }

  // Runs a single check-phase compilation of the input, discarding all output.
  // If `mem_usage` is non-null, the driver collects each compiled file's memory
  // usage into it. Returns whether the compilation succeeded.
  auto RunCompile(MemUsage* mem_usage = nullptr) -> bool {
    llvm::raw_null_ostream output;
    Driver driver(fs_, &installation_, /*input_stream=*/nullptr, &output,
                  &llvm::errs());
    if (mem_usage) {
      driver.set_mem_usage(mem_usage);
    }
    return driver.RunCommand({"compile", "--phase=check", InputFileName})
        .success;
  }

  // Compiles once and records the memory sizes of interest as benchmark
  // counters.
  auto RecordMemoryCounters(benchmark::State& state) -> void {
    // The driver collects each compiled file's usage into this one `MemUsage`:
    // the prelude files and the input each contribute entries with the same
    // labels, which `SumMemUsage` accumulates.
    MemUsage mem_usage;
    bool success = RunCompile(&mem_usage);
    CARBON_CHECK(success);
    MemSizes sizes = SumMemUsage(mem_usage);
    // Guard against a `MemUsage` label being renamed out from under us, which
    // would otherwise silently report zero sizes.
    CARBON_CHECK(sizes.insts > 0, "no SemIR instruction memory was recorded");

    // The `Mem` prefix marks these as memory-cost metrics (smaller is better)
    // for `scripts/bench_runner.py`.
    auto set = [&](const char* name, int64_t bytes) {
      state.counters[name] = benchmark::Counter(static_cast<double>(bytes));
    };
    set("MemTotal", sizes.total);
    set("MemSemIR", sizes.sem_ir);
    set("MemSemIRInsts", sizes.insts);
    set("MemSemIRInstLocs", sizes.inst_locs);
    set("MemSemIRConstantValues", sizes.constant_values);
    set("MemSemIRInstBlocks", sizes.inst_blocks);
    set("MemSemIRAllocator", sizes.allocator);
  }

 private:
  // Sums the memory usage collected by the driver into the sizes of interest.
  // The specific `sem_ir_.*` sizes are also included in the overall `sem_ir`
  // size, which sums every `sem_ir_.*` label.
  static auto SumMemUsage(const MemUsage& mem_usage) -> MemSizes {
    MemSizes sizes;
    for (const MemUsage::Entry& entry : mem_usage.entries()) {
      llvm::StringRef label = entry.label;
      sizes.total += entry.used_bytes;
      if (label.starts_with("sem_ir_")) {
        sizes.sem_ir += entry.used_bytes;
      }
      if (label == "sem_ir_.insts_.values_") {
        sizes.insts += entry.used_bytes;
      } else if (label == "sem_ir_.insts_.loc_ids_") {
        sizes.inst_locs += entry.used_bytes;
      } else if (label == "sem_ir_.constant_values_.values_") {
        sizes.constant_values += entry.used_bytes;
      } else if (label == "sem_ir_.inst_blocks_.values_") {
        sizes.inst_blocks += entry.used_bytes;
      } else if (label == "sem_ir_.allocator_") {
        sizes.allocator += entry.used_bytes;
      }
    }
    return sizes;
  }

  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> fs_;
  const InstallPaths installation_;
};

template <Import V>
static auto BM_PreludeCompile(benchmark::State& state) -> void {
  PreludeCompileBenchmark bench(ImportSource(V));
  bench.RecordMemoryCounters(state);

  for (auto _ : state) {
    bool success = bench.RunCompile();
    CARBON_CHECK(success);
    benchmark::DoNotOptimize(success);
  }
}

BENCHMARK(BM_PreludeCompile<Import::Nothing>)
    ->Name("BM_PreludeCompile/Nothing");
BENCHMARK(BM_PreludeCompile<Import::SingleType>)
    ->Name("BM_PreludeCompile/SingleType");
BENCHMARK(BM_PreludeCompile<Import::ManyTypesManyImpls>)
    ->Name("BM_PreludeCompile/ManyTypesManyImpls");
BENCHMARK(BM_PreludeCompile<Import::NonTrivialAmount>)
    ->Name("BM_PreludeCompile/NonTrivialAmount");

}  // namespace
}  // namespace Carbon::Testing

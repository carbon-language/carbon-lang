// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <fstream>

#include "absl/random/random.h"
#include "common/filesystem.h"
#include "llvm/ADT/StringExtras.h"

namespace Carbon::Filesystem {
namespace {

// Alternative implementation strategies that we may want to compare performance
// between.
enum BenchmarkComparables {
  Carbon,
  Std,
};

constexpr llvm::StringLiteral Text =
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim "
    "veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea "
    "commodo consequat. Duis aute irure dolor in reprehenderit in voluptate "
    "velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint "
    "occaecat cupidatat non proident, sunt in culpa qui officia deserunt "
    "mollit anim id est laborum.";

static auto GetText(int length) -> std::string {
  std::string content;
  content.reserve(length);
  while (static_cast<int>(content.size()) < length) {
    content += Text.substr(0, length - content.size());
  }
  CARBON_CHECK(static_cast<int>(content.size()) == length);
  return content;
}

struct BenchContext {
  RemovingDir tmpdir;
  absl::BitGen rng;

  BenchContext() : tmpdir(std::move(*MakeTmpDir())) {
    auto f = tmpdir.OpenWriteOnly("file", CreationOptions::CreateNew);
    CARBON_CHECK(f.ok(), "{0}", f.error());
  }

  auto CreateTree(std::filesystem::path base, int entries, int entries_per_dir)
      -> void {
    CARBON_CHECK(entries >= 1);
    CARBON_CHECK(entries_per_dir >= 1);
    int num_subdirs = std::max<int>(entries_per_dir / 2, 1);
    struct DirStackEntry {
      Dir dir;
      int num_entries;
      int subdir_count;
    };
    llvm::SmallVector<DirStackEntry> dir_stack;
    auto d = tmpdir.OpenDir(base, CreationOptions::CreateNew);
    CARBON_CHECK(d.ok(), "{0}", d.error());
    dir_stack.push_back({std::move(*d), entries, 0});

    while (!dir_stack.empty()) {
      auto& [dir, num_entries, subdir_count] = dir_stack.back();

      int entries_per_subdir = ((num_entries - entries_per_dir) / num_subdirs);
      CARBON_CHECK(entries_per_subdir < num_entries);
      if (entries_per_subdir >= entries_per_dir && subdir_count < num_subdirs) {
        auto name = llvm::formatv("dir_{0}", subdir_count).str();
        auto subdir = dir.OpenDir(name, CreationOptions::CreateNew);
        CARBON_CHECK(subdir.ok(), "{0}", subdir.error());
        ++subdir_count;

        // Note we have to continue after `push_back` as this will invalidate
        // the current references.
        dir_stack.push_back({std::move(*subdir), entries_per_subdir, 0});
        continue;
      }

      int num_files = entries_per_dir - subdir_count;
      CARBON_CHECK(num_files >= 0);

      for (int i = 0; i < num_files; ++i) {
        auto name = llvm::formatv("file_{0}", i).str();
        auto f = dir.OpenWriteOnly(name, CreationOptions::CreateNew);
        CARBON_CHECK(f.ok(), "{0}", f.error());
        auto close_result = std::move(*f).Close();
        CARBON_CHECK(close_result.ok(), "{0}", close_result.error());
      }
      dir_stack.pop_back();
    }
  }
};

template <BenchmarkComparables Comp>
auto BM_OpenClose(benchmark::State& state) -> void {
  BenchContext context;
  for (auto _ : state) {
    if constexpr (Comp == Carbon) {
      auto f = context.tmpdir.OpenReadOnly("file");
      CARBON_CHECK(f.ok(), "{0}", f.error());
      auto close_result = std::move(*f).Close();
      CARBON_CHECK(close_result.ok(), "{0}", close_result.error());
    } else if constexpr (Comp == Std) {
      std::ifstream f(context.tmpdir.abs_path() / "file");
      CARBON_CHECK(f.is_open());
    } else {
      static_assert(false, "Invalid benchmark comparable");
    }
  }
}
BENCHMARK(BM_OpenClose<Carbon>)->UseRealTime();
BENCHMARK(BM_OpenClose<Std>)->UseRealTime();

template <BenchmarkComparables Comp>
auto BM_CreateRemove(benchmark::State& state) -> void {
  BenchContext context;
  std::string filename = "new_file";
  for (auto _ : state) {
    if constexpr (Comp == Carbon) {
      // Create the file by opening it.
      auto f =
          context.tmpdir.OpenWriteOnly(filename, CreationOptions::CreateNew);
      CARBON_CHECK(f.ok(), "{0}", f.error());
      // Close it right away.
      auto close_result = std::move(*f).Close();
      CARBON_CHECK(close_result.ok(), "{0}", close_result.error());
      // Remove it.
      auto remove_result = context.tmpdir.Unlink(filename);
      CARBON_CHECK(remove_result.ok(), "{0}", remove_result.error());
    } else if constexpr (Comp == Std) {
      auto path = context.tmpdir.abs_path() / filename;
      // Create the file by opening it.
      std::ofstream f(path);
      CARBON_CHECK(f.is_open());
      // Close it right away.
      f.close();
      // Remove it.
      std::error_code ec;
      std::filesystem::remove(path, ec);
      CARBON_CHECK(!ec, "{0}", ec.message());
    } else {
      static_assert(false, "Invalid benchmark comparable");
    }
  }
}
BENCHMARK(BM_CreateRemove<Carbon>)->UseRealTime();
BENCHMARK(BM_CreateRemove<Std>)->UseRealTime();

template <BenchmarkComparables Comp>
auto BM_Read(benchmark::State& state) -> void {
  BenchContext context;
  int length = state.range(0);
  std::string content = GetText(length);
  {
    auto f =
        context.tmpdir.OpenWriteOnly("file", CreationOptions::CreateAlways);
    CARBON_CHECK(f.ok(), "{0}", f.error());
    auto write_result = f->WriteFromString(content);
    CARBON_CHECK(write_result.ok(), "{0}", write_result.error());
    auto close_result = std::move(*f).Close();
    CARBON_CHECK(close_result.ok(), "{0}", close_result.error());
  }
  for (auto _ : state) {
    if constexpr (Comp == Carbon) {
      auto f = context.tmpdir.OpenReadOnly("file");
      CARBON_CHECK(f.ok(), "{0}", f.error());
      auto read_result = f->ReadToString();
      CARBON_CHECK(read_result.ok(), "{0}", read_result.error());
      benchmark::DoNotOptimize(*read_result);
      auto close_result = std::move(*f).Close();
      CARBON_CHECK(close_result.ok(), "{0}", close_result.error());
    } else if constexpr (Comp == Std) {
      std::ifstream f(context.tmpdir.abs_path() / "file", std::ios::binary);
      CARBON_CHECK(f.is_open());
      // This may be a somewhat surprising implementation, but benchmarking
      // against several other ways of reading the file with `std::ifstream` all
      // have the same or worse performance.
      std::string read_content((std::istreambuf_iterator<char>(f)),
                               (std::istreambuf_iterator<char>()));
      benchmark::DoNotOptimize(read_content);
    } else {
      static_assert(false, "Invalid benchmark comparable");
    }
  }
}
BENCHMARK(BM_Read<Carbon>)->Range(4, 1024LL * 1024)->UseRealTime();
BENCHMARK(BM_Read<Std>)->Range(4, 1024LL * 1024)->UseRealTime();

template <BenchmarkComparables Comp>
auto BM_Write(benchmark::State& state) -> void {
  BenchContext context;
  std::string content;
  int length = state.range(0);
  content.reserve(length);
  while (static_cast<int>(content.size()) < length) {
    content += Text.substr(0, length - content.size());
  }
  CARBON_CHECK(static_cast<int>(content.size()) == length);
  for (auto _ : state) {
    if constexpr (Comp == Carbon) {
      auto f =
          context.tmpdir.OpenWriteOnly("file", CreationOptions::CreateAlways);
      CARBON_CHECK(f.ok(), "{0}", f.error());
      auto write_result = f->WriteFromString(content);
      CARBON_CHECK(write_result.ok(), "{0}", write_result.error());
      auto close_result = std::move(*f).Close();
      CARBON_CHECK(close_result.ok(), "{0}", close_result.error());
    } else if constexpr (Comp == Std) {
      std::ofstream f(context.tmpdir.abs_path() / "file", std::ios::binary);
      CARBON_CHECK(f.is_open());
      f.write(content.data(), content.length());
    } else {
      static_assert(false, "Invalid benchmark comparable");
    }
  }
}
BENCHMARK(BM_Write<Carbon>)->Range(4, 1024LL * 1024)->UseRealTime();
BENCHMARK(BM_Write<Std>)->Range(4, 1024LL * 1024)->UseRealTime();

template <BenchmarkComparables Comp>
auto BM_RemoveTree(benchmark::State& state) -> void {
  BenchContext context;
  int entries = state.range(0);
  int depth = state.range(1);

  // Configure our batch size based on the number of entries. Creating large
  // numbers of entries in the filesystem can cause problems, and is also very
  // slow. We don't need that much accuracy once the trees get large.
  int batch_size = entries <= 1024 ? 10 : entries <= (32 * 1024) ? 5 : 1;

  while (state.KeepRunningBatch(batch_size)) {
    state.PauseTiming();
    for (int i = 0; i < batch_size; ++i) {
      context.CreateTree(llvm::formatv("tree_{0}", i).str(), entries, depth);
    }
    state.ResumeTiming();

    for (int i = 0; i < batch_size; ++i) {
      std::string tree = llvm::formatv("tree_{0}", i).str();
      if constexpr (Comp == Carbon) {
        auto rmdir_result = context.tmpdir.Rmtree(tree);
        CARBON_CHECK(rmdir_result.ok(), "{0}", rmdir_result.error());
      } else if constexpr (Comp == Std) {
        std::error_code ec;
        std::filesystem::remove_all(context.tmpdir.abs_path() / tree, ec);
        CARBON_CHECK(!ec, "{0}", ec.message());
      } else {
        static_assert(false, "Invalid benchmark comparable");
      }
    }
  }
}
BENCHMARK(BM_RemoveTree<Carbon>)
    ->Ranges({{1, 256}, {1, 32}})
    ->Ranges({{2 * 1024, 256 * 1024}, {512, 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();
BENCHMARK(BM_RemoveTree<Std>)
    ->Ranges({{1, 256}, {1, 32}})
    ->Ranges({{2 * 1024, 256 * 1024}, {512, 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

template <BenchmarkComparables Comp>
auto BM_CreateDirectories(benchmark::State& state) -> void {
  BenchContext context;
  int depth = state.range(0);
  int existing_depth = state.range(1);
  CARBON_CHECK(existing_depth <= depth);
  CARBON_CHECK(depth > 0);

  // Use a batch size of 10 to get avoid completely swamping the measurements
  // with overhead from creating existing directories and cleaning up.
  constexpr int BatchSize = 10;

  // Pre-build both the paths and the existing paths. Note that we use
  // relatively short paths here, which if anything makes the benefits of the
  // Carbon library smaller.
  llvm::SmallVector<std::string> paths;
  llvm::SmallVector<std::string> existing_paths;
  for (int i = 0; i < BatchSize; ++i) {
    RawStringOstream path;
    llvm::ListSeparator sep("/");
    for (int j = 0; j < existing_depth; ++j) {
      path << sep << "exists_" << (j == 0 ? i : j);
    }
    existing_paths.push_back(path.TakeStr());
    path << existing_paths.back();
    for (int k = existing_depth; k < depth; ++k) {
      path << sep << "dir_" << (k == 0 ? i : k);
    }
    paths.push_back(path.TakeStr());
  }

  while (state.KeepRunningBatch(BatchSize)) {
    state.PauseTiming();
    for (int i = 0; i < BatchSize; ++i) {
      if (existing_depth > 0) {
        auto result = context.tmpdir.CreateDirectories(existing_paths[i]);
        CARBON_CHECK(result.ok(), "{0}", result.error());
      }
    }
    state.ResumeTiming();

    for (int i = 0; i < BatchSize; ++i) {
      if constexpr (Comp == Carbon) {
        auto result = context.tmpdir.CreateDirectories(paths[i]);
        CARBON_CHECK(result.ok(), "Failed to create '{0}': {1}", paths[i],
                     result.error());

        // Create a file in the provided directory. This adds some baseline
        // overhead but matches the realistic use case and ensures that there
        // isn't some laziness that makes just creating a directory have an
        // unusually low cost.
        auto f = result->OpenWriteOnly("test", CreationOptions::CreateNew);
        CARBON_CHECK(f.ok(), "{0}", f.error());
        auto close_result = std::move(*f).Close();
        CARBON_CHECK(close_result.ok(), "{0}", close_result.error());
      } else if constexpr (Comp == Std) {
        std::filesystem::path path = context.tmpdir.abs_path() / paths[i];
        std::error_code ec;
        std::filesystem::create_directories(path, ec);
        CARBON_CHECK(!ec, "{0}", ec.message());

        // Create a file in the directory, similar to above. This has a (much)
        // bigger effect though because the C++ APIs don't open the created
        // directory, and so the creation cost of it can very much be hidden
        // from the benchmark if we don't use it. This also lets us see the
        // benefit of not needing to re-walk the path to create the file.
        std::ofstream f(path / "test");
        CARBON_CHECK(f.is_open());
        f.close();
      } else {
        static_assert(false, "Invalid benchmark comparable");
      }
    }

    state.PauseTiming();
    for (int i = 0; i < BatchSize; ++i) {
      auto result = context.tmpdir.Rmtree(
          llvm::formatv("{0}_{1}", existing_depth > 0 ? "exists" : "dir", i)
              .str());
      CARBON_CHECK(result.ok(), "{0}", result.error());
    }
    state.ResumeTiming();
  }
}
static auto CreateDirectoriesArgs(benchmark::internal::Benchmark* b) {
  // The first argument is the depth of directory to create. We mostly care
  // about reasonably small depths here. It must be >= 1 for there to be
  // something to benchmark. The second number is the depth of pre-existing
  // directories which can vary from 0 to equal to the depth to benchmark the
  // case of no new directory being needed.
  for (int i = 1; i <= 8; i *= 2) {
    b->Args({i, 0});
    for (int j = 1; j <= i; j *= 2) {
      b->Args({i, j});
    }
  }
}
BENCHMARK(BM_CreateDirectories<Carbon>)
    ->Apply(CreateDirectoriesArgs)
    ->UseRealTime();
BENCHMARK(BM_CreateDirectories<Std>)
    ->Apply(CreateDirectoriesArgs)
    ->UseRealTime();

}  // namespace
}  // namespace Carbon::Filesystem

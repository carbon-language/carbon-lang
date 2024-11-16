// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_MEM_USAGE_H_
#define CARBON_TOOLCHAIN_BASE_MEM_USAGE_H_

#include <cstdint>

#include "common/map.h"
#include "common/set.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/base/yaml.h"

namespace Carbon {

// Helps track memory usage for a compile.
//
// Users will mix `Add` and `Collect` calls, using `ConcatLabel` to label
// allocation sources. Typically we'll collect stats for growable, potentially
// large data types (such as `SmallVector`), ignoring small fixed-size members
// (such as pointers or `int32_t`).
//
// For example:
//
//   auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
//       -> void {
//     // Explicit tracking.
//     mem_usage.Add(MemUsage::ConcatLabel(label, "data_"), data_.used_bytes(),
//                   data_.reserved_bytes());
//     // Common library types like `Map` and `llvm::SmallVector` have
//     // type-specific support.
//     mem_usage.Add(MemUsage::Concat(label, "array_"), array_);
//     // Implementing `CollectMemUsage` allows use with the same interface.
//     mem_usage.Collect(MemUsage::Concat(label, "obj_"), obj_);
//   }
class MemUsage {
 public:
  // Adds tracking for used and reserved bytes, paired with the given label.
  auto Add(std::string label, int64_t used_bytes, int64_t reserved_bytes)
      -> void {
    mem_usage_.push_back({.label = std::move(label),
                          .used_bytes = used_bytes,
                          .reserved_bytes = reserved_bytes});
  }

  // Adds memory usage for a `llvm::BumpPtrAllocator`.
  auto Collect(std::string label, const llvm::BumpPtrAllocator& allocator)
      -> void {
    Add(std::move(label), allocator.getBytesAllocated(),
        allocator.getTotalMemory());
  }

  // Adds memory usage for a `Map`.
  template <typename KeyT, typename ValueT, ssize_t SmallSize,
            typename KeyContextT>
  auto Collect(std::string label,
               const Map<KeyT, ValueT, SmallSize, KeyContextT>& map,
               KeyContextT key_context = KeyContextT()) -> void {
    // These don't track used bytes, so we set the same value for used and
    // reserved bytes.
    auto bytes = map.ComputeMetrics(key_context).storage_bytes;
    Add(std::move(label), bytes, bytes);
  }

  // Adds memory usage for a `Set`.
  template <typename KeyT, ssize_t SmallSize, typename KeyContextT>
  auto Collect(std::string label, const Set<KeyT, SmallSize, KeyContextT>& set,
               KeyContextT key_context = KeyContextT()) -> void {
    // These don't track used bytes, so we set the same value for used and
    // reserved bytes.
    auto bytes = set.ComputeMetrics(key_context).storage_bytes;
    Add(std::move(label), bytes, bytes);
  }

  // Adds memory usage of an array's data. This ignores the possible overhead of
  // a SmallVector's in-place storage; if it's used, it's going to be tiny
  // relative to scaling memory costs.
  //
  // This uses SmallVector in order to get proper inference for T, which
  // ArrayRef misses.
  template <typename T>
  auto Collect(std::string label, const llvm::SmallVectorImpl<T>& array)
      -> void {
    Add(std::move(label), array.size_in_bytes(), array.capacity_in_bytes());
  }

  // Adds memory usage for an object that provides `CollectMemUsage`.
  //
  // The expected signature of `CollectMemUsage` is above, in MemUsage class
  // comments.
  template <typename T>
    requires requires(MemUsage& mem_usage, llvm::StringRef label,
                      const T& arg) {
      { arg.CollectMemUsage(mem_usage, label) };
    }
  auto Collect(std::string label, const T& arg) -> void {
    arg.CollectMemUsage(*this, label);
  }

  // Constructs a label for memory usage, handling the `.` concatenation.
  // We don't expect much depth in labels per-call.
  static auto ConcatLabel(llvm::StringRef label, llvm::StringRef child_label)
      -> std::string {
    return llvm::formatv("{0}.{1}", label, child_label);
  }
  static auto ConcatLabel(llvm::StringRef label, llvm::StringRef child_label1,
                          llvm::StringRef child_label2) -> std::string {
    return llvm::formatv("{0}.{1}.{2}", label, child_label1, child_label2);
  }

  auto OutputYaml(llvm::StringRef filename) const -> Yaml::OutputMapping {
    // Explicitly copy the filename.
    return Yaml::OutputMapping([&, filename](Yaml::OutputMapping::Map map) {
      map.Add("filename", filename);
      int64_t total_used = 0;
      int64_t total_reserved = 0;
      for (const auto& entry : mem_usage_) {
        total_used += entry.used_bytes;
        total_reserved += entry.reserved_bytes;
        map.Add(entry.label,
                Yaml::OutputMapping([&](Yaml::OutputMapping::Map byte_map) {
                  byte_map.Add("used_bytes", entry.used_bytes);
                  byte_map.Add("reserved_bytes", entry.reserved_bytes);
                }));
      }
      map.Add("Total",
              Yaml::OutputMapping([&](Yaml::OutputMapping::Map byte_map) {
                byte_map.Add("used_bytes", total_used);
                byte_map.Add("reserved_bytes", total_reserved);
              }));
    });
  }

 private:
  // Memory usage for a specific label.
  struct Entry {
    std::string label;
    int64_t used_bytes;
    int64_t reserved_bytes;
  };

  // The accumulated data on memory usage.
  llvm::SmallVector<Entry> mem_usage_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_MEM_USAGE_H_

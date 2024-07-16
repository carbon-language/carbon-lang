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

// Types supporting memory usage tracking should define a method:
//
//   // Collects memory usage of members.
//   auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
//       -> void;
//
// The label should be concatenated with any child labels using MemUsageLabel in
// order to reflect allocation structure.
//
// The arguments for AddMemUsageFn are the label and byte size. It should be
// called once per tracked size.
class MemUsage {
 public:
  auto Add(std::string label, int64_t used, int64_t reserved) -> void {
    mem_usage_.push_back({std::move(label), used, reserved});
  }

  auto Add(std::string label, const llvm::BumpPtrAllocator& allocator) -> void {
    mem_usage_.push_back({std::move(label), allocator.getBytesAllocated(),
                          allocator.getTotalMemory()});
  }

  template <typename KeyT, typename ValueT, ssize_t SmallSize,
            typename KeyContextT>
  auto Add(std::string label, Map<KeyT, ValueT, SmallSize, KeyContextT> map,
           KeyContextT key_context = KeyContextT()) -> void {
    auto bytes = map.ComputeMetrics(key_context).storage_bytes;
    mem_usage_.push_back({std::move(label), bytes, bytes});
  }

  template <typename KeyT, ssize_t SmallSize, typename KeyContextT>
  auto Add(std::string label, Set<KeyT, SmallSize, KeyContextT> set,
           KeyContextT key_context = KeyContextT()) -> void {
    auto bytes = set.ComputeMetrics(key_context).storage_bytes;
    mem_usage_.push_back({std::move(label), bytes, bytes});
  }

  // Adds memory usage of an array's data. This ignores the possible overhead of
  // a SmallVector's in-place storage; if it's used, it's going to be tiny
  // relative to scaling memory costs.
  //
  // This uses SmallVector in order to get proper inference for T, which
  // ArrayRef misses.
  template <typename T, unsigned N>
  auto Add(std::string label, const llvm::SmallVector<T, N>& array) -> void {
    Add(std::move(label), array.size_in_bytes(), array.capacity_in_bytes());
  }

  template <typename T>
  auto Collect(llvm::StringRef label, const T& arg) -> void {
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
      for (auto [label, used, reserved] : mem_usage_) {
        total_used += used;
        total_reserved += reserved;
        map.Add(label,
                Yaml::OutputMapping([&](Yaml::OutputMapping::Map byte_map) {
                  byte_map.Add("used", used);
                  byte_map.Add("reserved", reserved);
                }));
      }
      map.Add("Total",
              Yaml::OutputMapping([&](Yaml::OutputMapping::Map byte_map) {
                byte_map.Add("used", total_used);
                byte_map.Add("reserved", total_reserved);
              }));
    });
  }

 private:
  llvm::SmallVector<std::tuple<std::string, int64_t, int64_t>> mem_usage_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_MEM_USAGE_H_

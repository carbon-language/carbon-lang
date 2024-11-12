// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_TIMINGS_H_
#define CARBON_TOOLCHAIN_BASE_TIMINGS_H_

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/base/yaml.h"

namespace Carbon {

// Helps track timings for a compile.
class Timings {
 public:
  // Adds tracking for nanoseconds, paired with the given label.
  auto Add(std::string label, std::chrono::nanoseconds nanoseconds) -> void {
    timings_.push_back({.label = std::move(label), .nanoseconds = nanoseconds});
  }

  auto OutputYaml(llvm::StringRef filename) const -> Yaml::OutputMapping {
    // Explicitly copy the filename.
    return Yaml::OutputMapping([&, filename](Yaml::OutputMapping::Map map) {
      map.Add("filename", filename);
      map.Add("nanoseconds",
              Yaml::OutputMapping([&](Yaml::OutputMapping::Map label_map) {
                std::chrono::nanoseconds total_nanoseconds(0);
                for (const auto& entry : timings_) {
                  total_nanoseconds += entry.nanoseconds;
                  label_map.Add(entry.label, static_cast<int64_t>(
                                                 entry.nanoseconds.count()));
                }
                label_map.Add("Total",
                              static_cast<int64_t>(total_nanoseconds.count()));
              }));
    });
  }

 private:
  // Timing for a specific label.
  struct Entry {
    std::string label;
    std::chrono::nanoseconds nanoseconds;
  };

  // The accumulated data on timings.
  llvm::SmallVector<Entry> timings_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_TIMINGS_H_

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_COMMON_GLOBALS_H_
#define CARBON_EXPLORER_COMMON_GLOBALS_H_

#include <optional>

#include "explorer/common/arena.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

// A container for shared 'explorer-wide' state like an arena.
class Globals {
 public:
  explicit Globals(
      Nonnull<Arena*> arena,
      std::optional<Nonnull<llvm::raw_ostream*>> trace_stream = std::nullopt)
      : arena_(arena), trace_stream_(trace_stream) {}

  auto arena() { return arena_; };
  auto trace_stream() { return trace_stream_; }

 private:
  Nonnull<Arena*> arena_;
  std::optional<Nonnull<llvm::raw_ostream*>> trace_stream_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_COMMON_GLOBALS_H_

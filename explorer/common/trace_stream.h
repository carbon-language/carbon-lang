// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_COMMON_TRACE_STREAM_H_
#define CARBON_EXPLORER_COMMON_TRACE_STREAM_H_

#include <optional>
#include <string>

#include "common/check.h"
#include "common/ostream.h"
#include "explorer/common/nonnull.h"

namespace Carbon {

// Encapsulates the trace stream so that we can cleanly disable tracing while
// the prelude is being processed. The prelude is expected to take a
// disproprotionate amount of time to log, so we try to avoid it.
//
// TODO: While the prelude is combined with the provided program as a single
// AST, the AST knows which declarations came from the prelude. When the prelude
// is fully treated as a separate file, we should be able to take a different
// approach where the caller explicitly toggles tracing when switching file
// contexts.
class TraceStream {
 public:
  // Returns true if tracing is currently enabled.
  auto is_enabled() const -> bool {
    return stream_.has_value() && !in_prelude_;
  }

  // Sets whether the prelude is being skipped.
  auto set_in_prelude(bool in_prelude) -> void { in_prelude_ = in_prelude; }

  // Sets the trace stream. This should only be called from the main.
  auto set_stream(Nonnull<llvm::raw_ostream*> stream) -> void {
    stream_ = stream;
  }

  // Returns the internal stream. Requires is_enabled.
  auto stream() const -> llvm::raw_ostream& {
    CARBON_CHECK(is_enabled());
    return **stream_;
  }

  // Outputs a trace message. Requires is_enabled.
  template <typename T>
  auto operator<<(T&& message) const -> llvm::raw_ostream& {
    CARBON_CHECK(is_enabled());
    **stream_ << message;
    return **stream_;
  }

 private:
  std::optional<Nonnull<llvm::raw_ostream*>> stream_;
  bool in_prelude_ = false;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_COMMON_TRACE_STREAM_H_

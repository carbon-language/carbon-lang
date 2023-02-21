// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_TRACE_STREAM_H_
#define CARBON_EXPLORER_INTERPRETER_TRACE_STREAM_H_

#include <optional>
#include <string>

#include "common/check.h"
#include "common/ostream.h"
#include "explorer/common/nonnull.h"
#include "explorer/common/source_location.h"

namespace Carbon {

// Encapsulates the trace stream so that we can cleanly disable tracing while
// the prelude is being processed. The prelude is expected to take a
// disproprotionate amount of time to log, so we try to avoid it.
//
// TODO: When the prelude is fully treated as a separate file, we may be able to
// take a different approach where the caller explicitly toggles tracing when
// switching file contexts.
class TraceStream {
 public:
  explicit TraceStream(std::string_view prelude_filename)
      : prelude_filename_(prelude_filename) {}

  // Returns true if tracing is currently enabled.
  auto is_enabled() const -> bool {
    return stream_.has_value() && !skipping_prelude_;
  }

  // Returns true if tracing is currently enabled. If the prelude is currently
  // being skipped and source_loc is a non-prelude file location, this will
  // mark the skip as done.
  auto is_enabled_at(SourceLocation source_loc) -> bool {
    if (!stream_.has_value()) {
      return false;
    }
    if (!skipping_prelude_) {
      return true;
    }
    if (!source_loc.filename().empty() &&
        source_loc.filename() != prelude_filename_) {
      **stream_ << "Finished prelude, resuming tracing at "
                << source_loc.filename() << "\n";
      skipping_prelude_ = false;
      return true;
    }
    return false;
  }

  // Sets whether the prelude is being skipped.
  auto set_skipping_prelude(bool skipping_prelude) -> void {
    if (skipping_prelude && is_enabled()) {
      **stream_ << "Skipping prelude traces...\n";
    }
    skipping_prelude_ = skipping_prelude;
  }

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
  std::string_view prelude_filename_;
  std::optional<Nonnull<llvm::raw_ostream*>> stream_;
  bool skipping_prelude_ = false;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_TRACE_STREAM_H_

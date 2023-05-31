// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_COMMON_TRACE_STREAM_H_
#define CARBON_EXPLORER_COMMON_TRACE_STREAM_H_

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "common/check.h"
#include "common/ostream.h"
#include "explorer/common/nonnull.h"

namespace Carbon {

// This enum class's members can be used to indicate the current phase.
enum class ProgramPhase {
  Other,
  SourceProgram,
  NameResolution,
  ControlFlowResolution,
  TypeChecking,
  UnformedVariableResolution,
  Declarations,
  Execution,
  Timing,
  All
};

// This enum class defines the context in which code is being added to the trace
// output.
enum class CodeContext { Main, Prelude, Import };

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
    return stream_.has_value() &&
           current_code_context_ != CodeContext::Prelude &&
           (allowed_phases_.count(current_phase_) > 0 ||
            allowed_phases_.count(ProgramPhase::All) > 0 ||
            current_phase_ == ProgramPhase::Other);
  }

  // Sets the trace stream. This should only be called from the main.
  auto set_stream(Nonnull<llvm::raw_ostream*> stream) -> void {
    stream_ = stream;
  }

  auto set_current_phase(ProgramPhase current_phase) -> void {
    current_phase_ = current_phase;
  }

  auto set_current_code_context(CodeContext current_code_context) -> void {
    current_code_context_ = current_code_context;
  }

  auto set_allowed_phases(std::vector<ProgramPhase> allowed_phases) {
    if (allowed_phases.empty()) {
      allowed_phases_.insert(ProgramPhase::Execution);
    } else {
      for (auto phase : allowed_phases) {
        allowed_phases_.insert(phase);
      }
    }
  }

  // Returns the internal stream. Requires is_enabled.
  auto stream() const -> llvm::raw_ostream& {
    CARBON_CHECK(is_enabled());
    CARBON_CHECK(stream_.has_value());
    return **stream_;
  }

  auto current_phase() const -> ProgramPhase { return current_phase_; }

  auto current_code_context() const -> CodeContext {
    return current_code_context_;
  }

  // Outputs a trace message. Requires is_enabled.
  template <typename T>
  auto operator<<(T&& message) const -> llvm::raw_ostream& {
    if (is_enabled()) {
      **stream_ << message;
    }
    return **stream_;
  }

 private:
  std::optional<Nonnull<llvm::raw_ostream*>> stream_;
  ProgramPhase current_phase_ = ProgramPhase::Other;
  std::unordered_set<ProgramPhase> allowed_phases_ = {ProgramPhase::Other};
  CodeContext current_code_context_ = CodeContext::Main;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_COMMON_TRACE_STREAM_H_

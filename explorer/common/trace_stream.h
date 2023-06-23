// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_COMMON_TRACE_STREAM_H_
#define CARBON_EXPLORER_COMMON_TRACE_STREAM_H_

#include <bitset>
#include <optional>
#include <string>
#include <vector>

#include "common/check.h"
#include "common/ostream.h"
#include "explorer/common/nonnull.h"

namespace Carbon {

class TraceStream;

// Enumerates the phases of the program used for tracing and controlling which
// program phases are included for tracing.
enum class ProgramPhase {
  Unknown,                     // Represents an unknown program phase.
  SourceProgram,               // Phase for the source program.
  NameResolution,              // Phase for name resolution.
  ControlFlowResolution,       // Phase for control flow resolution.
  TypeChecking,                // Phase for type checking.
  UnformedVariableResolution,  // Phase for unformed variables resolution.
  Declarations,                // Phase for printing declarations.
  Execution,                   // Phase for program execution.
  Timing,                      // Phase for timing logs.
  All,                         // Represents all program phases.
  Last = All                   // Last program phase indicator.
};

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
  // TODO: use current source location for file context based filtering instead
  // of just checking if current code context is Prelude.
  auto is_enabled() const -> bool {
    return stream_.has_value() && !in_prelude_ &&
           allowed_phases_[static_cast<int>(current_phase_)];
  }

  // Sets whether the prelude is being skipped.
  auto set_in_prelude(bool in_prelude) -> void { in_prelude_ = in_prelude; }

  // Sets the trace stream. This should only be called from the main.
  auto set_stream(Nonnull<llvm::raw_ostream*> stream) -> void {
    stream_ = stream;
  }

  auto set_current_phase(ProgramPhase current_phase) -> void {
    current_phase_ = current_phase;
  }

  auto set_allowed_phases(std::vector<ProgramPhase> allowed_phases_list) {
    if (allowed_phases_list.empty()) {
      allowed_phases_.set(static_cast<int>(ProgramPhase::Execution));
    } else {
      for (auto phase : allowed_phases_list) {
        if (phase == ProgramPhase::All) {
          allowed_phases_.set();
        } else {
          allowed_phases_.set(static_cast<int>(phase));
        }
      }
    }
  }

  // Returns the internal stream. Requires is_enabled.
  auto stream() const -> llvm::raw_ostream& {
    CARBON_CHECK(is_enabled() && stream_.has_value());
    return **stream_;
  }

  auto current_phase() const -> ProgramPhase { return current_phase_; }

  // Outputs a trace message. Requires is_enabled.
  template <typename T>
  auto operator<<(T&& message) const -> llvm::raw_ostream& {
    CARBON_CHECK(is_enabled());
    **stream_ << message;
    return **stream_;
  }

 private:
  bool in_prelude_ = false;
  std::optional<Nonnull<llvm::raw_ostream*>> stream_;
  ProgramPhase current_phase_ = ProgramPhase::Unknown;
  std::bitset<static_cast<int>(ProgramPhase::Last) + 1> allowed_phases_;
};

// This is a RAII class to set the current program phase, destructor invocation
// restores the previous phase
class SetProgramPhase {
 public:
  explicit SetProgramPhase(TraceStream& trace_stream,
                           ProgramPhase program_phase)
      : trace_stream_(trace_stream),
        initial_phase_(trace_stream.current_phase()) {
    trace_stream.set_current_phase(program_phase);
  }

  // This can be used for cases when current phase is set multiple times within
  // the same scope
  auto update_phase(ProgramPhase program_phase) -> void {
    trace_stream_.set_current_phase(program_phase);
  }

  ~SetProgramPhase() { trace_stream_.set_current_phase(initial_phase_); }

 private:
  TraceStream& trace_stream_;
  ProgramPhase initial_phase_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_COMMON_TRACE_STREAM_H_

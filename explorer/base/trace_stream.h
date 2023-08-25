// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_BASE_TRACE_STREAM_H_
#define CARBON_EXPLORER_BASE_TRACE_STREAM_H_

#include <bitset>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "common/check.h"
#include "common/ostream.h"
#include "explorer/base/nonnull.h"
#include "explorer/base/source_location.h"
#include "llvm/ADT/ArrayRef.h"

namespace Carbon {

class TraceStream;

// Enumerates the phases of the program used for tracing and controlling which
// program phases are included for tracing.
enum class ProgramPhase {
  SourceProgram,               // Phase for the source program.
  NameResolution,              // Phase for name resolution.
  ControlFlowResolution,       // Phase for control flow resolution.
  TypeChecking,                // Phase for type checking.
  UnformedVariableResolution,  // Phase for unformed variables resolution.
  Declarations,                // Phase for printing declarations.
  Execution,                   // Phase for program execution.
  Timing,                      // Phase for timing logs.
  Unknown,                     // Represents an unknown program phase.
  All,                         // Represents all program phases.
  Last = All                   // Last program phase indicator.
};

// Encapsulates the trace stream so that we can cleanly disable tracing while
// the prelude is being processed. The prelude is expected to take a
// disproprotionate amount of time to log, so we try to avoid it.
class TraceStream {
 public:
  explicit TraceStream() = default;

  // Returns true if tracing is currently enabled.
  auto is_enabled() const -> bool {
    return stream_.has_value() && !in_prelude_ &&
           allowed_phases_[static_cast<int>(current_phase_)] &&
           (!source_loc_ ||
            allowed_file_kinds_[static_cast<int>(source_loc_->file_kind())]);
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

  auto set_allowed_phases(llvm::ArrayRef<ProgramPhase> allowed_phases_list)
      -> void {
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

  auto set_allowed_file_kinds(llvm::ArrayRef<FileKind> kind_list) -> void {
    for (auto kind : kind_list) {
      allowed_file_kinds_.set(static_cast<int>(kind));
    }
  }

  auto set_source_loc(std::optional<SourceLocation> source_loc) -> void {
    source_loc_ = source_loc;
  }

  auto source_loc() const -> std::optional<SourceLocation> {
    return source_loc_;
  }

  // Returns the internal stream. Requires is_enabled.
  auto stream() const -> llvm::raw_ostream& {
    CARBON_CHECK(is_enabled() && stream_.has_value());
    return **stream_;
  }

  auto current_phase() const -> ProgramPhase { return current_phase_; }

  auto add_blank_lines(int num_blank_lines) const -> void {
    CARBON_CHECK(is_enabled() && stream_);
    if (!is_trace_empty_) {
      for (int i = 0; i < num_blank_lines; ++i) {
        **stream_ << "\n";
      }
    }
  }

  // Outputs a trace message. Requires is_enabled.
  template <typename T>
  auto operator<<(T&& message) const -> llvm::raw_ostream& {
    CARBON_CHECK(is_enabled() && stream_);
    if (is_trace_empty_) {
      is_trace_empty_ = false;
    }
    **stream_ << message;
    return **stream_;
  }

  // These functions can be used for adding line prefixes in the trace output.
  auto Indent() const -> llvm::raw_ostream& { return *this << "    "; }
  auto Start() const -> llvm::raw_ostream& { return *this << "->> "; }
  auto End() const -> llvm::raw_ostream& { return *this << "<<- "; }
  auto Call() const -> llvm::raw_ostream& { return *this << "-() "; }
  auto Match() const -> llvm::raw_ostream& { return *this << "=== "; }
  auto Result() const -> llvm::raw_ostream& { return *this << "==> "; }
  auto Add() const -> llvm::raw_ostream& { return *this << " +  "; }
  auto Remove() const -> llvm::raw_ostream& { return *this << " -  "; }
  auto Read() const -> llvm::raw_ostream& { return *this << "<-- "; }
  auto Write() const -> llvm::raw_ostream& { return *this << "--> "; }
  auto Allocate() const -> llvm::raw_ostream& { return *this << "++# "; }
  auto Deallocate() const -> llvm::raw_ostream& { return *this << "--# "; }
  auto Substitute() const -> llvm::raw_ostream& { return *this << "->+ "; }
  auto Push() const -> llvm::raw_ostream& { return *this << ">[] "; }
  auto Pop() const -> llvm::raw_ostream& { return *this << "<[] "; }
  auto Not() const -> llvm::raw_ostream& { return *this << "-!- "; }
  auto Skip() const -> llvm::raw_ostream& { return *this << ">>> "; }
  auto Source() const -> llvm::raw_ostream& {
    // add a blank line before prefix.
    add_blank_lines(1);
    return *this << "*** ";
  }

  // Format utility methods
  void Heading(std::string_view heading) const {
    add_blank_lines(2);
    const std::string_view stars = "* * * * * * * * * *";
    const std::string dashed_line(stars.size() * 2 + heading.size() + 4, '-');
    *this << stars << "  " << heading << "  " << stars << "\n"
          << dashed_line << "\n";
  }

  void SubHeading(std::string_view sub_heading) const {
    add_blank_lines(1);
    const std::string_view dashes = "- - - - -";
    const std::string dashed_line(dashes.size() * 2 + sub_heading.size() + 4,
                                  '-');
    *this << dashes << "  " << sub_heading << "  " << dashes << "\n"
          << dashed_line << "\n";
  }

 private:
  bool in_prelude_ = false;
  mutable bool is_trace_empty_ = true;
  ProgramPhase current_phase_ = ProgramPhase::Unknown;
  std::optional<SourceLocation> source_loc_ = std::nullopt;
  std::optional<Nonnull<llvm::raw_ostream*>> stream_;
  std::bitset<static_cast<int>(ProgramPhase::Last) + 1> allowed_phases_;
  std::bitset<static_cast<int>(FileKind::Last) + 1> allowed_file_kinds_;
};

// This is a RAII class to set the current program phase, destructor invocation
// restores the previous phase.
class SetProgramPhase {
 public:
  explicit SetProgramPhase(TraceStream& trace_stream,
                           ProgramPhase program_phase)
      : trace_stream_(trace_stream),
        initial_phase_(trace_stream.current_phase()) {
    trace_stream.set_current_phase(program_phase);
  }

  ~SetProgramPhase() { trace_stream_.set_current_phase(initial_phase_); }

  // This can be used for cases when current phase is set multiple times within
  // the same scope.
  auto update_phase(ProgramPhase program_phase) -> void {
    trace_stream_.set_current_phase(program_phase);
  }

 private:
  TraceStream& trace_stream_;
  ProgramPhase initial_phase_;
};

// This is a RAII class to set the source location in trace stream, destructor
// invocation restores the initial source location.
class SetFileContext {
 public:
  explicit SetFileContext(TraceStream& trace_stream,
                          std::optional<SourceLocation> source_loc)
      : trace_stream_(trace_stream),
        initial_source_loc_(trace_stream.source_loc()) {
    trace_stream_.set_source_loc(source_loc);
  }

  ~SetFileContext() { trace_stream_.set_source_loc(initial_source_loc_); }

  // This can be used for cases when source location needs to be updated
  // multiple times within the same scope.
  auto update_source_loc(std::optional<SourceLocation> source_loc) {
    trace_stream_.set_source_loc(source_loc);
  }

 private:
  TraceStream& trace_stream_;
  std::optional<SourceLocation> initial_source_loc_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_BASE_TRACE_STREAM_H_

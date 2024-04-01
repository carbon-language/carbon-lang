// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_CONSUMER_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_CONSUMER_H_

#include "common/ostream.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/diagnostics/diagnostic.h"

namespace Carbon {

// Receives diagnostics as they are emitted.
class DiagnosticConsumer {
 public:
  virtual ~DiagnosticConsumer() = default;

  // Handle a diagnostic.
  //
  // This relies on moves of the Diagnostic. At present, diagnostics are
  // allocated on the stack, so their lifetime is that of HandleDiagnostic.
  // However, SortingDiagnosticConsumer needs a longer lifetime, until all
  // diagnostics have been produced. As a consequence, it needs to either copy
  // or move the Diagnostic, and right now we're moving due to the overhead of
  // notes.
  //
  // At present, there is no persistent storage of diagnostics because IDEs
  // would be fine with diagnostics being printed immediately and discarded,
  // without SortingDiagnosticConsumer. If this becomes a performance issue, we
  // may want to investigate alternative ownership models that address both IDE
  // and CLI user needs.
  virtual auto HandleDiagnostic(Diagnostic diagnostic) -> void = 0;

  // Flushes any buffered input.
  virtual auto Flush() -> void {}
};

// A diagnostic consumer that prints to a stream.
class StreamDiagnosticConsumer : public DiagnosticConsumer {
 public:
  explicit StreamDiagnosticConsumer(llvm::raw_ostream& stream)
      : stream_(&stream) {}

  auto HandleDiagnostic(Diagnostic diagnostic) -> void override;
  auto Flush() -> void override { stream_->flush(); }

 private:
  auto Print(const DiagnosticMessage& message, llvm::StringRef prefix) -> void;

  llvm::raw_ostream* stream_;
  // Whethere we've printed a diagnostic. Used for printing separators.
  bool printed_diagnostic_ = false;
};

// Returns a diagnostic consumer instance that prints to stderr.
auto ConsoleDiagnosticConsumer() -> DiagnosticConsumer&;

// Diagnostic consumer adaptor that tracks whether any errors have been
// produced.
class ErrorTrackingDiagnosticConsumer : public DiagnosticConsumer {
 public:
  explicit ErrorTrackingDiagnosticConsumer(DiagnosticConsumer& next_consumer)
      : next_consumer_(&next_consumer) {}

  auto HandleDiagnostic(Diagnostic diagnostic) -> void override {
    seen_error_ |= diagnostic.level == DiagnosticLevel::Error;
    next_consumer_->HandleDiagnostic(std::move(diagnostic));
  }

  // Reset whether we've seen an error.
  auto Reset() -> void { seen_error_ = false; }

  // Returns whether we've seen an error since the last reset.
  auto seen_error() const -> bool { return seen_error_; }

 private:
  DiagnosticConsumer* next_consumer_;
  bool seen_error_ = false;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_CONSUMER_H_

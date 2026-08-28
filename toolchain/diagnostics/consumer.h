// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_CONSUMER_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_CONSUMER_H_

#include "common/ostream.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/diagnostics/diagnostic.h"

namespace Carbon::Diagnostics {

// Receives diagnostics as they are emitted.
class Consumer {
 public:
  virtual ~Consumer() = default;

  // Handle a diagnostic.
  //
  // This relies on moves of the Diagnostic. At present, diagnostics are
  // allocated on the stack, so their lifetime is that of HandleDiagnostic.
  // However, SortingConsumer needs a longer lifetime, until all
  // diagnostics have been produced. As a consequence, it needs to either copy
  // or move the Diagnostic, and right now we're moving due to the overhead of
  // labels.
  //
  // At present, there is no persistent storage of diagnostics because IDEs
  // would be fine with diagnostics being printed immediately and discarded,
  // without SortingConsumer. If this becomes a performance issue, we
  // may want to investigate alternative ownership models that address both IDE
  // and CLI user needs.
  virtual auto HandleDiagnostic(Diagnostic diagnostic) -> void = 0;

  // Flushes any buffered input.
  virtual auto Flush() -> void {}
};

// Returns a diagnostic consumer instance that prints to stderr. Defined
// alongside `StreamConsumer` in `stream_consumer.h`, which it renders through.
auto ConsoleConsumer() -> Consumer&;

// Diagnostic consumer adaptor that tracks whether any errors have been
// produced.
class ErrorTrackingConsumer : public Consumer {
 public:
  explicit ErrorTrackingConsumer(Consumer& next_consumer)
      : next_consumer_(&next_consumer) {}

  auto HandleDiagnostic(Diagnostic diagnostic) -> void override {
    seen_error_ |= diagnostic.level == Level::Error;
    next_consumer_->HandleDiagnostic(std::move(diagnostic));
  }

  // Reset whether we've seen an error.
  auto Reset() -> void { seen_error_ = false; }

  // Returns whether we've seen an error since the last reset.
  auto seen_error() const -> bool { return seen_error_; }

 private:
  Consumer* next_consumer_;
  bool seen_error_ = false;
};

}  // namespace Carbon::Diagnostics

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_CONSUMER_H_

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/diagnostic_consumer.h"

#include <algorithm>
#include <cstdint>

namespace Carbon {

auto StreamDiagnosticConsumer::HandleDiagnostic(Diagnostic diagnostic) -> void {
  if (printed_diagnostic_) {
    *stream_ << "\n";
  } else {
    printed_diagnostic_ = true;
  }

  for (const auto& message : diagnostic.messages) {
    message.loc.FormatLocation(*stream_);
    *stream_ << ": ";
    switch (message.level) {
      case DiagnosticLevel::Error:
        *stream_ << "ERROR: ";
        break;
      case DiagnosticLevel::Warning:
        *stream_ << "WARNING: ";
        break;
      case DiagnosticLevel::Note:
        break;
    }
    *stream_ << message.format_fn(message) << "\n";
    message.loc.FormatSnippet(*stream_);
  }
}

auto ConsoleDiagnosticConsumer() -> DiagnosticConsumer& {
  static auto* consumer = new StreamDiagnosticConsumer(llvm::errs());
  return *consumer;
}

}  // namespace Carbon

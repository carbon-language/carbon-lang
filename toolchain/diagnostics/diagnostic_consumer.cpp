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
    switch (message.level) {
      case DiagnosticLevel::Error:
        *stream_ << "error: ";
        break;
      case DiagnosticLevel::Warning:
        *stream_ << "warning: ";
        break;
      case DiagnosticLevel::Note:
        *stream_ << "note: ";
        break;
      case DiagnosticLevel::LocationInfo:
        break;
    }
    *stream_ << message.Format();
    if (include_diagnostic_kind_) {
      *stream_ << " [" << message.kind << "]";
    }
    *stream_ << "\n";
    // Don't include a snippet for location information to keep this diagnostic
    // more visually associated with the following diagnostic that it describes
    // and to better match C++ compilers.
    if (message.level != DiagnosticLevel::LocationInfo) {
      message.loc.FormatSnippet(*stream_);
    }
  }
}

auto ConsoleDiagnosticConsumer() -> DiagnosticConsumer& {
  static auto* consumer = new StreamDiagnosticConsumer(&llvm::errs());
  return *consumer;
}

}  // namespace Carbon

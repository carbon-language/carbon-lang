// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/diagnostic_consumer.h"

#include <algorithm>
#include <cstdint>

namespace Carbon {

auto StreamDiagnosticConsumer::HandleDiagnostic(Diagnostic diagnostic) -> void {
  for (const auto& message : diagnostic.messages) {
    *stream_ << message.location.filename;
    if (message.location.line_number > 0) {
      *stream_ << ":" << message.location.line_number;
      if (message.location.column_number > 0) {
        *stream_ << ":" << message.location.column_number;
      }
    }
    *stream_ << ": ";
    if (message.level == DiagnosticLevel::Error) {
      *stream_ << "ERROR: ";
    }
    *stream_ << message.format_fn(message) << "\n";
    if (message.location.column_number > 0) {
      *stream_ << message.location.line << "\n";
      stream_->indent(message.location.column_number - 1);
      *stream_ << "^";
      int underline_length = std::max(0, message.location.length - 1);
      // We want to ensure that we don't underline past the end of the line in
      // case of a multiline token.
      // TODO: revisit this once we can reference multiple ranges on multiple
      // lines in a single diagnostic message.
      underline_length = std::min(
          underline_length, static_cast<int32_t>(message.location.line.size()) -
                                message.location.column_number);
      for (int i = 0; i < underline_length; ++i) {
        *stream_ << "~";
      }
      *stream_ << "\n";
    }
  }
}

auto ConsoleDiagnosticConsumer() -> DiagnosticConsumer& {
  static auto* consumer = new StreamDiagnosticConsumer(llvm::errs());
  return *consumer;
}

}  // namespace Carbon

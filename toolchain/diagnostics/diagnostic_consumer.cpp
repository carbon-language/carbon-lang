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
    *stream_ << message.loc.filename;
    if (message.loc.line_number > 0) {
      *stream_ << ":" << message.loc.line_number;
      if (message.loc.column_number > 0) {
        *stream_ << ":" << message.loc.column_number;
      }
    }
    *stream_ << ": ";
    if (message.level == DiagnosticLevel::Error) {
      *stream_ << "ERROR: ";
    }
    *stream_ << message.format_fn(message) << "\n";
    if (message.loc.column_number > 0) {
      *stream_ << message.loc.line << "\n";
      stream_->indent(message.loc.column_number - 1);
      *stream_ << "^";
      int underline_length = std::max(0, message.loc.length - 1);
      // We want to ensure that we don't underline past the end of the line in
      // case of a multiline token.
      // TODO: revisit this once we can reference multiple ranges on multiple
      // lines in a single diagnostic message.
      underline_length = std::min(
          underline_length, static_cast<int32_t>(message.loc.line.size()) -
                                message.loc.column_number);
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

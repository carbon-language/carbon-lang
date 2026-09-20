// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/stream_consumer.h"

#include <utility>

#include "llvm/ADT/SmallString.h"

namespace Carbon::Diagnostics {

auto StreamConsumer::HandleDiagnostic(Diagnostic diagnostic) -> void {
  // The whole diagnostic is assembled before any of it is written, so that it
  // reaches the stream as one write and can't be interleaved with another
  // writer partway through.
  llvm::SmallString<256> bytes;
  if (std::exchange(printed_diagnostic_, true)) {
    bytes.push_back('\n');
  }
  renderer_.Render(bytes, diagnostic);
  *stream_ << bytes;
}

auto ConsoleConsumer() -> Consumer& {
  static auto* consumer = new StreamConsumer(&llvm::errs());
  return *consumer;
}

}  // namespace Carbon::Diagnostics

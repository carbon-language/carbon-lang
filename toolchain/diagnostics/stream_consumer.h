// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_STREAM_CONSUMER_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_STREAM_CONSUMER_H_

#include "common/ostream.h"
#include "common/terminal/capabilities.h"
#include "toolchain/diagnostics/consumer.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/diagnostics/renderer.h"

namespace Carbon::Diagnostics {

// A diagnostic consumer that prints to a stream.
//
// Rendering is plain ASCII until `set_capabilities` says what the terminal
// behind `stream` can do. See `Diagnostics::Renderer`.
//
// This is separate from `consumer.h` because it holds a `Renderer`, which
// reaches the terminal library and through it `<filesystem>`. `Consumer` itself
// is named by every layer that emits a diagnostic, so what it includes is what
// most of the toolchain parses.
class StreamConsumer : public Consumer {
 public:
  explicit StreamConsumer(llvm::raw_ostream* stream) : stream_(stream) {}

  auto HandleDiagnostic(Diagnostic diagnostic) -> void override;
  auto Flush() -> void override { stream_->flush(); }

  auto set_capabilities(const Terminal::Capabilities& capabilities) -> void {
    renderer_.set_capabilities(capabilities);
  }
  auto set_include_diagnostic_kind(bool value) -> void {
    renderer_.set_include_kind(value);
  }
  auto set_diagnostic_snippets(bool value) -> void {
    renderer_.set_snippets(value);
  }

 private:
  llvm::raw_ostream* stream_;

  Renderer renderer_;

  // Whether we've printed a diagnostic. Used for printing separators.
  bool printed_diagnostic_ = false;
};

}  // namespace Carbon::Diagnostics

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_STREAM_CONSUMER_H_

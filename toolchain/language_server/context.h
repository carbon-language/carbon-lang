// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LANGUAGE_SERVER_CONTEXT_H_
#define CARBON_TOOLCHAIN_LANGUAGE_SERVER_CONTEXT_H_

#include <string>

#include "common/map.h"
#include "toolchain/diagnostics/diagnostic_consumer.h"
#include "toolchain/diagnostics/file_diagnostics.h"

namespace Carbon::LanguageServer {

// Context for LSP call handling.
class Context {
 public:
  // `emitter` is required.
  explicit Context(DiagnosticConsumer* consumer) : emitter_(consumer) {}

  auto emitter() -> FileDiagnosticEmitter& { return emitter_; }

  auto files() -> Map<std::string, std::string>& { return files_; }

 private:
  FileDiagnosticEmitter emitter_;

  // Content of files managed by the language client.
  Map<std::string, std::string> files_;
};

}  // namespace Carbon::LanguageServer

#endif  // CARBON_TOOLCHAIN_LANGUAGE_SERVER_CONTEXT_H_

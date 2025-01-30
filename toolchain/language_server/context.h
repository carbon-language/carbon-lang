// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LANGUAGE_SERVER_CONTEXT_H_
#define CARBON_TOOLCHAIN_LANGUAGE_SERVER_CONTEXT_H_

#include <string>

#include "common/map.h"
#include "toolchain/diagnostics/diagnostic_consumer.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon::LanguageServer {

// Context for LSP call handling.
class Context {
 public:
  // `consumer` is required.
  explicit Context(DiagnosticConsumer* consumer) : no_loc_emitter_(consumer) {}

  auto no_loc_emitter() -> NoLocDiagnosticEmitter& { return no_loc_emitter_; }

  auto files() -> Map<std::string, std::string>& { return files_; }

 private:
  NoLocDiagnosticEmitter no_loc_emitter_;

  // Content of files managed by the language client.
  Map<std::string, std::string> files_;
};

}  // namespace Carbon::LanguageServer

#endif  // CARBON_TOOLCHAIN_LANGUAGE_SERVER_CONTEXT_H_

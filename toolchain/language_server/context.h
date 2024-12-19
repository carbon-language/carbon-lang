// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LANGUAGE_SERVER_CONTEXT_H_
#define CARBON_TOOLCHAIN_LANGUAGE_SERVER_CONTEXT_H_

#include "clang-tools-extra/clangd/Protocol.h"
#include "clang-tools-extra/clangd/support/Function.h"
#include "common/map.h"

namespace Carbon::LanguageServer {

// Handles LSP calls. This is the main implementation.
class Context {
 public:
  // Updates the content of already-open documents.
  auto HandleDidChangeTextDocument(
      const clang::clangd::DidChangeTextDocumentParams& params) -> void;

  // Stores the content of newly-opened documents.
  auto HandleDidOpenTextDocument(
      const clang::clangd::DidOpenTextDocumentParams& params) -> void;

  // Tells the client what features are supported.
  auto HandleInitialize(const clang::clangd::NoParams& client_capabilities,
                        clang::clangd::Callback<llvm::json::Object> on_done)
      -> void;

  // Provides information about document symbols.
  auto HandleDocumentSymbol(
      const clang::clangd::DocumentSymbolParams& params,
      clang::clangd::Callback<std::vector<clang::clangd::DocumentSymbol>>
          on_done) -> void;

 private:
  // Content of files managed by the language client.
  Map<std::string, std::string> files_;
};

}  // namespace Carbon::LanguageServer

#endif  // CARBON_TOOLCHAIN_LANGUAGE_SERVER_CONTEXT_H_

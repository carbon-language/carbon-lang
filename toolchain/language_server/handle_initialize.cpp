// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "toolchain/language_server/handle.h"

namespace Carbon::LanguageServer {

auto HandleInitialize(
    Context& /*context*/,
    const clang::clangd::NoParams& /*client_capabilities*/,
    llvm::function_ref<auto(llvm::Expected<llvm::json::Object>)->void> on_done)
    -> void {
  llvm::json::Object capabilities{
      {"declarationProvider", true},    {"definitionProvider", true},
      {"documentSymbolProvider", true}, {"hoverProvider", true},
      {"referencesProvider", true},     {"textDocumentSync", /*Incremental=*/2},
      {"typeDefinitionProvider", true}};
  llvm::json::Object reply{{"capabilities", std::move(capabilities)}};
  on_done(reply);
}

// Implements `initialized`:
// https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#initialized
auto HandleInitialized(Context& /*context*/,
                       const clang::clangd::NoParams& /*params*/) -> void {
  // Nothing to do, but every client sends this, so we handle it rather than
  // warning about an unsupported notification.
  // TODO: This is when we would use `client/registerCapability` for any
  // capabilities we want to register dynamically.
}

}  // namespace Carbon::LanguageServer

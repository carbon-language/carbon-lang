// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "toolchain/language_server/handle.h"

namespace Carbon::LanguageServer {

// Picks the encoding used to measure `Position::character` on the wire.
// Use UTF-8 if offered, since Carbon source is natively UTF-8. Otherwise fall
// back to UTF-16, to support VS Code and old LSP clients.
static auto NegotiatePositionEncoding(
    const clang::clangd::ClientCapabilities& capabilities)
    -> clang::clangd::OffsetEncoding {
  if (capabilities.PositionEncodings &&
      llvm::is_contained(*capabilities.PositionEncodings,
                         clang::clangd::OffsetEncoding::UTF8)) {
    return clang::clangd::OffsetEncoding::UTF8;
  }
  return clang::clangd::OffsetEncoding::UTF16;
}

auto HandleInitialize(
    Context& context, const clang::clangd::InitializeParams& params,
    llvm::function_ref<auto(llvm::Expected<llvm::json::Object>)->void> on_done)
    -> void {
  auto encoding = NegotiatePositionEncoding(params.capabilities);
  context.SetPositionEncoding(encoding);

  llvm::json::Object capabilities{{"declarationProvider", true},
                                  {"definitionProvider", true},
                                  {"documentFormattingProvider", true},
                                  {"documentSymbolProvider", true},
                                  {"hoverProvider", true},
                                  {"positionEncoding", encoding},
                                  {"referencesProvider", true},
                                  {"textDocumentSync", /*Incremental=*/2},
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

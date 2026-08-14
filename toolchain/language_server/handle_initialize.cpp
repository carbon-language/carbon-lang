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

  llvm::json::Object capabilities{{"documentSymbolProvider", true},
                                  {"textDocumentSync", /*Incremental=*/2},
                                  {"positionEncoding", encoding}};
  llvm::json::Object reply{{"capabilities", std::move(capabilities)}};
  on_done(reply);
}

}  // namespace Carbon::LanguageServer

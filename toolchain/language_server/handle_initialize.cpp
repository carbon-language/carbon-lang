// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/language_server/handle.h"

namespace Carbon::LanguageServer {

auto HandleInitialize(
    Context& /*context*/,
    const clang::clangd::NoParams& /*client_capabilities*/,
    llvm::function_ref<void(llvm::Expected<llvm::json::Object>)> on_done)
    -> void {
  llvm::json::Object capabilities{{"documentSymbolProvider", true},
                                  {"textDocumentSync", /*Full=*/1}};
  llvm::json::Object reply{{"capabilities", std::move(capabilities)}};
  on_done(reply);
}

}  // namespace Carbon::LanguageServer

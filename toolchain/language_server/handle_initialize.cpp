// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/language_server/context.h"
#include "toolchain/language_server/handler_registry.h"

namespace Carbon::LanguageServer {

// Tells the client what features are supported.
static auto HandleInitialize(
    Context& /*context*/,
    const clang::clangd::NoParams& /*client_capabilities*/,
    llvm::function_ref<void(llvm::Expected<llvm::json::Object>)> on_done)
    -> void {
  llvm::json::Object capabilities{{"documentSymbolProvider", true},
                                  {"textDocumentSync", /*Full=*/1}};

  llvm::json::Object reply{{"capabilities", std::move(capabilities)}};
  on_done(reply);
}

static RegisterCallHandler<HandleInitialize> register_call("initialize", "");

}  // namespace Carbon::LanguageServer

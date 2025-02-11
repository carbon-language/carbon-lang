// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/language_server/handle.h"

namespace Carbon::LanguageServer {

auto HandleShutdown(
    Context& /*context*/,
    const clang::clangd::NoParams& /*client_capabilities*/,
    llvm::function_ref<auto(llvm::Expected<std::nullptr_t>)->void> on_done)
    -> void {
  // TODO: Track that `shutdown` was called, and:
  // - Warn on duplicate calls.
  // - Make `exit` return `1` if `shutdown` wasn't called.
  //   https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#exit
  // - Error on other post-`shutdown` calls.
  on_done(nullptr);
}

}  // namespace Carbon::LanguageServer

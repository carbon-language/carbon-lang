// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/language_server/handle.h"

namespace Carbon::LanguageServer {

auto HandleShutdown(
    Context& /*context*/,
    const clang::clangd::NoParams& /*client_capabilities*/,
    llvm::function_ref<void(llvm::Expected<std::nullptr_t>)> on_done) -> void {
  on_done(nullptr);
}

}  // namespace Carbon::LanguageServer

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/language_server/handle.h"

namespace Carbon::LanguageServer {

auto HandleDidOpenTextDocument(
    Context& context, const clang::clangd::DidOpenTextDocumentParams& params)
    -> void {
  context.files().Update(params.textDocument.uri.file(),
                         params.textDocument.text);
}

auto HandleDidChangeTextDocument(
    Context& context, const clang::clangd::DidChangeTextDocumentParams& params)
    -> void {
  // Full text is sent if full sync is specified in capabilities.
  CARBON_CHECK(params.contentChanges.size() == 1);
  context.files().Update(params.textDocument.uri.file(),
                         params.contentChanges[0].text);
}

}  // namespace Carbon::LanguageServer

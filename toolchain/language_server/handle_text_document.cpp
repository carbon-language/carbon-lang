// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/language_server/context.h"
#include "toolchain/language_server/handler_registry.h"

namespace Carbon::LanguageServer {

// Updates the content of already-open documents.
static auto HandleDidOpenTextDocument(
    Context& context, const clang::clangd::DidOpenTextDocumentParams& params)
    -> void {
  context.files().Update(params.textDocument.uri.file(),
                         params.textDocument.text);
}

// Stores the content of newly-opened documents.
static auto HandleDidChangeTextDocument(
    Context& context, const clang::clangd::DidChangeTextDocumentParams& params)
    -> void {
  // Full text is sent if full sync is specified in capabilities.
  CARBON_CHECK(params.contentChanges.size() == 1);
  context.files().Update(params.textDocument.uri.file(),
                         params.contentChanges[0].text);
}

static RegisterNotificationHandler<HandleDidChangeTextDocument>
    register_notification1("textDocument/didChange", "");
static RegisterNotificationHandler<HandleDidOpenTextDocument>
    register_notification2("textDocument/didOpen", "");

}  // namespace Carbon::LanguageServer

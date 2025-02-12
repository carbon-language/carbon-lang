// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/check.h"
#include "toolchain/language_server/handle.h"

namespace Carbon::LanguageServer {

auto HandleDidOpenTextDocument(
    Context& context, const clang::clangd::DidOpenTextDocumentParams& params)
    -> void {
  llvm::StringRef filename = params.textDocument.uri.file();
  if (!filename.ends_with(".carbon")) {
    // Ignore non-Carbon files.
    return;
  }

  auto insert_result = context.files().Insert(
      filename, [&] { return Context::File(params.textDocument.uri); });
  insert_result.value().SetText(context, params.textDocument.version,
                                params.textDocument.text);
  if (!insert_result.is_inserted()) {
    CARBON_DIAGNOSTIC(LanguageServerOpenDuplicateFile, Warning,
                      "duplicate open file request; updating content");
    context.file_emitter().Emit(filename, LanguageServerOpenDuplicateFile);
  }
}

auto HandleDidChangeTextDocument(
    Context& context, const clang::clangd::DidChangeTextDocumentParams& params)
    -> void {
  llvm::StringRef filename = params.textDocument.uri.file();
  if (!filename.ends_with(".carbon")) {
    // Ignore non-Carbon files.
    return;
  }

  if (auto* file = context.LookupFile(filename)) {
    file->ApplyChanges(context, params.textDocument.version,
                       params.contentChanges);
  }
}

auto HandleDidCloseTextDocument(
    Context& context, const clang::clangd::DidCloseTextDocumentParams& params)
    -> void {
  llvm::StringRef filename = params.textDocument.uri.file();
  if (!filename.ends_with(".carbon")) {
    // Ignore non-Carbon files.
    return;
  }

  if (!context.files().Erase(filename)) {
    CARBON_DIAGNOSTIC(LanguageServerCloseUnknownFile, Warning,
                      "tried closing unknown file; ignoring request");
    context.file_emitter().Emit(filename, LanguageServerCloseUnknownFile);
  }
}

}  // namespace Carbon::LanguageServer

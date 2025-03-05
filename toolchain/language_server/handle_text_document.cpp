// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/language_server/handle.h"

namespace Carbon::LanguageServer {

// Implements `textDocument/didOpen`:
// https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_didOpen
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

// Returns the result of moving from `cursor_index` past `line_count` lines.
static auto GetNthLineIndex(llvm::StringRef contents, size_t cursor_index,
                            size_t line_count) -> size_t {
  for ([[maybe_unused]] auto _ : llvm::seq(line_count)) {
    const size_t newline_index = contents.find('\n', cursor_index);
    CARBON_CHECK(newline_index != std::string::npos,
                 "Line number greater than number of lines in the file");
    cursor_index = newline_index + 1;
  }
  return cursor_index;
}

// Takes start and end positions and returns a tuple with start and end
// offsets. Positions are based on row and column numbers in the source
// code. We often need to know the offsets when modifying strings, so
// this function helps us calculate the offsets. It assumes that the start
// position comes before the end position.
static auto PositionToIndex(llvm::StringRef contents,
                            const clang::clangd::Position& start,
                            const clang::clangd::Position& end)
    -> std::tuple<size_t, size_t> {
  size_t start_index =
      GetNthLineIndex(contents, /*cursor_index=*/0, start.line);
  size_t end_index = GetNthLineIndex(contents, /*cursor_index=*/start_index,
                                     end.line - start.line);

  start_index += start.character;
  end_index += end.character;

  CARBON_CHECK(end_index <= contents.size(),
               "Position greater than source code size");

  return {start_index, end_index};
}

// LSP allows full and incremental document synchronization. It sends the
// entire source code when doing full sync. However, when doing incremental
// sync, it only sends the list of changes to be performed on the source
// code. It is necessary to apply changes in the order in which they are
// received. These changes can have one of 1) full document or 2) start
// position, end position, and the text which replaces original text between
// these start and end positions. If the range property is absent, then we
// do full sync. Otherwise, we calculate start and end offsets and replace
// the contents between them with the text we received in change.text.
static auto ApplyChanges(
    std::string& source,
    const std::vector<clang::clangd::TextDocumentContentChangeEvent>&
        content_changes) -> void {
  for (const auto& change : content_changes) {
    // If range is not present, then we replace entire text.
    if (!change.range) {
      source = change.text;
      continue;
    }

    // TODO: Use lexer line number data but avoid re-lexing multiple times.
    auto [start_index, end_index] =
        PositionToIndex(source, change.range->start, change.range->end);

    source.replace(start_index, end_index - start_index, change.text);
  }
}

// Implements `textDocument/didChange`:
// https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_didChange
auto HandleDidChangeTextDocument(
    Context& context, const clang::clangd::DidChangeTextDocumentParams& params)
    -> void {
  llvm::StringRef filename = params.textDocument.uri.file();
  if (!filename.ends_with(".carbon")) {
    // Ignore non-Carbon files.
    return;
  }

  if (auto* file = context.LookupFile(filename)) {
    // We copy the document to a new string, apply changes to the string, and
    // set the string as the new text.
    std::string source = file->text().str();
    ApplyChanges(source, params.contentChanges);
    file->SetText(context, params.textDocument.version, source);
  }
}

// Implements `textDocument/didClose`:
// https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_didClose
auto HandleDidCloseTextDocument(
    Context& context, const clang::clangd::DidCloseTextDocumentParams& params)
    -> void {
  llvm::StringRef filename = params.textDocument.uri.file();
  if (!filename.ends_with(".carbon")) {
    // Ignore non-Carbon files.
    return;
  }

  if (context.files().Erase(filename)) {
    // Clear diagnostics when the document closes. Otherwise, any diagnostics
    // will linger.
    context.PublishDiagnostics({.uri = params.textDocument.uri});
  } else {
    CARBON_DIAGNOSTIC(LanguageServerCloseUnknownFile, Warning,
                      "tried closing unknown file; ignoring request");
    context.file_emitter().Emit(filename, LanguageServerCloseUnknownFile);
  }
}

}  // namespace Carbon::LanguageServer

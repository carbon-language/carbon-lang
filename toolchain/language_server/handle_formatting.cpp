// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "toolchain/format/format.h"
#include "toolchain/language_server/handle.h"

namespace Carbon::LanguageServer {

// Converts the formatter's byte-offset replacements into LSP text edits against
// `text`. The replacements are ordered by offset and non-overlapping, so a
// single forward scan suffices to map each offset to a line/character position.
// Positions are in bytes, matching the rest of the server.
// TODO: LSP nominally uses UTF-16 code units for character positions; convert
// byte offsets accordingly once the server handles non-ASCII source uniformly.
static auto ReplacementsToTextEdits(llvm::StringRef text,
                                    llvm::ArrayRef<Format::Replacement> edits)
    -> std::vector<clang::clangd::TextEdit> {
  int line = 0;
  int line_start = 0;
  int scan = 0;
  auto position_at = [&](int32_t offset) -> clang::clangd::Position {
    for (; scan < offset; ++scan) {
      if (text[scan] == '\n') {
        ++line;
        line_start = scan + 1;
      }
    }
    return {.line = line, .character = offset - line_start};
  };

  std::vector<clang::clangd::TextEdit> text_edits;
  for (const Format::Replacement& edit : edits) {
    clang::clangd::Position start = position_at(edit.offset);
    clang::clangd::Position end = position_at(edit.offset + edit.length);
    text_edits.push_back(
        {.range = {.start = start, .end = end}, .newText = edit.text});
  }
  return text_edits;
}

auto HandleFormatting(
    Context& context, const clang::clangd::DocumentFormattingParams& params,
    llvm::function_ref<
        auto(llvm::Expected<std::vector<clang::clangd::TextEdit>>)->void>
        on_done) -> void {
  auto* file = context.LookupFile(params.textDocument.uri.file());
  if (!file) {
    // TODO: A call handler should reply even on failure (an error or an empty
    // result) rather than leave the request unanswered; this mirrors
    // `HandleDocumentSymbol`, and both should change together.
    return;
  }
  // Format best-effort: the parse tree is always structurally valid, so edits
  // are produced even when the document has errors (the return value, reporting
  // whether the input was error-free, is intentionally ignored here).
  llvm::SmallVector<Format::Replacement> edits;
  Format::FormatReplacements(file->tree_and_subtrees().tree(), edits);
  on_done(ReplacementsToTextEdits(file->text(), edits));
}

// Converts a 0-based, end-exclusive LSP range to the formatter's 1-based
// inclusive line range. An end at column 0 does not reach into its line, so
// that line is excluded (unless it is also the start line).
static auto LspRangeToLineRange(const clang::clangd::Range& range)
    -> Format::LineRange {
  int last_line = range.end.line;
  if (range.end.character == 0 && range.end.line > range.start.line) {
    --last_line;
  }
  return {.first_line = range.start.line + 1, .last_line = last_line + 1};
}

auto HandleRangeFormatting(
    Context& context,
    const clang::clangd::DocumentRangeFormattingParams& params,
    llvm::function_ref<
        auto(llvm::Expected<std::vector<clang::clangd::TextEdit>>)->void>
        on_done) -> void {
  auto* file = context.LookupFile(params.textDocument.uri.file());
  if (!file) {
    return;
  }
  Format::LineRange lines = LspRangeToLineRange(params.range);

  llvm::SmallVector<Format::Replacement> edits;
  Format::FormatReplacements(file->tree_and_subtrees().tree(), edits, lines);
  on_done(ReplacementsToTextEdits(file->text(), edits));
}

}  // namespace Carbon::LanguageServer

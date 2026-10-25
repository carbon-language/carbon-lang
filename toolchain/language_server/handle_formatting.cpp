// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <bit>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/format/format.h"
#include "toolchain/language_server/handle.h"

namespace Carbon::LanguageServer {

// Measures `Position::character` units in `slice` (within a single line) under
// the negotiated LSP position encoding.
static auto MeasureCharacterUnits(llvm::StringRef slice,
                                  clang::clangd::OffsetEncoding encoding)
    -> int {
  if (encoding == clang::clangd::OffsetEncoding::UTF8) {
    return static_cast<int>(slice.size());
  }
  int units = 0;
  for (size_t i = 0; i < slice.size();) {
    uint8_t byte = static_cast<uint8_t>(slice[i]);
    if ((byte & 0x80) == 0) {
      ++units;
      ++i;
      continue;
    }
    int utf8_len = std::countl_one(byte);
    if (utf8_len < 2 || utf8_len > 4 || i + utf8_len > slice.size()) {
      ++units;
      ++i;
      continue;
    }
    bool valid_continuation = true;
    for (int k = 1; k < utf8_len; ++k) {
      if ((static_cast<uint8_t>(slice[i + k]) & 0xC0) != 0x80) {
        valid_continuation = false;
        break;
      }
    }
    if (!valid_continuation) {
      ++units;
      ++i;
      continue;
    }
    // Code points >= U+10000 (4-byte UTF-8 sequences) require a 2-unit UTF-16
    // surrogate pair; BMP code points require 1 UTF-16 code unit.
    units += (encoding == clang::clangd::OffsetEncoding::UTF16 && utf8_len == 4)
                 ? 2
                 : 1;
    i += utf8_len;
  }
  return units;
}

// Converts ordered, non-overlapping byte-offset replacements into LSP text
// edits in a single forward scan over `text`.
static auto ReplacementsToTextEdits(llvm::StringRef text,
                                    clang::clangd::OffsetEncoding encoding,
                                    llvm::ArrayRef<Format::Replacement> edits)
    -> std::vector<clang::clangd::TextEdit> {
  int line = 0;
  int character = 0;
  int32_t scan = 0;
  auto position_at = [&](int32_t offset) -> clang::clangd::Position {
    while (scan < offset) {
      size_t newline = text.find('\n', scan);
      if (newline == llvm::StringRef::npos ||
          static_cast<int32_t>(newline) >= offset) {
        character += MeasureCharacterUnits(text.slice(scan, offset), encoding);
        scan = offset;
      } else {
        ++line;
        character = 0;
        scan = static_cast<int32_t>(newline) + 1;
      }
    }
    return {.line = line, .character = character};
  };

  std::vector<clang::clangd::TextEdit> text_edits;
  text_edits.reserve(edits.size());
  for (const Format::Replacement& edit : edits) {
    clang::clangd::Position start = position_at(edit.offset);
    clang::clangd::Position end = position_at(edit.offset + edit.length);
    text_edits.push_back(
        {.range = {.start = start, .end = end}, .newText = edit.text});
  }
  return text_edits;
}

static auto FormatDocument(
    Context& context, llvm::StringRef filename,
    std::optional<Format::LineRange> lines,
    llvm::function_ref<
        auto(llvm::Expected<std::vector<clang::clangd::TextEdit>>)->void>
        on_done) -> void {
  auto* file = context.LookupFile(filename);
  if (!file) {
    on_done(llvm::make_error<clang::clangd::LSPError>(
        llvm::formatv("Unknown textDocument `{0}`", filename),
        clang::clangd::ErrorCode::InvalidParams));
    return;
  }
  if (file->is_test_file()) {
    // Toolchain test files contain test directives and multiple split files
    // rather than standalone Carbon source.
    on_done(std::vector<clang::clangd::TextEdit>());
    return;
  }

  llvm::SmallVector<Format::Replacement> edits;
  Format::FormatReplacements(file->tree_and_subtrees().tree(), edits, lines);
  on_done(ReplacementsToTextEdits(file->text(), context.position_encoding(),
                                  edits));
}

auto HandleFormatting(
    Context& context, const clang::clangd::DocumentFormattingParams& params,
    llvm::function_ref<
        auto(llvm::Expected<std::vector<clang::clangd::TextEdit>>)->void>
        on_done) -> void {
  FormatDocument(context, params.textDocument.uri.file(),
                 /*lines=*/std::nullopt, on_done);
}

// Converts a 0-based, end-exclusive LSP range to a 1-based inclusive
// `LineRange`, clamping out-of-range coordinates safely.
static auto LspRangeToLineRange(const clang::clangd::Range& range)
    -> Format::LineRange {
  if (range.end.line < range.start.line ||
      (range.end.line == range.start.line &&
       range.end.character < range.start.character) ||
      range.end.line < 0) {
    return {.first_line = 1, .last_line = 0};
  }
  int start_line = std::clamp(range.start.line, 0, INT_MAX - 1);
  int end_line = std::clamp(range.end.line, start_line, INT_MAX - 1);
  if (range.end.character <= 0 && end_line > start_line) {
    --end_line;
  }
  return {.first_line = start_line + 1, .last_line = end_line + 1};
}

auto HandleRangeFormatting(
    Context& context,
    const clang::clangd::DocumentRangeFormattingParams& params,
    llvm::function_ref<
        auto(llvm::Expected<std::vector<clang::clangd::TextEdit>>)->void>
        on_done) -> void {
  FormatDocument(context, params.textDocument.uri.file(),
                 LspRangeToLineRange(params.range), on_done);
}

}  // namespace Carbon::LanguageServer

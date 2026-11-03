// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>

#include "common/check.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "testing/fuzzing/libfuzzer.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/diagnostics/null_diagnostics.h"
#include "toolchain/format/format.h"
#include "toolchain/lex/lex.h"
#include "toolchain/lex/token_index.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/parse.h"
#include "toolchain/parse/tree.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon::Testing {
namespace {

// The outcome of formatting one source text, recorded for the invariant checks
// below.
struct FormatResult {
  // Output from `Format`.
  std::string formatted;
  // Output reconstructed from `FormatReplacements` + `ApplyReplacements`.
  std::string via_replacements;
  // Output from applying `FormatReplacements` restricted to `line_range` when
  // `line_range` was provided.
  std::string via_range_replacements;
  // Whether `FormatReplacements` produced any edits.
  bool has_replacements;
  // Whether the input was free of lex errors.
  bool lex_clean;
  // Whether the input was free of both lex and parse errors.
  bool clean;
  // Token kinds and non-string token spellings on clean inputs, in order.
  llvm::SmallVector<Lex::TokenKind> token_kinds;
  llvm::SmallVector<std::string> non_string_token_texts;
  // Comment words (and `""` sentinels for blank `//` lines) on lex-clean
  // inputs, invariant under comment re-indentation and line wrapping.
  llvm::SmallVector<std::string> comment_tokens;
};

}  // namespace

static auto ExtractCommentTokens(const Lex::TokenizedBuffer& tokens)
    -> llvm::SmallVector<std::string> {
  llvm::SmallVector<std::string> result;
  for (Lex::CommentIndex comment : tokens.comments()) {
    llvm::SmallVector<llvm::StringRef> lines;
    tokens.GetCommentText(comment).split(lines, '\n');
    for (llvm::StringRef line : lines) {
      line = line.trim(" \t");
      if (!line.consume_front("//")) {
        continue;
      }
      line = line.trim(" \t");
      if (line.empty()) {
        result.push_back("");
        continue;
      }
      while (!line.empty()) {
        size_t word_end = line.find_first_of(" \t");
        result.push_back(line.take_front(word_end).str());
        if (word_end == llvm::StringRef::npos) {
          break;
        }
        line = line.drop_front(word_end).ltrim(" \t");
      }
    }
  }
  return result;
}

// Lexes, parses, and formats `text` via `Format` and `FormatReplacements`,
// returning `std::nullopt` only when no `SourceBuffer` can be constructed.
static auto FormatOnce(llvm::StringRef text,
                       std::optional<Format::LineRange> line_range =
                           std::nullopt) -> std::optional<FormatResult> {
  std::optional<SourceBuffer> source = SourceBuffer::MakeFromStringCopy(
      "fuzz.carbon", text, Diagnostics::NullConsumer());
  if (!source) {
    return std::nullopt;
  }

  SharedValueStores value_stores;
  Lex::LexOptions lex_options;
  lex_options.consumer = &Diagnostics::NullConsumer();
  Lex::TokenizedBuffer tokens = Lex::Lex(value_stores, *source, lex_options);

  Parse::ParseOptions parse_options;
  parse_options.consumer = &Diagnostics::NullConsumer();
  Parse::Tree tree = Parse::Parse(tokens, parse_options);

  FormatResult result;
  result.lex_clean = !tokens.has_errors();
  result.clean = result.lex_clean && !tree.has_errors();
  if (result.lex_clean) {
    result.comment_tokens = ExtractCommentTokens(tokens);
  }
  if (result.clean) {
    for (Lex::TokenIndex token : tokens.tokens()) {
      Lex::TokenKind kind = tokens.GetKind(token);
      result.token_kinds.push_back(kind);
      if (kind != Lex::TokenKind::StringLiteral) {
        result.non_string_token_texts.push_back(
            tokens.GetTokenText(token).str());
      }
    }
  }

  // Disable embedded C++ reformatting here so fuzzing exercises Carbon's
  // formatter without triggering clang-format debug assertions on arbitrary
  // non-C++ byte sequences; `cpp_snippet_test` covers `CppSnippet` directly.
  Format::Style style;
  style.format_cpp_snippets = false;

  RawStringOstream out;
  Format::Format(tree, out, style);
  result.formatted = out.TakeStr();

  llvm::SmallVector<Format::Replacement> replacements;
  Format::FormatReplacements(tree, replacements, /*lines=*/std::nullopt, style);
  result.has_replacements = !replacements.empty();
  result.via_replacements = Format::ApplyReplacements(text, replacements);

  if (line_range) {
    llvm::SmallVector<Format::Replacement> range_replacements;
    Format::FormatReplacements(tree, range_replacements, line_range, style);
    result.via_range_replacements =
        Format::ApplyReplacements(text, range_replacements);
  }
  return result;
}

// Whether `text` is free of control characters other than tab and newline.
// Formatting invariants are checked on ordinary error-free source, while inputs
// containing stray control characters (such as carriage returns or NULs) still
// exercise crash-freedom and replacement equivalence.
static auto IsOrdinarySource(llvm::StringRef text) -> bool {
  for (char c : text) {
    auto byte = static_cast<unsigned char>(c);
    if ((byte < ' ' || byte == 0x7F) && c != '\t' && c != '\n') {
      return false;
    }
  }
  return true;
}

// NOLINTNEXTLINE: Match the documented fuzzer entry point declaration style.
extern "C" int LLVMFuzzerTestOneInput(const unsigned char* data, size_t size) {
  // Ignore large inputs.
  // TODO: See tokenized_buffer_fuzzer.cpp.
  if (size > 100000) {
    return 0;
  }
  llvm::StringRef input(reinterpret_cast<const char*>(data), size);

  std::optional<Format::LineRange> line_range;
  if (!input.empty()) {
    int line_count =
        static_cast<int>(input.count('\n')) + (input.ends_with('\n') ? 0 : 1);
    int a = (static_cast<unsigned char>(input.front()) % line_count) + 1;
    int b = (static_cast<unsigned char>(input.back()) % line_count) + 1;
    line_range = Format::LineRange{.first_line = std::min(a, b),
                                   .last_line = std::max(a, b)};
  }

  // Run whole-text, full replacement, and line-range formatting across all
  // inputs, including lex and parse errors.
  std::optional<FormatResult> first = FormatOnce(input, line_range);
  if (!first) {
    return 0;
  }

  // Applying full replacements must reproduce `Format` on every input,
  // including malformed and control-character inputs.
  CARBON_CHECK(first->via_replacements == first->formatted,
               "minimal-edit output did not match whole-text output");

  if (!first->lex_clean || !IsOrdinarySource(input)) {
    return 0;
  }

  std::optional<FormatResult> second = FormatOnce(first->formatted);
  CARBON_CHECK(second && second->lex_clean,
               "formatted output of lex-clean input had lex errors");
  CARBON_CHECK(second->comment_tokens == first->comment_tokens,
               "formatting dropped, duplicated, or altered comment text");

  if (!first->clean) {
    return 0;
  }
  CARBON_CHECK(second->clean,
               "formatted output of clean input had parse errors");

  // Idempotency: formatting already-formatted output changes nothing, and
  // produces no edits.
  CARBON_CHECK(second->formatted == first->formatted,
               "formatting is not idempotent");
  CARBON_CHECK(!second->has_replacements,
               "re-formatting already-formatted output produced edits");

  // Token preservation: formatting preserves the sequence of token kinds and
  // non-string token spellings.
  CARBON_CHECK(second->token_kinds == first->token_kinds,
               "formatting changed the token sequence");
  CARBON_CHECK(second->non_string_token_texts == first->non_string_token_texts,
               "formatting changed non-string token text");

  // Line-range consistency: formatting a subrange of clean input keeps it clean
  // and converges to the same output under full `Format`.
  if (line_range) {
    std::optional<FormatResult> after_partial =
        FormatOnce(first->via_range_replacements);
    CARBON_CHECK(after_partial && after_partial->clean,
                 "range-formatted output of clean input had errors");
    CARBON_CHECK(after_partial->formatted == first->formatted,
                 "range formatting diverged from full formatting");
  }
  return 0;
}

}  // namespace Carbon::Testing

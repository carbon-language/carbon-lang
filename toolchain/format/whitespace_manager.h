// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_FORMAT_WHITESPACE_MANAGER_H_
#define CARBON_TOOLCHAIN_FORMAT_WHITESPACE_MANAGER_H_

#include <cstdint>
#include <deque>
#include <string>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/format/style.h"
#include "toolchain/lex/token_index.h"
#include "toolchain/lex/tokenized_buffer.h"

namespace Carbon::Format {

// Byte span of an emitted token in both the original source and the formatted
// output. `ComputeReplacements` uses these unchanged spans as anchors and
// diffs only the gaps between them.
struct TokenSpan {
  int32_t source_begin;
  int32_t output_begin;
  int32_t length;
};

// Records leading whitespace decisions for tokens and comments, aligns
// consecutive trailing comments when enabled by `style`, and renders the
// formatted output and token anchor map in a single pass.
class WhitespaceManager {
 public:
  WhitespaceManager(const Lex::TokenizedBuffer* tokens, const Style& style)
      : tokens_(tokens), style_(style) {}

  // Records `token`, preceded by `newlines` line breaks and `spaces` columns
  // of indentation or inter-token spacing, at unwrapped-line indentation
  // `indent_level` (used to group same-indent trailing-comment runs).
  //
  // If `rewritten` is non-empty, it is emitted in place of the token's source
  // spelling (such as a reformatted C++ snippet) and the token is omitted from
  // `TokenSpan` anchors so its edit folds into the surrounding gap.
  auto AddToken(int newlines, int spaces, int indent_level,
                Lex::TokenIndex token, std::string rewritten = "") -> void;

  // Records `token` preceded verbatim by `gap` (a slice of the original source
  // buffer between two tokens in the same error region), at unwrapped-line
  // indentation `indent_level`, with optional `rewritten` token text as in
  // `AddToken`.
  auto AddVerbatimGapToken(llvm::StringRef gap, int indent_level,
                           Lex::TokenIndex token, std::string rewritten = "")
      -> void;

  // Records a formatted full-line comment block `text` (with no trailing
  // newline), preceded by `newlines` line breaks.
  auto AddRaw(int newlines, std::string text) -> void;

  // Records a single-line trailing comment `text` (a slice of the source with
  // no trailing newline) appended to the current line after one separating
  // space plus any alignment padding.
  auto AddTrailingComment(llvm::StringRef text) -> void;

  // Runs trailing-comment alignment, appends 1-based source-line extents of
  // aligned comment groups to `aligned_line_extents`, and returns the formatted
  // output while recording anchored tokens in `token_map`.
  auto Generate(
      llvm::SmallVectorImpl<TokenSpan>& token_map,
      llvm::SmallVectorImpl<std::pair<int, int>>& aligned_line_extents)
      -> std::string;

 private:
  enum class ChangeKind : uint8_t {
    Token,
    VerbatimGapToken,
    FullLineComment,
    TrailingComment,
  };

  // A single recorded emission item: leading whitespace plus either a token
  // reference or comment text. `gap` holds the raw leading source gap for
  // `VerbatimGapToken`, while `text` holds any owned rewritten spelling for
  // token items or the comment text for comment items.
  struct Change {
    ChangeKind kind;
    int newlines = 0;
    int spaces = 0;
    int indent_level = 0;
    int start_column = 0;
    Lex::TokenIndex token = Lex::TokenIndex::None;
    llvm::StringRef gap;
    llvm::StringRef text;
  };

  auto OwnText(std::string text) -> llvm::StringRef;
  auto TokenText(const Change& change) const -> llvm::StringRef;

  // Populates `start_column` on tokens and trailing comments from the current
  // `newlines` and `spaces` decisions.
  auto ComputeStartColumns() -> void;

  // Pads consecutive same-indent trailing comments to a shared column when
  // `style_.align_trailing_comments` is enabled, recording the 1-based source
  // line span of each aligned group in `aligned_line_extents`.
  auto AlignTrailingComments(
      llvm::SmallVectorImpl<std::pair<int, int>>& aligned_line_extents) -> void;

  const Lex::TokenizedBuffer* tokens_;
  Style style_;
  llvm::SmallVector<Change> changes_;
  std::deque<std::string> owned_texts_;
};

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_WHITESPACE_MANAGER_H_

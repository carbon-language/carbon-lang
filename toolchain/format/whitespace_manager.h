// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_FORMAT_WHITESPACE_MANAGER_H_
#define CARBON_TOOLCHAIN_FORMAT_WHITESPACE_MANAGER_H_

#include <cstdint>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "toolchain/lex/token_index.h"
#include "toolchain/lex/tokenized_buffer.h"

namespace Carbon::Format {

// Where one emitted token sits in both the source and the formatted output.
// Token text is copied verbatim, so the two spans have equal length and
// content; the differences between source and output are entirely in the gaps
// between tokens. `ComputeReplacements` uses these as fixed anchors.
struct TokenSpan {
  int32_t source_begin;
  int32_t output_begin;
  int32_t length;
};

// Collects the whitespace decision before every emitted token, mirroring
// clang-format's `WhitespaceManager`: layout is expressed as a sequence of
// `Change`s (so many newlines and spaces before each token), and the formatted
// text is generated from them in one place. Routing all spacing through this
// one channel is what makes the per-token edit model (and, later, column
// alignment) possible without the rest of the formatter knowing.
//
// Carbon's comments are not tokens, so a formatted comment block is recorded as
// raw, verbatim text between tokens; it is emitted unchanged and is not a
// replacement anchor. A trailing comment is a raw change too, but one that
// appends to the current line rather than starting a new one.
//
// TODO: This is the minimal model the minimal-edit support needs. It gains the
// per-token bookkeeping (brace/bracket nesting, start columns) and the
// consecutive-token alignment passes (including trailing-comment alignment)
// when those land.
class WhitespaceManager {
 public:
  explicit WhitespaceManager(const Lex::TokenizedBuffer* tokens)
      : tokens_(tokens) {}

  // Records `token`, preceded by `newlines` line breaks and then `spaces`
  // spaces of indentation.
  auto AddToken(int newlines, int spaces, Lex::TokenIndex token) -> void;

  // Records `text` (a formatted comment block, which ends with a newline)
  // preceded by `newlines` line breaks. It is emitted verbatim.
  auto AddRaw(int newlines, std::string text) -> void;

  // Records `text` (a single-line trailing comment, with no trailing newline)
  // appended to the current line: no line break, then a single separating
  // space. Must follow the token (or comment) it trails. The newline that ends
  // the line is attributed, as always, to the following content.
  auto AddTrailingComment(std::string text) -> void;

  // Returns the formatted text, recording where each token landed into
  // `token_map`.
  auto Generate(llvm::SmallVectorImpl<TokenSpan>& token_map) -> std::string;

 private:
  // One whitespace decision: the spacing before a single token, or a raw
  // comment block. The token text is taken verbatim from the buffer when
  // generating.
  struct Change {
    // Whether this change introduces a token (anchored) or raw comment text.
    bool is_token;
    // For a raw change, whether it is a trailing comment that appends to the
    // current line (rather than a full-line comment block on its own line).
    bool is_trailing_comment = false;
    // The whitespace before the content: `newlines` line breaks, then `spaces`
    // columns of indentation.
    int newlines;
    int spaces;
    // Token changes only: the token whose text is emitted and anchored.
    Lex::TokenIndex token = Lex::TokenIndex::None;
    // Raw changes only: the verbatim text to emit.
    std::string raw;
  };

  const Lex::TokenizedBuffer* tokens_;
  llvm::SmallVector<Change> changes_;
};

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_WHITESPACE_MANAGER_H_

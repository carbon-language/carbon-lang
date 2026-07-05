// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_FORMAT_WHITESPACE_MANAGER_H_
#define CARBON_TOOLCHAIN_FORMAT_WHITESPACE_MANAGER_H_

#include <cstdint>
#include <string>

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/format/style.h"
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
// one channel is what lets a column-alignment pass adjust the space before a
// run of tokens so they line up, without the rest of the formatter knowing.
//
// Carbon's comments are not tokens, so a formatted comment block is recorded as
// raw, verbatim text between tokens; it is emitted unchanged and is neither an
// alignment anchor nor (since tokens are the replacement anchors) edited apart
// from the gaps around it. A trailing comment is a raw change too, but one that
// appends to the current line rather than starting a new one, and, unlike a
// full-line block, it does participate in alignment, so a run of them lines
// up (clang-format's `alignTrailingComments`).
class WhitespaceManager {
 public:
  WhitespaceManager(const Lex::TokenizedBuffer* tokens, const Style& style)
      : tokens_(tokens), style_(style) {}

  // Records `token`, preceded by `newlines` line breaks and then `spaces`
  // spaces of indentation. `indent_level` is the unwrapped line's indentation
  // column and `nesting_level` the bracket-nesting depth within the line; the
  // alignment pass groups tokens that share both.
  //
  // If `rewritten` is non-empty, it is emitted in place of the token's source
  // text (for example a reformatted embedded C++ snippet). A rewritten token is
  // not an alignment or replacement anchor, so its edit folds into the
  // surrounding gap when computing minimal replacements.
  auto AddToken(int newlines, int spaces, int indent_level, int nesting_level,
                Lex::TokenIndex token, llvm::StringRef rewritten = "") -> void;

  // Records `token` preceded by `gap`, the token's original leading source
  // text (which may hold newlines, comments, or any other whitespace), emitted
  // verbatim in place of computed spacing. Used inside a verbatim error
  // region; the token itself is still an anchor, but its spacing is exempt
  // from alignment.
  auto AddVerbatimGapToken(std::string gap, int indent_level, int nesting_level,
                           Lex::TokenIndex token) -> void;

  // Records `text` (a formatted comment block, which ends with a newline)
  // preceded by `newlines` line breaks. It is emitted verbatim.
  auto AddRaw(int newlines, std::string text) -> void;

  // Records `text` (a single-line trailing comment, with no trailing newline)
  // appended to the current line: no line break, then a single separating space
  // plus any alignment padding. Must follow the token (or comment) it trails.
  // The newline that ends the line is attributed, as always, to the following
  // content.
  auto AddTrailingComment(std::string text) -> void;

  // Runs the configured alignment pass and returns the formatted text,
  // recording where each token landed into `token_map`.
  auto Generate(llvm::SmallVectorImpl<TokenSpan>& token_map) -> std::string;

 private:
  // One whitespace decision: the spacing before a single token, or a raw
  // comment block. The token text is taken verbatim from the buffer when
  // generating.
  struct Change {
    // Whether this change introduces a token (anchored) or raw comment text.
    bool is_token;
    // For a raw change, whether it is a trailing comment that appends to the
    // current line (rather than a full-line comment block on its own line). A
    // trailing comment carries its separating space in `spaces` plus alignment
    // `padding`, and is aligned into a column with adjacent trailing comments.
    bool is_trailing_comment = false;
    // The whitespace before the content: `newlines` line breaks, then `spaces`
    // columns of indentation, then `padding` more columns added by alignment.
    int newlines;
    int spaces;
    int padding = 0;

    // Token changes only:
    Lex::TokenIndex token = Lex::TokenIndex::None;
    // The unwrapped line's indentation column, used to group alignment runs by
    // scope.
    int indent_level = 0;
    // The bracket-nesting depth within the unwrapped line; the alignment pass
    // uses it to tell a wrapped continuation line from a new statement.
    int nesting_level = 0;
    // The column the token starts at, filled in while generating and refreshed
    // before each alignment pass.
    int start_column = 0;
    // Text to emit in place of the token's verbatim source spelling, or empty
    // to emit the token verbatim. A rewritten token is not recorded as a
    // `TokenSpan` anchor.
    std::string rewritten;
    // Whether the token's leading whitespace is `verbatim_gap`, the original
    // source text before it, rather than `newlines`/`spaces` (which then only
    // describe the gap for the alignment pass's line partitioning).
    bool is_verbatim_gap = false;
    std::string verbatim_gap;

    // Raw changes only: the verbatim text to emit.
    std::string raw;
  };

  // Fills `start_column` for every token change from the current spacing, so an
  // alignment pass can find the rightmost natural column in a run.
  auto ComputeStartColumns() -> void;

  // The shared alignment engine: partitions the changes into physical lines,
  // asks the pass's matcher for each line's alignment change (given the line's
  // half-open change range, returning -1 for none), and pads each matched
  // change in a run of consecutive same-indent lines so they share a column
  // (the run's rightmost natural one). A blank line, an indent change, or an
  // unmatched line breaks the run, except a wrapped continuation line still
  // inside brackets, which neither joins nor breaks it (mirroring
  // clang-format's deeper-nesting skip). Written over a generic matcher so
  // that further consecutive-token alignment passes (clang-format's
  // off-by-default `alignConsecutiveAssignments` family) can reuse it; see
  // the future-work notes in toolchain/docs/format.md.
  auto AlignChanges(llvm::function_ref<auto(int, int)->int> find_match) -> void;

  // Aligns the `//` of a run of consecutive trailing comments at the same
  // indent into one column (clang-format's `alignTrailingComments`), enabled by
  // the `align_trailing_comments` knob. On in the canonical style.
  auto AlignTrailingComments() -> void;

  const Lex::TokenizedBuffer* tokens_;
  Style style_;
  llvm::SmallVector<Change> changes_;
};

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_WHITESPACE_MANAGER_H_

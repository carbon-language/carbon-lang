// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEX_MISMATCHED_BRACKETS_H_
#define CARBON_TOOLCHAIN_LEX_MISMATCHED_BRACKETS_H_

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/lex/token_index.h"
#include "toolchain/lex/token_kind.h"

namespace Carbon::Lex {

// Represents a category of token significant for bracket matching recovery.
enum class BracketTokenKind : int8_t {
  OpenParen,
  CloseParen,
  OpenCurlyBrace,
  CloseCurlyBrace,
  OpenSquareBracket,
  CloseSquareBracket,
  Semi,
  Comma,
  Period,
  StatementIntroducer,  // fn, class, var, if, while, etc.

  // A token that is a complete primary expression on its own: an identifier, a
  // literal, `self`, a type keyword, and so on. A leaf token never directly
  // follows another leaf token or a close paren or close square bracket, so
  // such an adjacent pair is evidence that a bracket is missing between them.
  Leaf,

  FileEnd,
  Other,
};

// Returns true if the token kind is an opening bracket.
constexpr auto IsOpeningBracket(BracketTokenKind kind) -> bool {
  return kind == BracketTokenKind::OpenParen ||
         kind == BracketTokenKind::OpenCurlyBrace ||
         kind == BracketTokenKind::OpenSquareBracket;
}

// Returns true if the token kind is a closing bracket.
constexpr auto IsClosingBracket(BracketTokenKind kind) -> bool {
  return kind == BracketTokenKind::CloseParen ||
         kind == BracketTokenKind::CloseCurlyBrace ||
         kind == BracketTokenKind::CloseSquareBracket;
}

// Returns the matching closing bracket kind for an opening bracket.
constexpr auto MatchingClosingKind(BracketTokenKind kind) -> BracketTokenKind {
  switch (kind) {
    case BracketTokenKind::OpenParen:
      return BracketTokenKind::CloseParen;
    case BracketTokenKind::OpenCurlyBrace:
      return BracketTokenKind::CloseCurlyBrace;
    case BracketTokenKind::OpenSquareBracket:
      return BracketTokenKind::CloseSquareBracket;
    default:
      return BracketTokenKind::Other;
  }
}

// Returns the matching opening bracket kind for a closing bracket.
constexpr auto MatchingOpeningKind(BracketTokenKind kind) -> BracketTokenKind {
  switch (kind) {
    case BracketTokenKind::CloseParen:
      return BracketTokenKind::OpenParen;
    case BracketTokenKind::CloseCurlyBrace:
      return BracketTokenKind::OpenCurlyBrace;
    case BracketTokenKind::CloseSquareBracket:
      return BracketTokenKind::OpenSquareBracket;
    default:
      return BracketTokenKind::Other;
  }
}

// Converts a BracketTokenKind to standard TokenKind.
constexpr auto ToTokenKind(BracketTokenKind kind) -> TokenKind {
  switch (kind) {
    case BracketTokenKind::OpenParen:
      return TokenKind::OpenParen;
    case BracketTokenKind::CloseParen:
      return TokenKind::CloseParen;
    case BracketTokenKind::OpenCurlyBrace:
      return TokenKind::OpenCurlyBrace;
    case BracketTokenKind::CloseCurlyBrace:
      return TokenKind::CloseCurlyBrace;
    case BracketTokenKind::OpenSquareBracket:
      return TokenKind::OpenSquareBracket;
    case BracketTokenKind::CloseSquareBracket:
      return TokenKind::CloseSquareBracket;
    case BracketTokenKind::Semi:
      return TokenKind::Semi;
    default:
      return TokenKind::Error;
  }
}

// Lightweight token description passed into the bracket matching algorithm.
struct MismatchedBracketToken {
  TokenIndex token_index = TokenIndex::None;
  BracketTokenKind kind;
  int32_t line;
  int32_t line_indent;
  int32_t column;

  // Whether this token is the last non-comment token on its line.
  bool is_at_end_of_line = false;

  // For OpenCurlyBrace, whether it has struct-like cues (e.g. followed by '.',
  // '}', or ':').
  bool is_struct_brace = false;

  // Whether the immediately preceding token in the source (ignoring comments)
  // is "value-ending": a leaf, `)`, or `]`. A leaf token whose predecessor is
  // value-ending is an illegal adjacency in a well-formed program, and is a
  // strong cue that an opening bracket is missing right before this token.
  bool prev_is_value_ending = false;

  // For StatementIntroducer, whether this is a keyword that must be directly
  // followed by an opening bracket: `if`, `while`, `for`, `match` (which
  // require `(`), or `forall` (which requires `[`).
  bool is_paren_keyword = false;

  // For StatementIntroducer, whether this is the `else` keyword, which
  // normally directly follows a `}` on the same line.
  bool is_else_keyword = false;

  // For Other, whether this is a statement-structuring operator (`=`, `->`,
  // `as`) that usually appears outside parens and square brackets, so is a
  // cue that an unclosed paren or square bracket should be closed before it.
  bool is_structural_op = false;

  // For Other, whether this is specifically an `=`, which can also directly
  // follow a `]` (`a[i] = v;`), unlike the other structural operators.
  bool is_assignment_op = false;

  // For Other, whether this is specifically an `as`, which commonly appears
  // inside parens as a cast (`(x as T)`), unlike the other structural
  // operators.
  bool is_as_op = false;

  // For Other, whether this is a comparison or logical operator (`==`, `<`,
  // `and`, ...), which is unlikely to appear inside square brackets.
  bool is_comparison_op = false;

  // For Other, whether this is a binding modifier keyword (`ref`, `unused`,
  // `template`), which, like a leaf, cannot directly follow a value-ending
  // token.
  bool is_modifier_keyword = false;

  // Whether this token has whitespace (or a comment) directly before it.
  bool has_leading_space = false;

  // Whether this token is mid-line with two or more bytes of whitespace
  // before it. Formatted code separates mid-line tokens by at most one
  // space, so a wide gap suggests something was deleted in it.
  bool has_wide_leading_space = false;

  // Byte offset of this token in source.
  int32_t byte_offset = 0;
};

// An action to fix mismatched brackets in the token stream.
enum class BracketFixAction : int8_t {
  // Insert a missing bracket before the specified token.
  InsertBefore,

  // Insert a missing bracket after the specified token.
  InsertAfter,

  // Replace an unmatched bracket with an error token.
  ReplaceWithError,
};

// A diagnostic to issue for an unmatched or repaired bracket.
enum class BracketDiagnosticKind : int8_t {
  UnmatchedOpening,
  UnmatchedClosing,
};

// Represents a single correction made to recover from a mismatched bracket,
// pairing the diagnostic to report with the token-stream fix to apply.
struct BracketCorrection {
  // The diagnostic to report for this bracket error.
  BracketDiagnosticKind diagnostic_kind;
  TokenIndex diagnostic_token_index = TokenIndex::None;

  // The fix action to apply to the token stream.
  BracketFixAction fix_action;
  TokenIndex fix_token_index = TokenIndex::None;
  TokenKind fix_token_kind;
  int32_t fix_byte_offset = -1;

  // Set to true if multiple optimal paths tie/disagree on the repair.
  bool is_tied = false;

  // Source transition/origin for diagnostics and evaluation.
  const char* origin = "";
};

// Analyzes the input token stream, finds the optimal set of bracket insertions
// and error replacements based on indentation and structural cues, and returns
// the corresponding list of corrections.
auto FixMismatchedBrackets(llvm::ArrayRef<MismatchedBracketToken> tokens)
    -> llvm::SmallVector<BracketCorrection>;

}  // namespace Carbon::Lex

#endif  // CARBON_TOOLCHAIN_LEX_MISMATCHED_BRACKETS_H_

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/mismatched_brackets.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

#include "common/check.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"

namespace Carbon::Lex {
namespace {

// Maximum number of collapsed items in a damaged region before falling back to
// naive greedy recovery. The beam search below is linear in the region size,
// so this is mostly a defense against pathological inputs.
constexpr int32_t MaxRegionItemsForSearch = 1500;

// Layered beam search width limit.
constexpr size_t MaxBeamWidth = 16;

// Maximum stack depth allowed during search before capping.
constexpr size_t MaxSearchStackDepth = 12;

// The cost model. All costs are relative; the search finds the cheapest way
// to make the whole region well-bracketed, and the resulting insertions and
// replacements become the suggested corrections. When several cheapest ways
// exist, corrections they disagree about are marked tied and downgraded to
// error replacements, so it's important that the intended repair for a common
// mistake is *strictly* cheaper than the alternatives.
//
// Costs of replacing a real bracket with an error token. These are the
// "give up on this bracket" fallbacks; a good targeted repair should beat
// them, and a dubious one should lose to them.
constexpr int32_t CostReplaceClosing = 30;
constexpr int32_t CostReplaceOpening = 100;

// Costs of inserting a synthetic closing bracket in front of the current
// token, keyed by how strongly the context suggests the group ends here.
// An empty group: the opener is the immediately preceding token.
constexpr int32_t CostCloseEmptyGroup = 8;
// Closing a paren/square bracket just before a `;`.
constexpr int32_t CostCloseParenBeforeSemi = 6;
// Closing a group because the current closer matches an outer open group.
constexpr int32_t CostCloseCascade = 6;
// Closing a paren/square bracket just before a scope `{`.
constexpr int32_t CostCloseParenBeforeBrace = 8;
// Closing a paren/square bracket just before `=`, `->`, or `as`.
constexpr int32_t CostCloseParenBeforeStructuralOp = 8;
// Closing a paren/square bracket before a mid-line `.` that has whitespace
// before it: member access is normally written without spaces. Priced below
// CostSpacedPeriodInParen so that closing here beats closing earlier and
// leaving the spaced `.` unexplained.
constexpr int32_t CostCloseParenBeforeSpacedPeriod = 6;
// Closing a `[` at the start of a continuation line: `[` groups rarely span
// lines except in wrapped declaration headers.
constexpr int32_t CostCloseSquareAtContinuation = 10;
// Closing a `[` at an illegal leaf adjacency, which the `]` repairs.
constexpr int32_t CostCloseSquareAtLeafAdjacency = 4;
// Closing a group at a wide mid-line whitespace gap, which suggests the
// closer was deleted in the gap.
constexpr int32_t CostCloseAtWideGap = 6;
// Closing a struct brace at a dedent.
constexpr int32_t CostCloseStructAtDedent = 12;
// Closing a scope brace at a dedent back to the header's indentation.
constexpr int32_t CostCloseScopeAtDedent = 6;
// Closing a scope brace before a first-on-line `else`, which normally
// directly follows a `}` on the same line. Priced below
// CostCloseScopeAtDedent so this wins over closing the else block early.
constexpr int32_t CostCloseScopeBeforeElse = 4;
// Closing anything at the end of the file or region.
constexpr int32_t CostCloseAtEnd = 12;
constexpr int32_t CostCloseParenAtEnd = 22;
constexpr int32_t CostCloseStructAtEnd = 20;
// Closing with no supporting context cue.
constexpr int32_t CostCloseParenBaseline = 40;
constexpr int32_t CostCloseScopeBaseline = 45;

// Costs of inserting a synthetic opening bracket in front of the current
// token.
// After `if`, `while`, `for`, `match` (for `(`) or `forall` (for `[`), which
// require a bracket to follow.
constexpr int32_t CostOpenAfterParenKeyword = 3;
// Before a leaf token that directly follows a value-ending token; this
// adjacency is illegal, and an opener here fixes it.
constexpr int32_t CostOpenAtLeafAdjacency = 3;
// An empty `()` after a value-ending token (a zero-argument call).
constexpr int32_t CostOpenEmptyParens = 5;
// An empty `[]` after a value-ending token; rarer than empty parens.
constexpr int32_t CostOpenEmptySquares = 12;
// An empty `()` directly after a `)`, calling a just-computed value; only
// applies when the target closer is spaced.
constexpr int32_t CostOpenEmptyParensAfterClose = 8;
// A `(` or `[` before a mid-line leaf with whitespace before it that directly
// follows an unspaced `.`: member access is written without spaces.
constexpr int32_t CostOpenAfterPeriodGap = 4;
// A `(` or `[` anywhere else an expression could start.
constexpr int32_t CostOpenParenBaseline = 35;
// A scope `{` between an unbraced declaration/statement header and its body.
constexpr int32_t CostOpenScopeAfterHeader = 8;
// A struct `{` before a `.` designator that isn't a member access.
constexpr int32_t CostOpenStructBeforeDesignator = 5;
// An empty `{}` just before an unmatched `}`. Priced above
// CostOpenScopeAfterHeader: when a single-line body lost its `{`, inserting
// it before the body is better than making empty braces at the `}`.
constexpr int32_t CostOpenStructEmptyBraces = 10;
// A brace anywhere else.
constexpr int32_t CostOpenBraceBaseline = 60;

// Penalties for advancing over a token in a context where it doesn't belong.
// These make "close the group before this token" win over "swallow this
// token into the group".
// `;` inside parens, square brackets, or a struct brace.
constexpr int32_t CostSemiInParen = 100;
// `,` at statement level (directly in a scope brace or at the top level).
constexpr int32_t CostCommaAtStatementLevel = 50;
// `,` directly following an open paren/square bracket that is still open.
constexpr int32_t CostCommaAfterOpen = 50;
// A statement introducer keyword inside parens or a struct brace.
constexpr int32_t CostIntroducerInParen = 60;
// A leaf token directly following a value-ending token (illegal adjacency).
constexpr int32_t CostLeafAdjacency = 60;
// `=`, `->`, or `as` inside parens or square brackets. These *can* occur
// there (default arguments, function types, casts), so this is mild; it
// serves to prefer the earliest sensible close point.
constexpr int32_t CostStructuralOpInParen = 10;
// An `as` inside parens is usually a legitimate cast; keep only a nominal
// preference for closing before it.
constexpr int32_t CostAsOpInParen = 1;
// A mid-line `.` with whitespace before it inside parens or square brackets;
// member access is normally written without spaces, so a bracket was likely
// deleted before it.
constexpr int32_t CostSpacedPeriodInParen = 10;
// A spaced `(` or `[` directly following a value-ending token; calls and
// indexing are written without spaces, so a bracket was likely deleted in
// between.
constexpr int32_t CostSpacedOpenAfterValue = 10;
// A comparison or logical operator inside square brackets.
constexpr int32_t CostComparisonInSquare = 8;
// A `,` with whitespace before it inside parens or square brackets;
// formatted code has no space before a comma.
constexpr int32_t CostSpacedCommaInParen = 8;
// An operator with a wide mid-line whitespace gap before it that no repair
// on this path explains.
constexpr int32_t CostWideGapUnexplained = 10;
// A spaced `)` or `]` directly following a value that no repair on this path
// explains; formatted code has no space before closers.
constexpr int32_t CostSpacedCloserUnexplained = 8;
// A scope `{` opening inside parens or a struct brace (a lambda; rare).
constexpr int32_t CostScopeBraceInParen = 40;
// A struct `{` opening inside parens (a struct literal argument; common).
constexpr int32_t CostStructBraceInParen = 5;
// A line that dedents to at-or-before the indentation of the enclosing brace
// header while the brace is still open.
constexpr int32_t CostDedentInScope = 40;
// A line that dedents to at-or-before the indentation of an enclosing open
// paren's line.
constexpr int32_t CostDedentInParen = 25;

// Penalty for matching a scope `}` whose indentation doesn't match its
// opener's header, when they're on different lines.
constexpr int32_t CostBraceIndentMismatchBase = 30;
constexpr int32_t CostBraceIndentMismatchPerColumn = 20;

// Internal representation of an item after clean subrange collapsing.
struct Item {
  int32_t token_start_index;
  int32_t token_end_index;
  bool is_collapsed_block = false;
  bool contains_scope_brace = false;
  MismatchedBracketToken token;
  int32_t effective_header_indent = 0;
  bool is_first_on_line = false;
  bool follows_statement_header = false;
  bool header_has_open_curly_brace = false;
  // Kind and flags of the token immediately preceding this item in the input.
  BracketTokenKind prev_kind = BracketTokenKind::Other;
  bool prev_is_paren_keyword = false;
  bool prev_is_structural_op = false;
  bool prev_is_assignment_op = false;
  bool prev_has_leading_space = false;
};

// Represents an unclosed opening bracket on the search stack.
struct OpenBracketInfo {
  // The real opener token, or None for a synthetic opener.
  TokenIndex token_index = TokenIndex::None;
  // Index of the real opener in the input token array, or -1 if synthetic.
  int32_t token_pos = -1;
  BracketTokenKind kind;
  int32_t line = -1;
  // Indentation of the line containing the opener.
  int32_t line_indent = 0;
  // Indentation of the start of the statement or declaration containing the
  // opener.
  int32_t effective_header_indent = 0;
  bool is_synthetic = false;
  bool is_struct_brace = false;
  // Whether this is a paren/square bracket directly following a value-ending
  // token (a call or index), rather than following a keyword like `if`.
  // Derived from the opener token, so not part of equality.
  bool is_call_paren = false;
  // For a synthetic opener, where it would be inserted.
  TokenIndex insertion_token_index = TokenIndex::None;
  int32_t insertion_byte_offset = 0;
  // For a synthetic opener, the rule that proposed it, for diagnostics.
  const char* origin = "";

  friend auto operator==(const OpenBracketInfo& a, const OpenBracketInfo& b)
      -> bool {
    return a.token_index == b.token_index && a.token_pos == b.token_pos &&
           a.kind == b.kind && a.line == b.line &&
           a.line_indent == b.line_indent &&
           a.effective_header_indent == b.effective_header_indent &&
           a.is_synthetic == b.is_synthetic &&
           a.is_struct_brace == b.is_struct_brace &&
           a.insertion_token_index == b.insertion_token_index &&
           a.insertion_byte_offset == b.insertion_byte_offset;
  }
};

// Computes the associated line indentation for a token by scanning backwards,
// skipping matched parens/brackets, looking for a statement introducer.
auto GetOuterStatementIntroducerIndent(
    llvm::ArrayRef<MismatchedBracketToken> tokens,
    llvm::ArrayRef<int32_t> match_partner, int32_t j) -> int32_t {
  int32_t result_indent = tokens[j].line_indent;
  while (j > 0) {
    int32_t p = j - 1;
    if ((tokens[p].kind == BracketTokenKind::CloseParen ||
         tokens[p].kind == BracketTokenKind::CloseSquareBracket) &&
        match_partner[p] != -1 && match_partner[p] < p) {
      p = match_partner[p];
      if (p <= 0) {
        break;
      }
      --p;
    }
    if (tokens[p].kind == BracketTokenKind::StatementIntroducer) {
      if (tokens[j].line == tokens[p].line ||
          tokens[j].line_indent <= tokens[p].line_indent) {
        j = p;
        result_indent = tokens[p].line_indent;
        continue;
      }
      if (tokens[j].line != tokens[p].line &&
          tokens[j].line_indent > tokens[p].line_indent) {
        result_indent = tokens[p].line_indent;
        break;
      }
    }
    break;
  }
  // A first-on-line `else` normally directly follows a deleted `}` whose
  // indentation the statement really starts at: `} else {` puts `else` two
  // columns past the brace.
  if (tokens[j].is_else_keyword &&
      (j == 0 || tokens[j].line != tokens[j - 1].line)) {
    result_indent = std::max(1, result_indent - 2);
  }
  return result_indent;
}

auto ComputeAssociatedLineIndent(llvm::ArrayRef<MismatchedBracketToken> tokens,
                                 llvm::ArrayRef<int32_t> match_partner,
                                 int32_t token_index) -> int32_t {
  if (token_index < 0 || token_index >= static_cast<int32_t>(tokens.size())) {
    return 0;
  }
  if (tokens[token_index].kind == BracketTokenKind::FileEnd) {
    return 0;
  }

  int32_t earliest_indent = tokens[token_index].line_indent;

  for (int32_t j = token_index - 1; j >= 0; --j) {
    auto kind = tokens[j].kind;

    if (IsClosingBracket(kind) && match_partner[j] != -1 &&
        match_partner[j] < j) {
      j = match_partner[j];
      earliest_indent = tokens[j].line_indent;
      continue;
    }

    if (kind == BracketTokenKind::Semi ||
        kind == BracketTokenKind::OpenCurlyBrace ||
        kind == BracketTokenKind::CloseCurlyBrace) {
      break;
    }

    earliest_indent = tokens[j].line_indent;

    if (kind == BracketTokenKind::StatementIntroducer) {
      return GetOuterStatementIntroducerIndent(tokens, match_partner, j);
    }
  }

  return earliest_indent;
}

// Determines if a token follows a statement/declaration header, so that a
// scope `{` could naturally be inserted directly before it. Only tokens that
// could start a body are considered: the first token on a line, a statement
// introducer (e.g. `return` in `if (c) return;`), or a token directly
// following a `)`/`]` that ends a header.
auto ComputeFollowsStatementHeader(
    llvm::ArrayRef<MismatchedBracketToken> tokens,
    llvm::ArrayRef<int32_t> match_partner, int32_t token_index) -> bool {
  if (token_index <= 0 || token_index >= static_cast<int32_t>(tokens.size())) {
    return false;
  }
  auto curr_kind = tokens[token_index].kind;
  if (curr_kind == BracketTokenKind::Semi ||
      curr_kind == BracketTokenKind::OpenCurlyBrace ||
      IsClosingBracket(curr_kind) || curr_kind == BracketTokenKind::FileEnd) {
    return false;
  }
  // Only tokens that could start a body are considered: the first token on a
  // line, a statement introducer (`if (c) return;`), or a token directly
  // after a `)` ending a header. (Not after `]`: declaration headers always
  // continue after implicit parameter lists and `forall` clauses.)
  bool is_first_on_line =
      tokens[token_index].line != tokens[token_index - 1].line;
  auto prev_kind = tokens[token_index - 1].kind;
  if (!is_first_on_line &&
      curr_kind != BracketTokenKind::StatementIntroducer &&
      prev_kind != BracketTokenKind::CloseParen) {
    return false;
  }
  // A body can't start with an operator like `as` or `==`, or with a `.`
  // designator (a `where`-clause continuation line).
  if (tokens[token_index].is_structural_op ||
      tokens[token_index].is_comparison_op ||
      curr_kind == BracketTokenKind::Period) {
    return false;
  }
  for (int32_t j = token_index - 1; j >= 0; --j) {
    auto kind = tokens[j].kind;
    if ((kind == BracketTokenKind::CloseParen ||
         kind == BracketTokenKind::CloseSquareBracket) &&
        match_partner[j] != -1 && match_partner[j] < j) {
      j = match_partner[j];
      continue;
    }
    if (kind == BracketTokenKind::Semi ||
        kind == BracketTokenKind::OpenCurlyBrace ||
        kind == BracketTokenKind::CloseCurlyBrace ||
        kind == BracketTokenKind::OpenParen ||
        kind == BracketTokenKind::OpenSquareBracket) {
      return false;
    }
    if (kind == BracketTokenKind::StatementIntroducer) {
      if (curr_kind == BracketTokenKind::StatementIntroducer) {
        if (tokens[token_index].line == tokens[j].line) {
          // On the same line, a chain of adjacent introducers (`private fn`)
          // is a single header; but with other tokens in between, this is a
          // body after a single-line header (`if (c) return x;`).
          bool all_introducers = true;
          for (int32_t k = j + 1; k < token_index; ++k) {
            if (tokens[k].kind != BracketTokenKind::StatementIntroducer) {
              all_introducers = false;
              break;
            }
          }
          if (all_introducers) {
            return false;
          }
        } else if (tokens[token_index].line_indent <= tokens[j].line_indent) {
          return false;
        }
      }
      return true;
    }
  }
  return false;
}

// Determines if the statement/declaration header starting at token_index
// contains an OpenCurlyBrace.
auto ComputeHeaderHasOpenCurlyBrace(
    llvm::ArrayRef<MismatchedBracketToken> tokens,
    llvm::ArrayRef<int32_t> match_partner, int32_t token_index) -> bool {
  if (token_index <= 0 || token_index >= static_cast<int32_t>(tokens.size())) {
    return false;
  }
  for (int32_t j = token_index; j < static_cast<int32_t>(tokens.size()); ++j) {
    auto kind = tokens[j].kind;
    if (kind == BracketTokenKind::OpenCurlyBrace) {
      return true;
    }
    if ((kind == BracketTokenKind::OpenParen ||
         kind == BracketTokenKind::OpenSquareBracket) &&
        match_partner[j] != -1 && match_partner[j] > j) {
      j = match_partner[j];
      continue;
    }
    if (kind == BracketTokenKind::Semi ||
        kind == BracketTokenKind::CloseCurlyBrace ||
        kind == BracketTokenKind::FileEnd ||
        kind == BracketTokenKind::StatementIntroducer) {
      return false;
    }
  }
  return false;
}

struct ParentEdge {
  int32_t parent_node_index;
  BracketCorrection correction;
  bool has_correction = false;
};

// Node in the beam search tree.
struct BeamNode {
  int32_t item_index;
  llvm::SmallVector<OpenBracketInfo, 4> stack;
  int32_t cost;
  // The kind of synthetic closer inserted directly before the current item,
  // or Other if none; an inserted closer repairs illegal adjacency with the
  // preceding token.
  BracketTokenKind closer_inserted = BracketTokenKind::Other;
  llvm::SmallVector<ParentEdge, 2> parent_edges;
};

// The parts of a BeamNode the search reads while expanding it. Snapshotting
// just these (rather than copying the whole node, whose `parent_edges` the
// expansion never reads) avoids copying that vector per node. A copy — not a
// reference — is needed because expanding a node pushes new nodes into the
// arena, which may reallocate.
struct SearchState {
  llvm::SmallVector<OpenBracketInfo, 4> stack;
  int32_t cost;
  BracketTokenKind closer_inserted;
};

auto Snapshot(const BeamNode& node) -> SearchState {
  return {node.stack, node.cost, node.closer_inserted};
}

// A correction that replaces a bracket token with an error token (the "give
// up on this bracket" repair).
auto ReplaceWithError(const MismatchedBracketToken& token,
                      BracketDiagnosticKind diagnostic_kind,
                      const char* origin) -> BracketCorrection {
  return BracketCorrection{
      .diagnostic_kind = diagnostic_kind,
      .diagnostic_token_index = token.token_index,
      .fix_action = BracketFixAction::ReplaceWithError,
      .fix_token_index = token.token_index,
      .fix_token_kind = ToTokenKind(token.kind),
      .fix_byte_offset = token.byte_offset,
      .origin = origin,
  };
}

// Solve a damaged region using the simple greedy fallback algorithm.
auto SolveNaive(llvm::ArrayRef<Item> items,
                llvm::SmallVectorImpl<BracketCorrection>& corrections) -> void {
  llvm::SmallVector<Item> open_stack;
  for (const auto& item : items) {
    if (item.is_collapsed_block) {
      continue;
    }
    auto kind = item.token.kind;
    if (kind == BracketTokenKind::Semi ||
        kind == BracketTokenKind::StatementIntroducer ||
        kind == BracketTokenKind::OpenCurlyBrace ||
        kind == BracketTokenKind::FileEnd) {
      while (!open_stack.empty() &&
             (open_stack.back().token.kind == BracketTokenKind::OpenParen ||
              open_stack.back().token.kind ==
                  BracketTokenKind::OpenSquareBracket)) {
        auto open = open_stack.pop_back_val();
        corrections.push_back(ReplaceWithError(
            open.token, BracketDiagnosticKind::UnmatchedOpening,
            "Naive_UnclosedParenBracket"));
      }
    }

    if (IsOpeningBracket(kind)) {
      open_stack.push_back(item);
    } else if (IsClosingBracket(kind)) {
      auto search_range = llvm::reverse(open_stack);
      size_t lookback = 0;
      auto match_it = search_range.end();
      for (auto it = search_range.begin();
           it != search_range.end() && lookback < 16; ++it, ++lookback) {
        if (MatchingClosingKind(it->token.kind) == kind) {
          match_it = it;
          break;
        }
      }

      if (match_it == search_range.end()) {
        corrections.push_back(ReplaceWithError(
            item.token, BracketDiagnosticKind::UnmatchedClosing,
            "Naive_UnmatchedClosing"));
      } else {
        for (auto it = search_range.begin(); it != match_it; ++it) {
          corrections.push_back(ReplaceWithError(
              it->token, BracketDiagnosticKind::UnmatchedOpening,
              "Naive_PoppedOpener"));
        }
        open_stack.erase(match_it.base() - 1, open_stack.end());
      }
    }
  }

  for (const auto& open : llvm::reverse(open_stack)) {
    corrections.push_back(ReplaceWithError(
        open.token, BracketDiagnosticKind::UnmatchedOpening,
        "Naive_UnclosedAtEnd"));
  }
}

auto HashStack(llvm::ArrayRef<OpenBracketInfo> stack) -> uint64_t {
  uint64_t h = stack.size();
  for (const auto& info : stack) {
    uint64_t k = (static_cast<uint64_t>(info.token_index.index) << 32) ^
                 (static_cast<uint64_t>(info.insertion_token_index.index)
                  << 8) ^
                 static_cast<uint64_t>(info.kind) ^
                 (static_cast<uint64_t>(info.is_struct_brace) << 7) ^
                 (static_cast<uint64_t>(info.line_indent) << 16);
    h ^= k + 0x9e3779b9 + (h << 6) + (h >> 2);
  }
  return h;
}

// A search state is identified by its open-bracket stack plus which closer (if
// any) was just inserted before the current token; this hash keys the
// per-layer dedup map.
auto StateHash(llvm::ArrayRef<OpenBracketInfo> stack,
               BracketTokenKind closer_inserted) -> uint64_t {
  return HashStack(stack) ^
         (static_cast<uint64_t>(closer_inserted) * 0x5bd1e995);
}

// Whether two parent edges represent the same predecessor and the same repair,
// so that keeping both would be a redundant duplicate.
auto EdgesEqual(const ParentEdge& a, const ParentEdge& b) -> bool {
  if (a.parent_node_index != b.parent_node_index ||
      a.has_correction != b.has_correction) {
    return false;
  }
  return !a.has_correction ||
         (a.correction.diagnostic_token_index ==
              b.correction.diagnostic_token_index &&
          a.correction.fix_action == b.correction.fix_action &&
          a.correction.fix_token_index == b.correction.fix_token_index &&
          a.correction.fix_token_kind == b.correction.fix_token_kind);
}

// Classification of the top of the bracket stack.
enum class TopKind : int8_t {
  None,
  Paren,        // `(` or `[`
  StructBrace,  // `{` with struct cues
  ScopeBrace,   // `{` without struct cues
};

auto GetTopKind(llvm::ArrayRef<OpenBracketInfo> stack) -> TopKind {
  if (stack.empty()) {
    return TopKind::None;
  }
  const auto& top = stack.back();
  if (top.kind == BracketTokenKind::OpenCurlyBrace) {
    return top.is_struct_brace ? TopKind::StructBrace : TopKind::ScopeBrace;
  }
  return TopKind::Paren;
}

// Penalty for a token that dedents out of the group it's nominally inside.
// FileEnd is not content, so it never counts as a dedent.
auto ContextPenalty(const SearchState& node, const Item& item) -> int32_t {
  if (node.stack.empty() || !item.is_first_on_line ||
      item.token.kind == BracketTokenKind::FileEnd) {
    return 0;
  }
  const auto& top = node.stack.back();
  if (top.kind == BracketTokenKind::OpenCurlyBrace) {
    return item.token.line_indent <= top.effective_header_indent
               ? CostDedentInScope
               : 0;
  }
  return item.token.line_indent <= top.line_indent ? CostDedentInParen : 0;
}

// Returns true if `kind`'s matching opener appears in `stack` below the top.
auto MatchesDeeperOpener(llvm::ArrayRef<OpenBracketInfo> stack,
                         BracketTokenKind closing_kind) -> bool {
  if (!IsClosingBracket(closing_kind) || stack.size() < 2) {
    return false;
  }
  auto req = MatchingOpeningKind(closing_kind);
  for (size_t s = 0; s + 1 < stack.size(); ++s) {
    if (stack[s].kind == req) {
      return true;
    }
  }
  return false;
}

// Whether the token before `item` ends a value. Unlike `prev_is_value_ending`
// (used for the leaf-adjacency rule), this also counts `]` and `}`, which end
// a value but can be followed by a leaf.
auto PrevIsValueLike(const Item& item) -> bool {
  return item.token.prev_is_value_ending ||
         item.prev_kind == BracketTokenKind::CloseSquareBracket ||
         item.prev_kind == BracketTokenKind::CloseCurlyBrace;
}

// Whether the top of `stack` is a synthetic opener inserted directly before
// `item` — i.e. this path just opened a bracket at this position.
auto OpenerSynthesizedHere(llvm::ArrayRef<OpenBracketInfo> stack,
                           const Item& item) -> bool {
  return !stack.empty() && stack.back().is_synthetic &&
         stack.back().insertion_token_index == item.token.token_index;
}

// Whether a `]` or `}` was inserted directly before the current token. Such a
// closer repairs an illegal leaf adjacency (unlike `)`, these don't end a
// value). `Other` means no closer was inserted here.
auto CloserFixesLeafAdjacency(BracketTokenKind closer_inserted) -> bool {
  return closer_inserted == BracketTokenKind::CloseSquareBracket ||
         closer_inserted == BracketTokenKind::CloseCurlyBrace;
}

// Computes the cost of inserting a synthetic closer for `top` directly before
// `item`, or nullopt if this insertion isn't worth exploring. Sets `origin` to
// the rule that fired.
auto ClassifyCloserInsertion(const OpenBracketInfo& top, const Item& item,
                             llvm::ArrayRef<OpenBracketInfo> stack,
                             const char*& origin) -> std::optional<int32_t> {
  auto t_kind = item.token.kind;

  bool cascade = MatchesDeeperOpener(stack, t_kind);
  bool prev_is_value_like = PrevIsValueLike(item);

  if (top.kind == BracketTokenKind::OpenParen ||
      top.kind == BracketTokenKind::OpenSquareBracket) {
    // An empty group: the opener is directly followed by a `,`, which can't
    // start group content, so the group's closer must have been directly
    // after the opener (e.g. `f(x(), y)` becoming `f(x(, y)`); or by a
    // spaced `.`, which as group content would be written unspaced
    // (`f(.a = 1)`).
    if (top.token_pos >= 0 && top.token_pos == item.token_start_index - 1 &&
        (t_kind == BracketTokenKind::Comma ||
         (t_kind == BracketTokenKind::Period &&
          item.token.has_leading_space))) {
      origin = "Close_EmptyGroup";
      return CostCloseEmptyGroup;
    }
    if (t_kind == BracketTokenKind::Semi) {
      origin = "Close_ParenBeforeSemi";
      return CostCloseParenBeforeSemi;
    }
    if (cascade) {
      origin = "Close_ParenCascade";
      return CostCloseCascade;
    }
    // A `{` starting a block means the paren should have closed: `if (c) {`,
    // `while (c) {`. A struct-literal `{...}` can legitimately sit inside a
    // *call* paren (`f({.x = 1})`), but not inside a keyword or grouping paren
    // (whose `{` — even an empty `{}` misread as a struct — is a block).
    if (t_kind == BracketTokenKind::OpenCurlyBrace &&
        (!item.token.is_struct_brace || !top.is_call_paren)) {
      origin = "Close_ParenBeforeBrace";
      return CostCloseParenBeforeBrace;
    }
    // `=` can directly follow both `)` and `]`, but `->` and `as` only
    // plausibly follow `)`. An unspaced structural operator is not a cue:
    // formatted code spaces these operators, and an unspaced `->` is a
    // pointer member access (`p->x`).
    if (item.token.is_structural_op && item.token.has_leading_space &&
        (top.kind == BracketTokenKind::OpenParen ||
         item.token.is_assignment_op)) {
      origin = "Close_ParenBeforeStructuralOp";
      return CostCloseParenBeforeStructuralOp;
    }
    // A leaf directly following a value-ending token is illegal, and a `]`
    // between them fixes the adjacency (unlike `)`, `]` can be directly
    // followed by a leaf, as in `impl forall [...] T as ...`).
    if (top.kind == BracketTokenKind::OpenSquareBracket &&
        t_kind == BracketTokenKind::Leaf && item.token.prev_is_value_ending) {
      origin = "Close_SquareAtLeafAdjacency";
      return CostCloseSquareAtLeafAdjacency;
    }
    // A `.` with whitespace before it mid-line suggests a closer was deleted
    // right before it: member access is written without spaces, `x.y`.
    if (t_kind == BracketTokenKind::Period && !item.is_first_on_line &&
        item.token.has_leading_space && prev_is_value_like) {
      origin = "Close_ParenBeforeSpacedPeriod";
      return CostCloseParenBeforeSpacedPeriod;
    }
    // Similarly, a `(` or `[` with whitespace before it directly following a
    // value-ending token: calls and indexing are written without spaces.
    if ((t_kind == BracketTokenKind::OpenParen ||
         t_kind == BracketTokenKind::OpenSquareBracket) &&
        !item.is_first_on_line && item.token.has_leading_space &&
        prev_is_value_like) {
      origin = "Close_ParenBeforeSpacedOpen";
      return CostCloseParenBeforeSpacedPeriod;
    }
    // A comparison or logical operator is unlikely inside square brackets or
    // call/index argument lists (but common in `if (...)` etc.).
    if (item.token.is_comparison_op &&
        (top.kind == BracketTokenKind::OpenSquareBracket ||
         top.is_call_paren)) {
      origin = "Close_ParenBeforeComparison";
      return CostCloseParenBeforeStructuralOp;
    }
    // A `,` with whitespace before it: formatted code has no space before a
    // comma, so a closer was likely deleted in the gap.
    if (t_kind == BracketTokenKind::Comma && !item.is_first_on_line &&
        item.token.has_leading_space && prev_is_value_like) {
      origin = "Close_BeforeSpacedComma";
      return CostCloseParenBeforeSemi;
    }
    // Likewise a `)` or `]` with whitespace before it: formatted code has no
    // space before closers either.
    if ((t_kind == BracketTokenKind::CloseParen ||
         t_kind == BracketTokenKind::CloseSquareBracket) &&
        !item.is_first_on_line && item.token.has_leading_space &&
        prev_is_value_like) {
      origin = "Close_BeforeSpacedCloser";
      return CostCloseParenBeforeSemi;
    }
    if (t_kind == BracketTokenKind::FileEnd) {
      origin = "Close_ParenAtFileEnd";
      return CostCloseAtEnd;
    }
    // A wide whitespace gap mid-line suggests a deleted token in the gap.
    if (!item.is_first_on_line && item.token.has_wide_leading_space) {
      origin = "Close_ParenAtWideGap";
      return CostCloseAtWideGap;
    }
    // A `[` group rarely spans lines except in wrapped declaration headers
    // (`impl forall [...]` etc.), where the line break follows the `]`.
    if (top.kind == BracketTokenKind::OpenSquareBracket &&
        item.is_first_on_line && item.token.line != top.line) {
      origin = "Close_SquareAtContinuation";
      return CostCloseSquareAtContinuation;
    }
    // No positive cue that a `(`/`[` closes here. Closing before a bare
    // dedent, statement introducer, or arbitrary token was never a correct
    // guess in practice (it just closes too early), so decline: the search
    // will close at a real cue, at the region end, or, failing both, replace
    // the unmatched opener with an error token.
    return std::nullopt;
  }

  if (top.is_struct_brace) {
    if (t_kind == BracketTokenKind::Semi) {
      origin = "Close_StructBeforeSemi";
      return CostCloseParenBeforeSemi;
    }
    if (cascade) {
      origin = "Close_StructCascade";
      return CostCloseCascade;
    }
    if (!item.is_first_on_line && item.token.has_wide_leading_space) {
      origin = "Close_StructAtWideGap";
      return CostCloseAtWideGap;
    }
    if (t_kind == BracketTokenKind::FileEnd) {
      origin = "Close_StructAtFileEnd";
      return CostCloseAtEnd;
    }
    if (item.is_first_on_line &&
        item.token.line_indent <= top.effective_header_indent) {
      origin = "Close_StructAtDedent";
      return CostCloseStructAtDedent;
    }
    origin = "Close_StructBaseline";
    return CostCloseParenBaseline;
  }

  // Scope brace.
  if (item.is_first_on_line &&
      item.token.line_indent <= top.effective_header_indent) {
    origin = "Close_ScopeAtDedent";
    return CostCloseScopeAtDedent;
  }
  // A first-on-line `else` normally directly follows a `}` on the same line,
  // so one was likely deleted before it.
  if (item.token.is_else_keyword && item.is_first_on_line) {
    origin = "Close_ScopeBeforeElse";
    return CostCloseScopeBeforeElse;
  }
  if (cascade) {
    origin = "Close_ScopeCascade";
    return CostCloseCascade;
  }
  if (t_kind == BracketTokenKind::FileEnd) {
    origin = "Close_ScopeAtFileEnd";
    return CostCloseAtEnd;
  }
  origin = "Close_ScopeBaseline";
  return CostCloseScopeBaseline;
}

// Computes the cost of inserting a synthetic opener of kind `kind` directly
// before `item`. `prev_item` is the preceding item, if any. Sets `origin` to
// the rule that fired.
auto ClassifyOpenerInsertion(BracketTokenKind kind, const Item& item,
                             const Item* prev_item, const char*& origin)
    -> int32_t {
  auto t_kind = item.token.kind;

  if (kind == BracketTokenKind::OpenParen ||
      kind == BracketTokenKind::OpenSquareBracket) {
    // `if`/`while`/`for`/`match` (statement introducers) require a following
    // `(`; `forall` (an Other token) requires a following `[`.
    if (item.prev_is_paren_keyword && !IsOpeningBracket(t_kind) &&
        kind == (item.prev_kind == BracketTokenKind::StatementIntroducer
                     ? BracketTokenKind::OpenParen
                     : BracketTokenKind::OpenSquareBracket)) {
      origin = "Open_AfterParenKeyword";
      return CostOpenAfterParenKeyword;
    }
    // A leaf or binding modifier directly following a value-ending token is
    // illegal; an opener here fixes the adjacency.
    if ((t_kind == BracketTokenKind::Leaf ||
         item.token.is_modifier_keyword) &&
        item.token.prev_is_value_ending) {
      origin = "Open_AtLeafAdjacency";
      return CostOpenAtLeafAdjacency;
    }
    // A `.` with whitespace before it directly following a value-ending
    // token: likely a designator argument that lost its `(`, as in
    // `ImplicitAs(.Self)`.
    if (t_kind == BracketTokenKind::Period && !item.is_first_on_line &&
        item.token.has_leading_space && item.token.prev_is_value_ending &&
        kind == BracketTokenKind::OpenParen) {
      origin = "Open_BeforeSpacedPeriod";
      return CostCloseParenBeforeSpacedPeriod;
    }
    // A mid-line leaf with whitespace before it directly following an
    // unspaced `.`: member access is written without spaces, `x.y`, so a
    // bracket was likely deleted in the gap.
    if (t_kind == BracketTokenKind::Leaf && !item.is_first_on_line &&
        item.token.has_leading_space &&
        item.prev_kind == BracketTokenKind::Period &&
        !item.prev_has_leading_space) {
      origin = "Open_AfterPeriodGap";
      return CostOpenAfterPeriodGap;
    }
    // A mid-line leaf with whitespace before it directly following an
    // opener: formatted code has no space after `(` or `[`, so a bracket was
    // likely deleted in the gap.
    if (t_kind == BracketTokenKind::Leaf && !item.is_first_on_line &&
        item.token.has_leading_space &&
        (item.prev_kind == BracketTokenKind::OpenParen ||
         item.prev_kind == BracketTokenKind::OpenSquareBracket)) {
      origin = "Open_AfterOpenGap";
      return CostOpenAfterPeriodGap;
    }
    // A wide whitespace gap before a token that could start a group suggests
    // an opener was deleted in the gap.
    if (!item.is_first_on_line && item.token.has_wide_leading_space &&
        (t_kind == BracketTokenKind::Leaf ||
         t_kind == BracketTokenKind::Period || IsOpeningBracket(t_kind) ||
         item.token.is_modifier_keyword)) {
      origin = "Open_AtWideGap";
      return CostCloseAtWideGap;
    }
    // An empty group directly after a name: `Op()`. Only applies after a
    // leaf (a call of a just-computed value, `f(x)()`, is much rarer than a
    // call of a name), and not when the name is a type after `as` or `->`,
    // where a parenthesized group is more plausible than an empty call.
    if (prev_item != nullptr &&
        prev_item->token.kind == BracketTokenKind::Leaf &&
        item.token.has_leading_space &&
        !(prev_item->prev_is_structural_op &&
          !prev_item->prev_is_assignment_op)) {
      if (t_kind == BracketTokenKind::CloseParen &&
          kind == BracketTokenKind::OpenParen) {
        origin = "Open_EmptyParens";
        return CostOpenEmptyParens;
      }
      if (t_kind == BracketTokenKind::CloseSquareBracket &&
          kind == BracketTokenKind::OpenSquareBracket) {
        origin = "Open_EmptySquares";
        return CostOpenEmptySquares;
      }
    }
    // An empty call of a just-computed value: `T.(Default.Op)()`. Only
    // trusted when the `)` is spaced, marking the deletion gap. `prev_kind`
    // is the last token of the previous item, so this fires after a collapsed
    // `(...)` block too.
    if (item.prev_kind == BracketTokenKind::CloseParen &&
        t_kind == BracketTokenKind::CloseParen &&
        kind == BracketTokenKind::OpenParen && item.token.has_leading_space &&
        !item.is_first_on_line) {
      origin = "Open_EmptyParensAfterClose";
      return CostOpenEmptyParensAfterClose;
    }
    origin = "Open_ParenBaseline";
    return CostOpenParenBaseline;
  }

  // Scope or struct `{`.
  origin = "Open_BraceBaseline";
  return CostOpenBraceBaseline;
}

// From the optimal goal nodes, reconstructs the repair corrections and appends
// them to `corrections`. Every optimal repair is enumerated (up to a cap); a
// correction that the optimal repairs disagree about is marked tied, so the
// caller downgrades it to an error token rather than guessing. Falls back to
// naive recovery if no path can be reconstructed.
auto ReconstructCorrections(
    llvm::ArrayRef<BeamNode> arena, llvm::ArrayRef<int32_t> goal_node_indices,
    llvm::ArrayRef<Item> items, TokenIndex region_end_token,
    llvm::SmallVectorImpl<BracketCorrection>& corrections) -> void {
  // Walk parent edges from each goal back to the root, collecting the
  // corrections along each distinct optimal path.
  llvm::SmallVector<llvm::SmallVector<BracketCorrection>> all_paths;
  llvm::SmallVector<BracketCorrection> current_path;
  auto dfs = [&](auto& self, int32_t node_idx) -> void {
    if (all_paths.size() >= 100) {
      return;
    }
    const auto& node = arena[node_idx];
    if (node.parent_edges.empty()) {
      auto path = current_path;
      std::reverse(path.begin(), path.end());
      all_paths.push_back(std::move(path));
      return;
    }
    for (const auto& edge : node.parent_edges) {
      if (edge.has_correction) {
        current_path.push_back(edge.correction);
      }
      self(self, edge.parent_node_index);
      if (edge.has_correction) {
        current_path.pop_back();
      }
    }
  };
  for (int32_t goal_idx : goal_node_indices) {
    dfs(dfs, goal_idx);
  }

  if (all_paths.empty()) {
    SolveNaive(items, corrections);
    return;
  }

  // Two insertions of the same bracket kind are equivalent if every token
  // between their insertion points is that same kind: inserting a `)` on
  // either side of an existing `)` produces the same token sequence.
  llvm::DenseMap<int32_t, int32_t> token_to_item;
  for (auto [idx, region_item] : llvm::enumerate(items)) {
    token_to_item[region_item.token.token_index.index] =
        static_cast<int32_t>(idx);
  }
  token_to_item[region_end_token.index] = static_cast<int32_t>(items.size());
  // Compares only the fixes, not the diagnosed brackets: two paths that
  // blame different brackets but repair the token stream identically don't
  // disagree about the repair.
  auto corrections_equivalent = [&](const BracketCorrection& a,
                                    const BracketCorrection& b) -> bool {
    if (a.fix_action != b.fix_action ||
        a.fix_token_kind != b.fix_token_kind) {
      return false;
    }
    if (a.fix_token_index == b.fix_token_index) {
      return true;
    }
    if (a.fix_action != BracketFixAction::InsertBefore) {
      return false;
    }
    auto a_it = token_to_item.find(a.fix_token_index.index);
    auto b_it = token_to_item.find(b.fix_token_index.index);
    if (a_it == token_to_item.end() || b_it == token_to_item.end()) {
      return false;
    }
    auto [lo, hi] = std::minmax(a_it->second, b_it->second);
    for (int32_t p = lo; p < hi; ++p) {
      if (items[p].is_collapsed_block ||
          ToTokenKind(items[p].token.kind) != a.fix_token_kind) {
        return false;
      }
    }
    return true;
  };

  // Match each baseline correction against an equivalent one in every other
  // optimal path; corrections with no counterpart in some path are tied.
  llvm::ArrayRef<BracketCorrection> baseline_path = all_paths.front();
  llvm::SmallVector<bool> tied(baseline_path.size(), false);
  for (const auto& path : all_paths) {
    llvm::SmallVector<bool> used(path.size(), false);
    for (auto [corr_idx, corr] : llvm::enumerate(baseline_path)) {
      bool found = false;
      for (auto [path_idx, path_corr] : llvm::enumerate(path)) {
        if (!used[path_idx] && corrections_equivalent(path_corr, corr)) {
          used[path_idx] = true;
          found = true;
          break;
        }
      }
      if (!found) {
        tied[corr_idx] = true;
      }
    }
  }

  for (auto [corr_idx, corr] : llvm::enumerate(baseline_path)) {
    corrections.push_back(corr);
    corrections.back().is_tied = tied[corr_idx];
  }
}

// Solve a damaged region using layered beam search with tie detection.
// `region_end_token` and `region_end_byte` identify the token directly after
// the region, where any still-unclosed brackets are closed.
auto SolveRegionCostBased(llvm::ArrayRef<Item> items,
                          TokenIndex region_end_token, int32_t region_end_byte,
                          llvm::SmallVectorImpl<BracketCorrection>& corrections)
    -> void {
  if (items.size() > static_cast<size_t>(MaxRegionItemsForSearch)) {
    SolveNaive(items, corrections);
    return;
  }

  llvm::SmallVector<BeamNode, 0> arena;
  arena.reserve(256);

  int32_t min_goal_cost = std::numeric_limits<int32_t>::max();

  auto try_add_to_layer =
      [&](llvm::SmallVectorImpl<int32_t>& layer_indices,
          llvm::DenseMap<uint64_t, int32_t>& layer_map, int32_t next_item_idx,
          llvm::SmallVector<OpenBracketInfo, 4> next_stack,
          BracketTokenKind closer_inserted, int32_t next_cost, ParentEdge edge,
          llvm::SmallVectorImpl<int32_t>* worklist = nullptr) {
        if (next_cost > min_goal_cost) {
          return;
        }
        auto merge_into = [&](int32_t idx) {
          auto& exist_node = arena[idx];
          if (next_cost < exist_node.cost) {
            exist_node.cost = next_cost;
            exist_node.parent_edges.clear();
            exist_node.parent_edges.push_back(edge);
            if (worklist) {
              worklist->push_back(idx);
            }
          } else if (next_cost == exist_node.cost) {
            if (llvm::none_of(exist_node.parent_edges,
                              [&](const ParentEdge& e) {
                                return EdgesEqual(e, edge);
                              })) {
              exist_node.parent_edges.push_back(edge);
            }
          }
        };
        uint64_t stack_hash = StateHash(next_stack, closer_inserted);
        auto map_it = layer_map.find(stack_hash);
        if (map_it != layer_map.end()) {
          if (arena[map_it->second].stack == next_stack &&
              arena[map_it->second].closer_inserted == closer_inserted) {
            merge_into(map_it->second);
            return;
          }
          // On hash collision, fall back to a linear scan over the layer.
          for (int32_t idx : layer_indices) {
            if (arena[idx].stack == next_stack &&
                arena[idx].closer_inserted == closer_inserted) {
              merge_into(idx);
              return;
            }
          }
        }
        int32_t new_idx = static_cast<int32_t>(arena.size());
        arena.push_back(BeamNode{
            .item_index = next_item_idx,
            .stack = std::move(next_stack),
            .cost = next_cost,
            .closer_inserted = closer_inserted,
            .parent_edges = {edge},
        });
        layer_indices.push_back(new_idx);
        layer_map[stack_hash] = new_idx;
        if (worklist) {
          worklist->push_back(new_idx);
        }
      };

  // Keeps a layer within the beam width by discarding the costliest states.
  auto prune_beam = [&](llvm::SmallVectorImpl<int32_t>& layer) {
    if (layer.size() > MaxBeamWidth) {
      llvm::stable_sort(layer, [&](int32_t a, int32_t b) {
        return arena[a].cost < arena[b].cost;
      });
      layer.resize(MaxBeamWidth);
    }
  };

  arena.push_back(BeamNode{
      .item_index = 0,
      .stack = {},
      .cost = 0,
      .parent_edges = {},
  });
  llvm::SmallVector<int32_t> current_layer = {0};
  llvm::DenseMap<uint64_t, int32_t> layer_map;

  for (int32_t i = 0; i < static_cast<int32_t>(items.size()); ++i) {
    const auto& item = items[i];
    auto kind = item.token.kind;

    // Step 1: Epsilon moves within layer `i` (insertions before token `i`).
    if (kind != BracketTokenKind::FileEnd) {
      for (int32_t idx : current_layer) {
        layer_map[StateHash(arena[idx].stack, arena[idx].closer_inserted)] =
            idx;
      }

      // Phase 1a: Synthetic closers. Uses a worklist so that several groups
      // can be closed at the same point.
      llvm::SmallVector<int32_t> worklist = current_layer;
      size_t worklist_head = 0;

      while (worklist_head < worklist.size()) {
        int32_t node_idx = worklist[worklist_head++];
        const SearchState current = Snapshot(arena[node_idx]);
        if (current.cost > min_goal_cost) {
          continue;
        }
        if (current.stack.empty()) {
          continue;
        }
        const auto& top = current.stack.back();
        // Synthetic openers exist only to consume real closers; closing one
        // synthetically would insert a pointless empty pair.
        if (top.is_synthetic) {
          continue;
        }
        // If the current token is the matching closer and would be allowed
        // to match directly, matching is strictly better than synthesizing a
        // duplicate closer in front of it — unless the closer has suspicious
        // whitespace before it suggesting a closer was deleted in the gap.
        if (kind == MatchingClosingKind(top.kind)) {
          bool direct_match_ok = true;
          if (kind == BracketTokenKind::CloseCurlyBrace &&
              !top.is_struct_brace && item.token.line != top.line &&
              item.token.line_indent < top.effective_header_indent) {
            direct_match_ok = false;
          }
          bool spaced_suspicious =
              kind != BracketTokenKind::CloseCurlyBrace &&
              !item.is_first_on_line && item.token.has_leading_space &&
              PrevIsValueLike(item);
          if (direct_match_ok && !spaced_suspicious) {
            continue;
          }
        }
        const char* origin = "";
        auto eps_cost =
            ClassifyCloserInsertion(top, item, current.stack, origin);
        if (!eps_cost) {
          continue;
        }

        auto next_stack = current.stack;
        auto popped = next_stack.pop_back_val();
        auto closer_kind = MatchingClosingKind(popped.kind);
        ParentEdge edge{
            .parent_node_index = node_idx,
            .correction =
                BracketCorrection{
                    .diagnostic_kind = BracketDiagnosticKind::UnmatchedOpening,
                    .diagnostic_token_index = popped.token_index,
                    .fix_action = BracketFixAction::InsertBefore,
                    .fix_token_index = item.token.token_index,
                    .fix_token_kind = ToTokenKind(closer_kind),
                    .fix_byte_offset = item.token.byte_offset,
                    .origin = origin,
                },
            .has_correction = true,
        };
        try_add_to_layer(current_layer, layer_map, i, std::move(next_stack),
                         closer_kind, current.cost + *eps_cost, edge,
                         &worklist);
      }

      // Phase 1b: Synthetic openers. Iterate only over the states present
      // after Phase 1a, without chaining synthetic openers onto each other.
      size_t num_states_after_1a = current_layer.size();
      for (size_t idx = 0; idx < num_states_after_1a; ++idx) {
        int32_t node_idx = current_layer[idx];
        const SearchState current = Snapshot(arena[node_idx]);
        if (current.cost > min_goal_cost ||
            current.stack.size() >= MaxSearchStackDepth) {
          continue;
        }

        auto push_synthetic = [&](BracketTokenKind open_kind,
                                  bool is_struct_brace, int32_t add_cost,
                                  const char* origin) {
          auto next_stack = current.stack;
          next_stack.push_back(OpenBracketInfo{
              .token_index = TokenIndex::None,
              .token_pos = -1,
              .kind = open_kind,
              .line = item.token.line,
              .line_indent = item.token.line_indent,
              .effective_header_indent = item.effective_header_indent,
              .is_synthetic = true,
              .is_struct_brace = is_struct_brace,
              .insertion_token_index = item.token.token_index,
              .insertion_byte_offset = item.token.byte_offset,
              .origin = origin,
          });
          ParentEdge edge{
              .parent_node_index = node_idx,
              .correction = {},
              .has_correction = false,
          };
          try_add_to_layer(current_layer, layer_map, i, std::move(next_stack),
                           current.closer_inserted, current.cost + add_cost,
                           edge, nullptr);
        };

        // Synthetic `(` and `[`.
        const Item* prev_item = i > 0 ? &items[i - 1] : nullptr;
        {
          const char* origin = "";
          int32_t cost = ClassifyOpenerInsertion(BracketTokenKind::OpenParen,
                                                 item, prev_item, origin);
          push_synthetic(BracketTokenKind::OpenParen, false, cost, origin);
        }
        {
          const char* origin = "";
          int32_t cost = ClassifyOpenerInsertion(
              BracketTokenKind::OpenSquareBracket, item, prev_item, origin);
          push_synthetic(BracketTokenKind::OpenSquareBracket, false, cost,
                         origin);
        }
        // Synthetic scope `{` between an unbraced header and its body.
        if (item.follows_statement_header &&
            !item.header_has_open_curly_brace && !IsOpeningBracket(kind)) {
          push_synthetic(BracketTokenKind::OpenCurlyBrace, false,
                         CostOpenScopeAfterHeader, "Open_ScopeAfterHeader");
        } else if (!IsOpeningBracket(kind)) {
          push_synthetic(BracketTokenKind::OpenCurlyBrace, false,
                         CostOpenBraceBaseline, "Open_ScopeBaseline");
        }
        // Synthetic struct `{`.
        if (kind == BracketTokenKind::Period &&
            !item.token.prev_is_value_ending) {
          push_synthetic(BracketTokenKind::OpenCurlyBrace, true,
                         CostOpenStructBeforeDesignator,
                         "Open_StructBeforeDesignator");
        } else if (kind == BracketTokenKind::CloseCurlyBrace &&
                   !item.is_first_on_line &&
                   (item.token.prev_is_value_ending ||
                    item.prev_kind == BracketTokenKind::CloseCurlyBrace ||
                    item.prev_kind == BracketTokenKind::CloseSquareBracket ||
                    item.prev_kind == BracketTokenKind::Comma)) {
          // A struct literal `{...}` that lost its `{`, leaving content
          // directly before the `}`. Requires real content before the `}`, so
          // a stray `}` is reported as an error instead.
          push_synthetic(BracketTokenKind::OpenCurlyBrace, true,
                         CostOpenStructEmptyBraces, "Open_StructEmptyBraces");
        }
      }

      layer_map.clear();
      prune_beam(current_layer);
    }

    // Step 2: Advance moves from layer `i` to layer `i + 1` (consuming token
    // `i`).
    llvm::SmallVector<int32_t> next_layer;

    for (int32_t node_idx : current_layer) {
      const SearchState current = Snapshot(arena[node_idx]);
      if (current.cost > min_goal_cost) {
        continue;
      }

      auto try_enqueue_advance =
          [&](llvm::SmallVector<OpenBracketInfo, 4> next_stack,
              int32_t add_cost, BracketCorrection correction = {},
              bool has_correction = false) {
            ParentEdge edge{
                .parent_node_index = node_idx,
                .correction = correction,
                .has_correction = has_correction,
            };
            try_add_to_layer(next_layer, layer_map, i + 1,
                             std::move(next_stack), BracketTokenKind::Other,
                             current.cost + add_cost, edge, nullptr);
          };

      auto top_kind = GetTopKind(current.stack);
      bool top_paren_like =
          top_kind == TopKind::Paren || top_kind == TopKind::StructBrace;
      bool prev_is_value_like = PrevIsValueLike(item);
      // Whether some bracket was inserted directly before this token, so its
      // suspicious leading whitespace is already explained.
      bool bracket_inserted_here =
          current.closer_inserted != BracketTokenKind::Other ||
          OpenerSynthesizedHere(current.stack, item);

      if (item.is_collapsed_block) {
        int32_t penalty = ContextPenalty(current, item);
        if (top_paren_like && item.contains_scope_brace) {
          penalty += CostScopeBraceInParen;
        }
        try_enqueue_advance(current.stack, penalty);
        continue;
      }

      if (IsOpeningBracket(kind)) {
        // Advance and push opener onto stack.
        if (current.stack.size() < MaxSearchStackDepth) {
          auto next_stack = current.stack;
          int32_t penalty = ContextPenalty(current, item);
          if (kind == BracketTokenKind::OpenCurlyBrace && top_paren_like) {
            penalty += item.token.is_struct_brace ? CostStructBraceInParen
                                                  : CostScopeBraceInParen;
          }
          // A spaced `(` or `[` directly after a value-ending token is
          // illegal-looking (calls and indexing are written without spaces),
          // unless a closer was just inserted between them.
          if (kind != BracketTokenKind::OpenCurlyBrace &&
              !item.is_first_on_line && item.token.has_leading_space &&
              prev_is_value_like &&
              current.closer_inserted == BracketTokenKind::Other) {
            penalty += CostSpacedOpenAfterValue;
          }
          next_stack.push_back(OpenBracketInfo{
              .token_index = item.token.token_index,
              .token_pos = item.token_start_index,
              .kind = kind,
              .line = item.token.line,
              .line_indent = item.token.line_indent,
              .effective_header_indent = item.effective_header_indent,
              .is_synthetic = false,
              .is_struct_brace = item.token.is_struct_brace,
              .is_call_paren = item.token.prev_is_value_ending,
              .insertion_byte_offset = item.token.byte_offset,
          });
          try_enqueue_advance(std::move(next_stack), penalty);
        }

        // Advance without pushing (replace unmatched opener with Error token).
        try_enqueue_advance(
            current.stack, CostReplaceOpening,
            ReplaceWithError(item.token, BracketDiagnosticKind::UnmatchedOpening,
                             "Adv_ReplaceOpener"),
            /*has_correction=*/true);
        continue;
      }

      if (IsClosingBracket(kind)) {
        // If matches top(stack): advance and pop stack.
        if (!current.stack.empty() &&
            current.stack.back().kind == MatchingOpeningKind(kind)) {
          const auto& top = current.stack.back();
          bool allow_match = true;
          int32_t penalty = 0;
          if (kind == BracketTokenKind::CloseCurlyBrace &&
              !top.is_struct_brace && item.token.line != top.line) {
            // A multi-line scope close must not be dedented past its header,
            // and pays for indentation disagreement with its header.
            if (item.token.line_indent < top.effective_header_indent) {
              allow_match = false;
            } else if (item.is_first_on_line &&
                       item.token.line_indent != top.effective_header_indent) {
              penalty +=
                  CostBraceIndentMismatchBase +
                  CostBraceIndentMismatchPerColumn *
                      std::abs(top.effective_header_indent -
                               item.token.line_indent);
            }
          }
          if (allow_match) {
            auto next_stack = current.stack;
            auto popped = next_stack.pop_back_val();
            // A spaced `)` or `]` suggests a deleted token in the gap; only
            // waived if this path inserted a bracket there.
            if (kind != BracketTokenKind::CloseCurlyBrace &&
                !item.is_first_on_line && item.token.has_leading_space &&
                prev_is_value_like && !bracket_inserted_here) {
              penalty += CostSpacedCloserUnexplained;
            }
            BracketCorrection correction;
            bool has_corr = false;
            if (popped.is_synthetic) {
              correction = BracketCorrection{
                  .diagnostic_kind = BracketDiagnosticKind::UnmatchedClosing,
                  .diagnostic_token_index = item.token.token_index,
                  .fix_action = BracketFixAction::InsertBefore,
                  .fix_token_index = popped.insertion_token_index,
                  .fix_token_kind = ToTokenKind(popped.kind),
                  .fix_byte_offset = popped.insertion_byte_offset,
                  .origin = popped.origin,
              };
              has_corr = true;
            }
            try_enqueue_advance(std::move(next_stack), penalty, correction,
                                has_corr);
          }
        }

        // Advance without matching/popping (replace unmatched closer with
        // Error token).
        try_enqueue_advance(
            current.stack, CostReplaceClosing,
            ReplaceWithError(item.token, BracketDiagnosticKind::UnmatchedClosing,
                             "Adv_ReplaceCloser"),
            /*has_correction=*/true);
        continue;
      }

      // A leaf or binding-modifier token can't directly follow a value; an
      // opener synthesized here, or a `]`/`}` inserted here, repairs it.
      auto leaf_adjacency_penalty = [&]() -> int32_t {
        return item.token.prev_is_value_ending &&
                       !OpenerSynthesizedHere(current.stack, item) &&
                       !CloserFixesLeafAdjacency(current.closer_inserted)
                   ? CostLeafAdjacency
                   : 0;
      };

      // Non-bracket token.
      int32_t penalty = ContextPenalty(current, item);
      switch (kind) {
        case BracketTokenKind::Semi:
          if (top_paren_like) {
            penalty += CostSemiInParen;
          }
          break;
        case BracketTokenKind::Comma:
          if (top_kind == TopKind::None || top_kind == TopKind::ScopeBrace) {
            penalty += CostCommaAtStatementLevel;
          } else if (!current.stack.empty() &&
                     current.stack.back().token_pos ==
                         item.token_start_index - 1) {
            // A `,` directly following a still-open `(`/`[` is illegal.
            penalty += CostCommaAfterOpen;
          } else if (!item.is_first_on_line &&
                     item.token.has_leading_space && prev_is_value_like &&
                     current.closer_inserted == BracketTokenKind::Other) {
            // Formatted code has no space before a `,`; a bracket was likely
            // deleted in the gap.
            penalty += CostSpacedCommaInParen;
          }
          break;
        case BracketTokenKind::StatementIntroducer:
          if (top_paren_like) {
            penalty += CostIntroducerInParen;
          }
          break;
        case BracketTokenKind::Leaf:
          penalty += leaf_adjacency_penalty();
          break;
        case BracketTokenKind::Other:
          if (item.token.is_structural_op && item.token.has_leading_space &&
              top_kind == TopKind::Paren) {
            // Casts `(x as T)` are common inside parens; the other
            // structural operators are much rarer there.
            penalty += item.token.is_as_op ? CostAsOpInParen
                                           : CostStructuralOpInParen;
          }
          if (item.token.is_comparison_op && !current.stack.empty() &&
              (current.stack.back().kind ==
                   BracketTokenKind::OpenSquareBracket ||
               (top_kind == TopKind::Paren &&
                current.stack.back().is_call_paren))) {
            penalty += CostComparisonInSquare;
          }
          if (!item.is_first_on_line && item.token.has_wide_leading_space &&
              !bracket_inserted_here) {
            // A wide whitespace gap mid-line suggests a deleted bracket that
            // this path hasn't repaired.
            penalty += CostWideGapUnexplained;
          }
          // Like a leaf, a binding modifier keyword can't directly follow a
          // value-ending token.
          if (item.token.is_modifier_keyword) {
            penalty += leaf_adjacency_penalty();
          }
          break;
        case BracketTokenKind::Period:
          // A mid-line `.` with whitespace before it suggests a deleted
          // bracket; prefer closing an open group before it, or opening one.
          // If we just inserted a bracket here, the gap is explained.
          if (!item.is_first_on_line && item.token.has_leading_space &&
              (prev_is_value_like || IsOpeningBracket(item.prev_kind)) &&
              !bracket_inserted_here) {
            penalty += CostSpacedPeriodInParen;
          }
          break;
        default:
          break;
      }
      try_enqueue_advance(current.stack, penalty);
    }

    // Step 3: Beam Pruning.
    layer_map.clear();
    prune_beam(next_layer);
    current_layer = std::move(next_layer);
  }

  // Finish: close any remaining real open brackets at the region end.
  llvm::SmallVector<int32_t> goal_node_indices;

  for (int32_t node_idx : current_layer) {
    const SearchState current = Snapshot(arena[node_idx]);
    if (current.cost > min_goal_cost) {
      continue;
    }
    // A synthetic opener that never matched a real closer is a meaningless
    // insertion; reject such states rather than dropping it silently.
    bool has_unmatched_synthetic = false;
    for (const auto& entry : current.stack) {
      if (entry.is_synthetic) {
        has_unmatched_synthetic = true;
        break;
      }
    }
    if (has_unmatched_synthetic) {
      continue;
    }
    int32_t finish_cost = current.cost;
    int32_t parent = node_idx;
    for (const auto& entry : llvm::reverse(current.stack)) {
      if (entry.kind == BracketTokenKind::OpenCurlyBrace) {
        finish_cost +=
            entry.is_struct_brace ? CostCloseStructAtEnd : CostCloseAtEnd;
      } else {
        finish_cost += CostCloseParenAtEnd;
      }
      int32_t new_idx = static_cast<int32_t>(arena.size());
      arena.push_back(BeamNode{
          .item_index = static_cast<int32_t>(items.size()),
          .stack = {},
          .cost = finish_cost,
          .parent_edges = {{
              .parent_node_index = parent,
              .correction =
                  BracketCorrection{
                      .diagnostic_kind = BracketDiagnosticKind::
                          UnmatchedOpening,
                      .diagnostic_token_index = entry.token_index,
                      .fix_action = BracketFixAction::InsertBefore,
                      .fix_token_index = region_end_token,
                      .fix_token_kind =
                          ToTokenKind(MatchingClosingKind(entry.kind)),
                      .fix_byte_offset = region_end_byte,
                      .origin = "Close_RegionEnd"},
              .has_correction = true,
          }},
      });
      parent = new_idx;
    }

    if (finish_cost < min_goal_cost) {
      min_goal_cost = finish_cost;
      goal_node_indices.clear();
    }
    if (finish_cost == min_goal_cost) {
      goal_node_indices.push_back(parent);
    }
  }

  if (goal_node_indices.empty()) {
    SolveNaive(items, corrections);
    return;
  }
  ReconstructCorrections(arena, goal_node_indices, items, region_end_token,
                         corrections);
}

}  // namespace

auto FixMismatchedBrackets(llvm::ArrayRef<MismatchedBracketToken> tokens)
    -> llvm::SmallVector<BracketCorrection> {
  llvm::SmallVector<BracketCorrection> corrections;
  if (tokens.empty()) {
    return corrections;
  }

  auto num_tokens = static_cast<int32_t>(tokens.size());

  // 1. Initial pass to find matched bracket pairs. A closer that doesn't match
  // the top of the stack pops through to a plausible match if one exists
  // (leaving the popped brackets unmatched), and is otherwise left unmatched
  // without disturbing the stack. `}` matches a `{` only when they're on the
  // same line, the `{` is a struct brace, or their line indentation agrees.
  llvm::SmallVector<int32_t> open_stack;
  llvm::SmallVector<int32_t> match_partner(num_tokens, -1);
  llvm::SmallVector<bool> is_clean_range(num_tokens, false);

  for (int32_t i = 0; i < num_tokens; ++i) {
    auto kind = tokens[i].kind;
    if (IsOpeningBracket(kind)) {
      open_stack.push_back(i);
    } else if (IsClosingBracket(kind)) {
      int32_t match_s = -1;
      if (kind == BracketTokenKind::CloseCurlyBrace) {
        for (int32_t s = static_cast<int32_t>(open_stack.size()) - 1; s >= 0;
             --s) {
          int32_t cand = open_stack[s];
          if (tokens[cand].kind != BracketTokenKind::OpenCurlyBrace) {
            continue;
          }
          if (tokens[cand].line == tokens[i].line ||
              tokens[cand].is_struct_brace ||
              tokens[cand].line_indent == tokens[i].line_indent) {
            match_s = s;
            break;
          }
        }
      } else {
        auto req = MatchingOpeningKind(kind);
        for (int32_t s = static_cast<int32_t>(open_stack.size()) - 1; s >= 0;
             --s) {
          if (tokens[open_stack[s]].kind == req) {
            match_s = s;
            break;
          }
        }
      }
      if (match_s != -1) {
        int32_t open_idx = open_stack[match_s];
        match_partner[open_idx] = i;
        match_partner[i] = open_idx;
        open_stack.resize(match_s);
      }
    }
  }

  // 2. Pre-pass: compute per-token segment ids (segments are separated by
  // `;`, `{`, and `}`), associated indentation, and header relationships.
  llvm::SmallVector<int32_t> seg_id(num_tokens, 0);
  llvm::SmallVector<int32_t> seg_first(num_tokens, 0);
  for (int32_t i = 1; i < num_tokens; ++i) {
    auto prev_kind = tokens[i - 1].kind;
    bool new_seg = prev_kind == BracketTokenKind::Semi ||
                   prev_kind == BracketTokenKind::OpenCurlyBrace ||
                   prev_kind == BracketTokenKind::CloseCurlyBrace;
    seg_id[i] = seg_id[i - 1] + (new_seg ? 1 : 0);
    seg_first[i] = new_seg ? i : seg_first[i - 1];
  }

  // Sorted lists of unmatched openers and closers, by kind, for the
  // cleanliness checks below.
  llvm::SmallVector<int32_t> unmatched_open_parens;
  llvm::SmallVector<int32_t> unmatched_open_squares;
  llvm::SmallVector<int32_t> unmatched_close_parens;
  llvm::SmallVector<int32_t> unmatched_close_squares;
  for (int32_t i = 0; i < num_tokens; ++i) {
    if (match_partner[i] != -1) {
      continue;
    }
    switch (tokens[i].kind) {
      case BracketTokenKind::OpenParen:
        unmatched_open_parens.push_back(i);
        break;
      case BracketTokenKind::OpenSquareBracket:
        unmatched_open_squares.push_back(i);
        break;
      case BracketTokenKind::CloseParen:
        unmatched_close_parens.push_back(i);
        break;
      case BracketTokenKind::CloseSquareBracket:
        unmatched_close_squares.push_back(i);
        break;
      default:
        break;
    }
  }

  llvm::SmallVector<int32_t> effective_header_indent(num_tokens, 0);
  llvm::SmallVector<bool> is_first_on_line(num_tokens, false);
  llvm::SmallVector<bool> follows_statement_header(num_tokens, false);
  llvm::SmallVector<bool> header_has_open_curly_brace(num_tokens, false);

  for (int32_t i = 0; i < num_tokens; ++i) {
    is_first_on_line[i] = (i == 0 || tokens[i].line != tokens[i - 1].line);
    effective_header_indent[i] =
        ComputeAssociatedLineIndent(tokens, match_partner, i);
    follows_statement_header[i] =
        ComputeFollowsStatementHeader(tokens, match_partner, i);
    if (follows_statement_header[i]) {
      header_has_open_curly_brace[i] =
          ComputeHeaderHasOpenCurlyBrace(tokens, match_partner, i);
    }
  }

  // Returns true if `list` contains an element in [lo, hi].
  auto contains_in_range = [](llvm::ArrayRef<int32_t> list, int32_t lo,
                              int32_t hi) {
    const auto* it = std::lower_bound(list.begin(), list.end(), lo);
    return it != list.end() && *it <= hi;
  };

  // 3. Mark clean subranges for safe collapsing (processed in reverse order
  // so inner ranges are evaluated before enclosing outer ranges).
  for (int32_t i = num_tokens - 1; i >= 0; --i) {
    auto kind = tokens[i].kind;
    if (match_partner[i] == -1 || match_partner[i] <= i) {
      continue;
    }
    int32_t close_idx = match_partner[i];
    bool clean = true;

    if (kind == BracketTokenKind::OpenCurlyBrace) {
      if (tokens[i].line != tokens[close_idx].line) {
        if (!tokens[i].is_struct_brace &&
            (!tokens[i].is_at_end_of_line ||
             effective_header_indent[i] != tokens[close_idx].line_indent)) {
          clean = false;
        } else if (tokens[i].is_struct_brace &&
                   tokens[close_idx].line_indent <
                       effective_header_indent[i]) {
          clean = false;
        }
      }
      // A `;` directly inside a struct brace, or a `,` directly inside a
      // scope brace, is illegal; the brace pairing has likely captured too
      // much.
      if (clean) {
        auto bad_kind = tokens[i].is_struct_brace ? BracketTokenKind::Semi
                                                  : BracketTokenKind::Comma;
        int32_t depth = 0;
        for (int32_t j = i + 1; j < close_idx; ++j) {
          if (IsOpeningBracket(tokens[j].kind)) {
            ++depth;
          } else if (IsClosingBracket(tokens[j].kind)) {
            --depth;
          } else if (tokens[j].kind == bad_kind && depth == 0) {
            clean = false;
            break;
          }
        }
      }
    } else {
      // For parens/brackets, an unmatched opener of the same kind earlier in
      // the same statement segment could really own our closer, and an
      // unmatched closer of the same kind later in the same segment could
      // really own our opener; both make the pairing suspect.
      const auto& unmatched_openers =
          kind == BracketTokenKind::OpenParen ? unmatched_open_parens
                                              : unmatched_open_squares;
      const auto& unmatched_closers =
          kind == BracketTokenKind::OpenParen ? unmatched_close_parens
                                              : unmatched_close_squares;
      if (contains_in_range(unmatched_openers, seg_first[i], i - 1)) {
        clean = false;
      }
      // Find the end of close_idx's segment: scan is bounded by the next
      // segment boundary, so just check for any unmatched closer after
      // close_idx in the same segment.
      if (clean && !unmatched_closers.empty()) {
        const auto* it = std::upper_bound(unmatched_closers.begin(),
                                          unmatched_closers.end(), close_idx);
        if (it != unmatched_closers.end() &&
            seg_id[*it] == seg_id[close_idx]) {
          clean = false;
        }
      }
    }

    // Note: an illegal leaf adjacency inside a *matched* pair is treated as
    // invalid user code, not a bracket error; the pair is still trusted and
    // collapsed. The adjacency cue only guides where to insert brackets in
    // regions that are already unbalanced.

    if (clean) {
      for (int32_t j = i + 1; j < close_idx; ++j) {
        if (match_partner[j] != -1 &&
            (match_partner[j] < i || match_partner[j] > close_idx)) {
          clean = false;
          break;
        }
        if (match_partner[j] == -1 && (IsOpeningBracket(tokens[j].kind) ||
                                       IsClosingBracket(tokens[j].kind))) {
          clean = false;
          break;
        }
        if (IsOpeningBracket(tokens[j].kind) && !is_clean_range[j]) {
          clean = false;
          break;
        }
        if (kind == BracketTokenKind::OpenCurlyBrace &&
            !tokens[i].is_struct_brace && is_first_on_line[j] &&
            tokens[j].line != tokens[close_idx].line &&
            tokens[j].line_indent <= effective_header_indent[i]) {
          clean = false;
          break;
        }
      }
    }

    if (clean) {
      is_clean_range[i] = true;
    }
  }

  // 4. Build item sequence.
  llvm::SmallVector<Item> items;
  auto make_item = [&](int32_t start, int32_t end, bool collapsed,
                       bool has_scope) {
    items.push_back(Item{
        .token_start_index = start,
        .token_end_index = end,
        .is_collapsed_block = collapsed,
        .contains_scope_brace = has_scope,
        .token = tokens[start],
        .effective_header_indent = effective_header_indent[start],
        .is_first_on_line = is_first_on_line[start],
        .follows_statement_header = follows_statement_header[start],
        .header_has_open_curly_brace = header_has_open_curly_brace[start],
        .prev_kind =
            start > 0 ? tokens[start - 1].kind : BracketTokenKind::Other,
        .prev_is_paren_keyword =
            start > 0 && tokens[start - 1].is_paren_keyword,
        .prev_is_structural_op =
            start > 0 && tokens[start - 1].is_structural_op,
        .prev_is_assignment_op =
            start > 0 && tokens[start - 1].is_assignment_op,
        .prev_has_leading_space =
            start > 0 && tokens[start - 1].has_leading_space,
    });
  };
  for (int32_t i = 0; i < num_tokens;) {
    if (is_clean_range[i] && match_partner[i] != -1) {
      int32_t close_idx = match_partner[i];
      bool has_scope = false;
      for (int32_t j = i; j <= close_idx; ++j) {
        if (tokens[j].kind == BracketTokenKind::OpenCurlyBrace &&
            !tokens[j].is_struct_brace) {
          has_scope = true;
          break;
        }
      }
      make_item(i, close_idx, /*collapsed=*/true, has_scope);
      i = close_idx + 1;
    } else {
      make_item(i, i, /*collapsed=*/false,
                tokens[i].kind == BracketTokenKind::OpenCurlyBrace ||
                    tokens[i].kind == BracketTokenKind::CloseCurlyBrace);
      ++i;
    }
  }

  // 5. Partition items into regions and solve each damaged region. A region
  // ends at a top-level declaration boundary: a statement introducer at zero
  // indentation whose predecessor ended a statement. This bounds how far a
  // single mistake can smear, and gives unclosed brackets a natural place to
  // be closed (the region end).
  llvm::SmallVector<int32_t> region_boundaries;
  region_boundaries.push_back(0);

  for (int32_t i = 0; i < static_cast<int32_t>(items.size()); ++i) {
    const auto& item = items[i];
    // Note: line_indent is a 1-based column number, so top-level tokens have
    // line_indent 1.
    if (i > 0 && !item.is_collapsed_block &&
        item.token.kind == BracketTokenKind::StatementIntroducer &&
        !item.token.is_else_keyword && item.token.line_indent <= 1 &&
        item.is_first_on_line) {
      auto prev_end_kind = tokens[items[i - 1].token_end_index].kind;
      if ((prev_end_kind == BracketTokenKind::Semi ||
           prev_end_kind == BracketTokenKind::CloseCurlyBrace) &&
          region_boundaries.back() != i) {
        region_boundaries.push_back(i);
      }
    }

  }
  if (region_boundaries.back() != static_cast<int32_t>(items.size())) {
    region_boundaries.push_back(static_cast<int32_t>(items.size()));
  }

  for (size_t b = 0; b + 1 < region_boundaries.size(); ++b) {
    int32_t start = region_boundaries[b];
    int32_t end = region_boundaries[b + 1];
    if (start >= end) {
      continue;
    }

    // A region needs solving only if its loose (non-collapsed) brackets don't
    // already form a balanced, well-nested sequence. A balanced region has no
    // unmatched bracket, so the search would simply match everything and emit
    // no corrections; skipping it avoids running the beam search over the many
    // regions whose matched pairs merely weren't collapsed.
    bool balanced = true;
    llvm::SmallVector<BracketTokenKind> open_kinds;
    for (int32_t i = start; i < end && balanced; ++i) {
      if (items[i].is_collapsed_block) {
        continue;
      }
      auto k = items[i].token.kind;
      if (IsOpeningBracket(k)) {
        open_kinds.push_back(k);
      } else if (IsClosingBracket(k)) {
        if (open_kinds.empty() || MatchingClosingKind(open_kinds.back()) != k) {
          balanced = false;
        } else {
          open_kinds.pop_back();
        }
      }
    }
    balanced = balanced && open_kinds.empty();

    if (!balanced) {
      TokenIndex region_end_token = (end < static_cast<int32_t>(items.size()))
                                        ? items[end].token.token_index
                                        : tokens.back().token_index;
      int32_t region_end_byte = (end < static_cast<int32_t>(items.size()))
                                    ? items[end].token.byte_offset
                                    : tokens.back().byte_offset;
      auto slice = llvm::ArrayRef<Item>(items).slice(start, end - start);
      SolveRegionCostBased(slice, region_end_token, region_end_byte,
                           corrections);
    }
  }

  llvm::stable_sort(
      corrections, [](const BracketCorrection& a, const BracketCorrection& b) {
        if (a.diagnostic_token_index != b.diagnostic_token_index) {
          return a.diagnostic_token_index < b.diagnostic_token_index;
        }
        return a.fix_token_index < b.fix_token_index;
      });

  return corrections;
}

}  // namespace Carbon::Lex

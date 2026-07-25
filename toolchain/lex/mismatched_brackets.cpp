// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/mismatched_brackets.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iterator>
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
// token live in the `CloserRules` table below, which is keyed by how strongly
// the context suggests the group ends here. These few are shared with the
// region-end handling, which isn't part of that table.
// Closing anything at the end of the file or region.
constexpr int32_t CostCloseAtEnd = 12;
constexpr int32_t CostCloseParenAtEnd = 22;
constexpr int32_t CostCloseStructAtEnd = 20;
// Closing a paren/square bracket before a mid-line `.` that has whitespace
// before it: member access is normally written without spaces. Priced below
// CostSpacedPeriodInParen so that closing here beats closing earlier and
// leaving the spaced `.` unexplained.
constexpr int32_t CostCloseParenBeforeSpacedPeriod = 6;
// Closing a group at a wide mid-line whitespace gap, which suggests the
// closer was deleted in the gap.
constexpr int32_t CostCloseAtWideGap = 6;

// Costs of inserting a synthetic opening bracket in front of the current
// token live in the `OpenerRules` table below, and penalties for advancing
// over a token in a context where it doesn't belong in `AdvanceRules`.

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
  // The `Cue::` bits that depend only on this item and its neighbours,
  // precomputed by `ComputeItemCues`. The rule tables test these together with
  // the few cues that depend on the search state.
  uint64_t cues = 0;
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
  if (!is_first_on_line && curr_kind != BracketTokenKind::StatementIntroducer &&
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
                      BracketDiagnosticKind diagnostic_kind, const char* origin)
    -> BracketCorrection {
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
    corrections.push_back(
        ReplaceWithError(open.token, BracketDiagnosticKind::UnmatchedOpening,
                         "Naive_UnclosedAtEnd"));
  }
}

auto HashStack(llvm::ArrayRef<OpenBracketInfo> stack) -> uint64_t {
  uint64_t h = stack.size();
  for (const auto& info : stack) {
    uint64_t k =
        (static_cast<uint64_t>(info.token_index.index) << 32) ^
        (static_cast<uint64_t>(info.insertion_token_index.index) << 8) ^
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

// The bracket-insertion rules below are expressed as data: each rule states
// the context it applies in and what that context costs, and the rules are
// tried in order so that the first (most specific) match wins. Two small
// categorical facts — a context class and the kind of the current token —
// form a bucket index, and a constexpr-built table maps each bucket to the
// bit-set of rules that can possibly apply there, so a lookup only tests the
// handful of rules relevant to the situation.

// Context classes for the closer table: the class of the innermost open
// bracket, as a bit-set so a rule can name several. `Struct` and `Scope`
// distinguish the two kinds of `{`.
namespace Top {
constexpr uint8_t Paren = 1 << 0;
constexpr uint8_t Square = 1 << 1;
constexpr uint8_t Struct = 1 << 2;
constexpr uint8_t Scope = 1 << 3;
constexpr uint8_t ParenLike = Paren | Square;
// No open bracket at all. Only the advance table, which also scores tokens at
// the top level, uses this.
constexpr uint8_t None = 1 << 4;
constexpr uint8_t Any = ParenLike | Struct | Scope | None;
}  // namespace Top

// Context classes for the opener table: which bracket would be inserted.
namespace Ins {
constexpr uint8_t Paren = 1 << 0;
constexpr uint8_t Square = 1 << 1;
constexpr uint8_t ScopeBrace = 1 << 2;
constexpr uint8_t StructBrace = 1 << 3;
constexpr uint8_t ParenLike = Paren | Square;
}  // namespace Ins

constexpr int32_t NumContextClasses = 5;

// Index of `top`'s class, matching the bit positions in `Top`.
auto TopClassOf(const OpenBracketInfo& top) -> int32_t {
  switch (top.kind) {
    case BracketTokenKind::OpenParen:
      return 0;
    case BracketTokenKind::OpenSquareBracket:
      return 1;
    default:
      return top.is_struct_brace ? 2 : 3;
  }
}

// Index of the innermost open bracket's class, or of `Top::None`.
auto TopClassOfStack(llvm::ArrayRef<OpenBracketInfo> stack) -> int32_t {
  return stack.empty() ? 4 : TopClassOf(stack.back());
}

// Index of a synthetic opener's class, matching the bit positions in `Ins`.
auto InsClassOf(BracketTokenKind kind, bool is_struct_brace) -> int32_t {
  switch (kind) {
    case BracketTokenKind::OpenParen:
      return 0;
    case BracketTokenKind::OpenSquareBracket:
      return 1;
    default:
      return is_struct_brace ? 3 : 2;
  }
}

// Contextual cues a rule can test, beyond the bucket's context class and token
// kind. All are properties of the current token, its neighbours, the innermost
// open bracket, or the relationship between them.
namespace Cue {
// The current token is the first on its line.
constexpr uint64_t FirstOnLine = uint64_t{1} << 0;
// There is whitespace (or a comment) directly before the current token.
constexpr uint64_t LeadingSpace = uint64_t{1} << 1;
// The current token is mid-line with two or more bytes of space before it.
constexpr uint64_t WideGap = uint64_t{1} << 2;
// The token before the current one is a leaf, `)`, or `]` — an adjacency
// that is illegal before a leaf.
constexpr uint64_t PrevValueEnding = uint64_t{1} << 3;
// As `PrevValueEnding`, but also counting `]` and `}`: the previous token
// ends a value, whether or not a leaf may follow it.
constexpr uint64_t PrevValueLike = uint64_t{1} << 4;
// The current token is `=`, `->`, or `as`.
constexpr uint64_t StructuralOp = uint64_t{1} << 5;
// The current token is specifically `=`.
constexpr uint64_t AssignmentOp = uint64_t{1} << 6;
// The current token is a comparison or logical operator.
constexpr uint64_t ComparisonOp = uint64_t{1} << 7;
// The current token is `else`.
constexpr uint64_t ElseKeyword = uint64_t{1} << 8;
// The current token is a `{` with struct-literal cues.
constexpr uint64_t StructBrace = uint64_t{1} << 9;
// The innermost open bracket is a call or index `(`/`[`, rather than one
// following a keyword such as `if` or `forall`.
constexpr uint64_t CallParenTop = uint64_t{1} << 10;
// The current token is a closer matching an opener further out on the stack,
// so the innermost group has to close first.
constexpr uint64_t Cascade = uint64_t{1} << 11;
// The innermost open bracket is the token directly before the current one,
// so closing here would make an empty group.
constexpr uint64_t AfterOpenTop = uint64_t{1} << 12;
// The current token is indented no further than the header of the statement
// or declaration containing the innermost open bracket.
constexpr uint64_t DedentToHeader = uint64_t{1} << 13;
// The current token is on a later line than the innermost open bracket.
constexpr uint64_t NewLineFromTop = uint64_t{1} << 14;
// The current token is a binding modifier keyword (`ref`, `unused`,
// `template`), which like a leaf can't directly follow a value.
constexpr uint64_t ModifierKeyword = uint64_t{1} << 15;
// The previous token is a keyword that must be followed by `(` (`if`,
// `while`, `for`, `match`), or by `[` (`forall`).
constexpr uint64_t PrevKeywordWantsParen = uint64_t{1} << 16;
constexpr uint64_t PrevKeywordWantsSquare = uint64_t{1} << 17;
// The previous token is a `.`.
constexpr uint64_t PrevIsPeriod = uint64_t{1} << 18;
// The previous token itself has whitespace before it.
constexpr uint64_t PrevHasLeadingSpace = uint64_t{1} << 19;
// The previous token is a `(` or `[`.
constexpr uint64_t PrevIsOpenBracket = uint64_t{1} << 20;
// The previous token is a `)`.
constexpr uint64_t PrevIsCloseParen = uint64_t{1} << 21;
// The previous *item* is a leaf. Unlike `PrevIsPeriod` and friends, which
// look at the last token of the previous item, this looks at the item itself,
// so it is false after a collapsed `(...)` block.
constexpr uint64_t PrevItemIsLeaf = uint64_t{1} << 22;
// The previous item is a name directly following `as` or `->`, so it is a
// type rather than something callable.
constexpr uint64_t PrevItemIsTypeName = uint64_t{1} << 23;
// The previous token is a `{`.
constexpr uint64_t PrevIsOpenCurly = uint64_t{1} << 24;
// The current token is specifically `as`.
constexpr uint64_t AsOp = uint64_t{1} << 25;
// This item is a collapsed well-bracketed block rather than a single token.
constexpr uint64_t CollapsedBlock = uint64_t{1} << 26;
// A collapsed block that contains a scope `{`.
constexpr uint64_t ContainsScopeBrace = uint64_t{1} << 27;
// The current token is indented no further than the line of the innermost
// open bracket itself (as opposed to that of its statement header).
constexpr uint64_t DedentToOpenerLine = uint64_t{1} << 28;
// This search path inserted a closer directly before the current token.
constexpr uint64_t CloserInserted = uint64_t{1} << 29;
// This search path inserted a closer that can be directly followed by a leaf
// (`]` or `}`), which repairs an otherwise illegal adjacency.
constexpr uint64_t CloserFixesAdjacency = uint64_t{1} << 30;
// This search path synthesized an opener directly before the current token.
constexpr uint64_t OpenerHere = uint64_t{1} << 31;
// This search path inserted some bracket directly before the current token,
// so suspicious whitespace before it is already explained.
constexpr uint64_t BracketInsertedHere = uint64_t{1} << 32;
// The current token starts the body of a statement or declaration whose
// header is complete, and whose header didn't contain a `{`.
constexpr uint64_t FollowsStatementHeader = uint64_t{1} << 33;
constexpr uint64_t HeaderHasOpenCurly = uint64_t{1} << 34;
// The previous token is a `}`, a `]`, or a `,`.
constexpr uint64_t PrevIsCloseCurly = uint64_t{1} << 35;
constexpr uint64_t PrevIsCloseSquare = uint64_t{1} << 36;
constexpr uint64_t PrevIsComma = uint64_t{1} << 37;
}  // namespace Cue

// Number of distinct `BracketTokenKind` values.
constexpr int32_t NumBracketTokenKinds =
    static_cast<int32_t>(BracketTokenKind::Other) + 1;

// Token-kind sets a rule's `kinds` field can name.
namespace Kind {
constexpr auto Bit(BracketTokenKind kind) -> uint32_t {
  return uint32_t{1} << static_cast<int32_t>(kind);
}
// Any kind at all: the default, for a rule that doesn't care.
constexpr uint32_t Any = ~uint32_t{0};
constexpr uint32_t Semi = Bit(BracketTokenKind::Semi);
constexpr uint32_t Comma = Bit(BracketTokenKind::Comma);
constexpr uint32_t Period = Bit(BracketTokenKind::Period);
constexpr uint32_t Leaf = Bit(BracketTokenKind::Leaf);
constexpr uint32_t Other = Bit(BracketTokenKind::Other);
constexpr uint32_t FileEnd = Bit(BracketTokenKind::FileEnd);
constexpr uint32_t Introducer = Bit(BracketTokenKind::StatementIntroducer);
constexpr uint32_t OpenParen = Bit(BracketTokenKind::OpenParen);
constexpr uint32_t OpenSquare = Bit(BracketTokenKind::OpenSquareBracket);
constexpr uint32_t OpenCurly = Bit(BracketTokenKind::OpenCurlyBrace);
constexpr uint32_t CloseParen = Bit(BracketTokenKind::CloseParen);
constexpr uint32_t CloseSquare = Bit(BracketTokenKind::CloseSquareBracket);
constexpr uint32_t CloseCurly = Bit(BracketTokenKind::CloseCurlyBrace);
// A `(` or `[`, opening or closing: the bracket kinds written without a
// preceding space in formatted code.
constexpr uint32_t OpenGroup = OpenParen | OpenSquare;
constexpr uint32_t CloseGroup = CloseParen | CloseSquare;
// Any opener, and everything that isn't one.
constexpr uint32_t Opener = OpenParen | OpenSquare | OpenCurly;
constexpr uint32_t NonOpener = Any & ~Opener;
// The kinds a dedent penalty can apply to: a closer is expected to dedent,
// and `FileEnd` isn't content.
constexpr uint32_t Dedentable = Any & ~(CloseGroup | CloseCurly) & ~FileEnd;
}  // namespace Kind

// A rule's `cost` when the move should not be considered at all.
constexpr int32_t DeclineCost = -1;

// A rule in a bracket-insertion table. The rule applies when the context class
// is in `ctx`, the current token's kind is in `kinds`, and all four cue
// conditions hold. Rules are tried in order and the first match wins, so
// earlier rules express stronger, more specific cues.
struct BracketRule {
  // Bit-set of context classes: `Top::` for the closer table (the class of the
  // innermost open bracket), `Ins::` for the opener table (which bracket would
  // be inserted).
  uint8_t ctx;
  // Bit-set of token kinds, from `Kind::`.
  uint32_t kinds = Kind::Any;
  // Every cue listed here must hold.
  uint64_t when = 0;
  // No cue listed here may hold.
  uint64_t unless = 0;
  // The cues listed here must not all hold together.
  uint64_t not_all = 0;
  // At least one cue listed here must hold, if any are listed.
  uint64_t any_of = 0;
  // Cost of the insertion this rule proposes, or `Decline`.
  int32_t cost;
  // Name of the rule, reported for diagnostics and evaluation.
  const char* origin = "";

  // Builders, so a rule reads as one sentence. Each returns an updated copy,
  // and `Cost` or `Decline` finishes the rule.
  constexpr auto When(uint64_t cues) const -> BracketRule {
    auto result = *this;
    result.when = cues;
    return result;
  }
  constexpr auto Unless(uint64_t cues) const -> BracketRule {
    auto result = *this;
    result.unless = cues;
    return result;
  }
  constexpr auto NotAll(uint64_t cues) const -> BracketRule {
    auto result = *this;
    result.not_all = cues;
    return result;
  }
  constexpr auto AnyOf(uint64_t cues) const -> BracketRule {
    auto result = *this;
    result.any_of = cues;
    return result;
  }
  constexpr auto Cost(int32_t cost, const char* origin) const -> BracketRule {
    auto result = *this;
    result.cost = cost;
    result.origin = origin;
    return result;
  }
  constexpr auto Decline() const -> BracketRule {
    auto result = *this;
    result.cost = DeclineCost;
    return result;
  }
};

// Starts a rule that applies to context classes `ctx` and token kinds `kinds`.
constexpr auto Rule(uint8_t ctx, uint32_t kinds = Kind::Any) -> BracketRule {
  return BracketRule{.ctx = ctx, .kinds = kinds, .cost = DeclineCost};
}

// Whether `rule`'s cue conditions hold for the cue bit-set `cues`.
constexpr auto Matches(const BracketRule& rule, uint64_t cues) -> bool {
  return (cues & rule.when) == rule.when && (cues & rule.unless) == 0 &&
         (rule.not_all == 0 || (cues & rule.not_all) != rule.not_all) &&
         (rule.any_of == 0 || (cues & rule.any_of) != 0);
}

// Where to insert a synthetic closing bracket, and what that costs. All costs
// are relative; see the cost model note above.
constexpr BracketRule CloserRules[] = {
    // `(` and `[`.
    //
    // An empty group: the opener is directly followed by a token that can't
    // start group content, so the group's closer must have been right after
    // the opener. Such a token is a `,` (`f(x(), y)` becoming `f(x(, y)`); a
    // binary connector like `as`/`->`/`==`, which needs a left operand
    // (`f() as T` becoming `f( as T`); or a spaced `.`, which as group content
    // would be written unspaced (`f(.a = 1)`).
    Rule(Top::ParenLike, Kind::Comma)
        .When(Cue::AfterOpenTop)
        .Cost(8, "Close_EmptyGroup"),
    Rule(Top::ParenLike)
        .When(Cue::AfterOpenTop)
        .AnyOf(Cue::StructuralOp | Cue::ComparisonOp)
        .Cost(8, "Close_EmptyGroup"),
    Rule(Top::ParenLike, Kind::Period)
        .When(Cue::AfterOpenTop | Cue::LeadingSpace)
        .Cost(8, "Close_EmptyGroup"),
    // A `;` can't appear inside parens or square brackets at all.
    Rule(Top::ParenLike, Kind::Semi).Cost(6, "Close_ParenBeforeSemi"),
    Rule(Top::ParenLike).When(Cue::Cascade).Cost(6, "Close_ParenCascade"),
    // A `{` starting a block means the paren should have closed: `if (c) {`,
    // `while (c) {`. A struct-literal `{...}` can legitimately sit inside a
    // *call* paren (`f({.x = 1})`), but not inside a keyword or grouping paren
    // (whose `{` — even an empty `{}` misread as a struct — is a block). This
    // is not a cue for `[`: a `]` is essentially never immediately followed by
    // `{` (only in `fn [captures] {...}`, which is unimplemented), so a `[`
    // should close at an earlier cue, or not here at all.
    Rule(Top::Paren, Kind::OpenCurly)
        .NotAll(Cue::StructBrace | Cue::CallParenTop)
        .Cost(8, "Close_ParenBeforeBrace"),
    // `=` can directly follow both `)` and `]`, but `->` and `as` only
    // plausibly follow `)`. An unspaced structural operator is not a cue:
    // formatted code spaces these operators, and an unspaced `->` is a
    // pointer member access (`p->x`).
    Rule(Top::Paren)
        .When(Cue::StructuralOp | Cue::LeadingSpace)
        .Cost(8, "Close_ParenBeforeStructuralOp"),
    Rule(Top::Square)
        .When(Cue::StructuralOp | Cue::LeadingSpace | Cue::AssignmentOp)
        .Cost(8, "Close_ParenBeforeStructuralOp"),
    // A leaf directly following a value-ending token is illegal, and a `]`
    // between them fixes the adjacency (unlike `)`, `]` can be directly
    // followed by a leaf, as in `impl forall [...] T as ...`).
    Rule(Top::Square, Kind::Leaf)
        .When(Cue::PrevValueEnding)
        .Cost(4, "Close_SquareAtLeafAdjacency"),
    // A `.` with whitespace before it mid-line suggests a closer was deleted
    // right before it: member access is written without spaces, `x.y`.
    Rule(Top::ParenLike, Kind::Period)
        .When(Cue::LeadingSpace | Cue::PrevValueLike)
        .Unless(Cue::FirstOnLine)
        .Cost(6, "Close_ParenBeforeSpacedPeriod"),
    // Similarly, a `(` or `[` with whitespace before it directly following a
    // value-ending token: calls and indexing are written without spaces.
    Rule(Top::ParenLike, Kind::OpenGroup)
        .When(Cue::LeadingSpace | Cue::PrevValueLike)
        .Unless(Cue::FirstOnLine)
        .Cost(6, "Close_ParenBeforeSpacedOpen"),
    // A comparison or logical operator is unlikely inside square brackets or
    // call/index argument lists (but common in `if (...)` etc.).
    Rule(Top::Square)
        .When(Cue::ComparisonOp)
        .Cost(8, "Close_ParenBeforeComparison"),
    Rule(Top::Paren)
        .When(Cue::ComparisonOp | Cue::CallParenTop)
        .Cost(8, "Close_ParenBeforeComparison"),
    // A `,` with whitespace before it: formatted code has no space before a
    // comma, so a closer was likely deleted in the gap.
    Rule(Top::ParenLike, Kind::Comma)
        .When(Cue::LeadingSpace | Cue::PrevValueLike)
        .Unless(Cue::FirstOnLine)
        .Cost(6, "Close_BeforeSpacedComma"),
    // Likewise a `)` or `]` with whitespace before it: formatted code has no
    // space before closers either.
    Rule(Top::ParenLike, Kind::CloseGroup)
        .When(Cue::LeadingSpace | Cue::PrevValueLike)
        .Unless(Cue::FirstOnLine)
        .Cost(6, "Close_BeforeSpacedCloser"),
    Rule(Top::ParenLike, Kind::FileEnd)
        .Cost(CostCloseAtEnd, "Close_ParenAtFileEnd"),
    // A wide whitespace gap mid-line suggests a deleted token in the gap.
    Rule(Top::ParenLike)
        .When(Cue::WideGap)
        .Unless(Cue::FirstOnLine)
        .Cost(CostCloseAtWideGap, "Close_ParenAtWideGap"),
    // A `[` group rarely spans lines except in wrapped declaration headers
    // (`impl forall [...]` etc.), where the line break follows the `]`.
    Rule(Top::Square)
        .When(Cue::FirstOnLine | Cue::NewLineFromTop)
        .Cost(10, "Close_SquareAtContinuation"),
    // A block `{` can't be content of a header/grouping `[` (only an index
    // `arr[...]` could hold a lambda block, but a `[` after a keyword can't):
    // the `]` must close before it. A last-resort bound, below the precise
    // cues above, that stops the `[` from swallowing the block. Priced above
    // the precise cues (so they win) but below closing at the region end, so
    // an unclosed group can't swallow a whole block.
    Rule(Top::Square, Kind::OpenCurly)
        .Unless(Cue::StructBrace | Cue::CallParenTop)
        .Cost(14, "Close_SquareBeforeBlock"),
    // No positive cue that a `(`/`[` closes here. Closing before a bare
    // dedent, statement introducer, or arbitrary token was never a correct
    // guess in practice (it just closes too early), so decline: the search
    // will close at a real cue, at the region end, or, failing both, replace
    // the unmatched opener with an error token.
    Rule(Top::ParenLike).Decline(),

    // Struct `{`.
    Rule(Top::Struct, Kind::Semi).Cost(6, "Close_StructBeforeSemi"),
    Rule(Top::Struct).When(Cue::Cascade).Cost(6, "Close_StructCascade"),
    // A block `{` can't be content of a struct literal/type `{...}` (a struct
    // field is `.name = value`, and a bare `{` isn't a value here): the struct
    // must close before it, as in `-> {.x: i32} { body }`.
    Rule(Top::Struct, Kind::OpenCurly)
        .Unless(Cue::StructBrace)
        .Cost(14, "Close_StructBeforeBlock"),
    Rule(Top::Struct)
        .When(Cue::WideGap)
        .Unless(Cue::FirstOnLine)
        .Cost(CostCloseAtWideGap, "Close_StructAtWideGap"),
    Rule(Top::Struct, Kind::FileEnd)
        .Cost(CostCloseAtEnd, "Close_StructAtFileEnd"),
    Rule(Top::Struct)
        .When(Cue::FirstOnLine | Cue::DedentToHeader)
        .Cost(12, "Close_StructAtDedent"),
    Rule(Top::Struct).Cost(40, "Close_StructBaseline"),

    // Scope `{`.
    Rule(Top::Scope)
        .When(Cue::FirstOnLine | Cue::DedentToHeader)
        .Cost(6, "Close_ScopeAtDedent"),
    // A first-on-line `else` normally directly follows a `}` on the same line,
    // so one was likely deleted before it. Priced below Close_ScopeAtDedent so
    // this wins over closing the else block early.
    Rule(Top::Scope)
        .When(Cue::ElseKeyword | Cue::FirstOnLine)
        .Cost(4, "Close_ScopeBeforeElse"),
    Rule(Top::Scope).When(Cue::Cascade).Cost(6, "Close_ScopeCascade"),
    Rule(Top::Scope, Kind::FileEnd)
        .Cost(CostCloseAtEnd, "Close_ScopeAtFileEnd"),
    Rule(Top::Scope).Cost(45, "Close_ScopeBaseline"),
};

// A bucket index: rules are identified by a bit in a `uint64_t`.
static_assert(std::size(CloserRules) <= 64);

// Maps each (top class, token kind) bucket to the bit-set of rules that can
// apply in it, so a lookup tests only those rules, in table order.
template <size_t N>
constexpr auto BuildRuleIndex(const BracketRule (&rules)[N])
    -> std::array<uint64_t, NumContextClasses * NumBracketTokenKinds> {
  std::array<uint64_t, NumContextClasses * NumBracketTokenKinds> index = {};
  for (size_t r = 0; r != N; ++r) {
    for (int32_t t = 0; t != NumContextClasses; ++t) {
      if ((rules[r].ctx & (1 << t)) == 0) {
        continue;
      }
      for (int32_t k = 0; k != NumBracketTokenKinds; ++k) {
        if ((rules[r].kinds & Kind::Bit(BracketTokenKind(k))) == 0) {
          continue;
        }
        index[t * NumBracketTokenKinds + k] |= uint64_t{1} << r;
      }
    }
  }
  return index;
}

constexpr auto CloserRuleIndex = BuildRuleIndex(CloserRules);

// Returns `bit` if `holds`, for building cue bit-sets.
constexpr auto CueIf(bool holds, uint64_t bit) -> uint64_t {
  return holds ? bit : 0;
}

// Computes the cues that depend only on `item` and its neighbours. Stored on
// the item, since the search revisits each item once per beam state.
auto ComputeItemCues(const Item& item, const Item* prev_item) -> uint64_t {
  const auto& token = item.token;
  return CueIf(item.is_first_on_line, Cue::FirstOnLine) |
         CueIf(token.has_leading_space, Cue::LeadingSpace) |
         CueIf(token.has_wide_leading_space, Cue::WideGap) |
         CueIf(token.prev_is_value_ending, Cue::PrevValueEnding) |
         CueIf(PrevIsValueLike(item), Cue::PrevValueLike) |
         CueIf(token.is_structural_op, Cue::StructuralOp) |
         CueIf(token.is_assignment_op, Cue::AssignmentOp) |
         CueIf(token.is_comparison_op, Cue::ComparisonOp) |
         CueIf(token.is_else_keyword, Cue::ElseKeyword) |
         CueIf(token.is_struct_brace, Cue::StructBrace) |
         CueIf(token.is_modifier_keyword, Cue::ModifierKeyword) |
         CueIf(item.prev_is_paren_keyword &&
                   item.prev_kind == BracketTokenKind::StatementIntroducer,
               Cue::PrevKeywordWantsParen) |
         CueIf(item.prev_is_paren_keyword &&
                   item.prev_kind != BracketTokenKind::StatementIntroducer,
               Cue::PrevKeywordWantsSquare) |
         CueIf(item.prev_kind == BracketTokenKind::Period, Cue::PrevIsPeriod) |
         CueIf(item.prev_has_leading_space, Cue::PrevHasLeadingSpace) |
         CueIf(item.prev_kind == BracketTokenKind::OpenParen ||
                   item.prev_kind == BracketTokenKind::OpenSquareBracket,
               Cue::PrevIsOpenBracket) |
         CueIf(item.prev_kind == BracketTokenKind::CloseParen,
               Cue::PrevIsCloseParen) |
         CueIf(item.prev_kind == BracketTokenKind::OpenCurlyBrace,
               Cue::PrevIsOpenCurly) |
         CueIf(item.prev_kind == BracketTokenKind::CloseCurlyBrace,
               Cue::PrevIsCloseCurly) |
         CueIf(item.prev_kind == BracketTokenKind::CloseSquareBracket,
               Cue::PrevIsCloseSquare) |
         CueIf(item.prev_kind == BracketTokenKind::Comma, Cue::PrevIsComma) |
         CueIf(item.follows_statement_header, Cue::FollowsStatementHeader) |
         CueIf(item.header_has_open_curly_brace, Cue::HeaderHasOpenCurly) |
         CueIf(token.is_as_op, Cue::AsOp) |
         CueIf(item.is_collapsed_block, Cue::CollapsedBlock) |
         CueIf(item.contains_scope_brace, Cue::ContainsScopeBrace) |
         CueIf(prev_item != nullptr &&
                   prev_item->token.kind == BracketTokenKind::Leaf,
               Cue::PrevItemIsLeaf) |
         CueIf(prev_item != nullptr && prev_item->prev_is_structural_op &&
                   !prev_item->prev_is_assignment_op,
               Cue::PrevItemIsTypeName);
}

// Computes the cues that depend on the innermost open bracket `top` and the
// search state, to combine with `item.cues`.
auto ComputeTopCues(const OpenBracketInfo& top, const Item& item,
                    llvm::ArrayRef<OpenBracketInfo> stack) -> uint64_t {
  const auto& token = item.token;
  return CueIf(top.is_call_paren, Cue::CallParenTop) |
         CueIf(MatchesDeeperOpener(stack, token.kind), Cue::Cascade) |
         CueIf(top.token_pos == item.token_start_index - 1, Cue::AfterOpenTop) |
         CueIf(token.line_indent <= top.effective_header_indent,
               Cue::DedentToHeader) |
         CueIf(token.line != top.line, Cue::NewLineFromTop);
}

// Computes the cost of inserting a synthetic closer for `top` directly before
// `item`, or nullopt if this insertion isn't worth exploring. Sets `origin` to
// the rule that fired.
auto ClassifyCloserInsertion(const OpenBracketInfo& top, const Item& item,
                             llvm::ArrayRef<OpenBracketInfo> stack,
                             const char*& origin) -> std::optional<int32_t> {
  uint64_t cues = item.cues | ComputeTopCues(top, item, stack);
  uint64_t candidates = CloserRuleIndex[TopClassOf(top) * NumBracketTokenKinds +
                                        static_cast<int32_t>(item.token.kind)];
  while (candidates != 0) {
    const auto& rule = CloserRules[std::countr_zero(candidates)];
    candidates &= candidates - 1;
    if (Matches(rule, cues)) {
      if (rule.cost == DeclineCost) {
        break;
      }
      origin = rule.origin;
      return rule.cost;
    }
  }
  return std::nullopt;
}

// Where to insert a synthetic opening bracket, and what that costs. A `(` or
// `[` can always be synthesized, just expensively without a cue, so those end
// in a baseline; a brace is only proposed where a cue supports it.
constexpr BracketRule OpenerRules[] = {
    // `if`/`while`/`for`/`match` (statement introducers) require a following
    // `(`; `forall` (an Other token) requires a following `[`.
    Rule(Ins::Paren, Kind::NonOpener)
        .When(Cue::PrevKeywordWantsParen)
        .Cost(3, "Open_AfterParenKeyword"),
    Rule(Ins::Square, Kind::NonOpener)
        .When(Cue::PrevKeywordWantsSquare)
        .Cost(3, "Open_AfterParenKeyword"),
    // A leaf or binding modifier directly following a value-ending token is
    // illegal; an opener here fixes the adjacency.
    Rule(Ins::ParenLike, Kind::Leaf)
        .When(Cue::PrevValueEnding)
        .Cost(3, "Open_AtLeafAdjacency"),
    Rule(Ins::ParenLike)
        .When(Cue::PrevValueEnding | Cue::ModifierKeyword)
        .Cost(3, "Open_AtLeafAdjacency"),
    // A `.` with whitespace before it directly following a value-ending
    // token: likely a designator argument that lost its `(`, as in
    // `ImplicitAs(.Self)`.
    Rule(Ins::Paren, Kind::Period)
        .When(Cue::LeadingSpace | Cue::PrevValueEnding)
        .Unless(Cue::FirstOnLine)
        .Cost(CostCloseParenBeforeSpacedPeriod, "Open_BeforeSpacedPeriod"),
    // A mid-line leaf with whitespace before it directly following an
    // unspaced `.`: member access is written without spaces, `x.y`, so a
    // bracket was likely deleted in the gap.
    Rule(Ins::ParenLike, Kind::Leaf)
        .When(Cue::LeadingSpace | Cue::PrevIsPeriod)
        .Unless(Cue::FirstOnLine | Cue::PrevHasLeadingSpace)
        .Cost(4, "Open_AfterPeriodGap"),
    // A mid-line leaf with whitespace before it directly following an
    // opener: formatted code has no space after `(` or `[`, so a bracket was
    // likely deleted in the gap.
    Rule(Ins::ParenLike, Kind::Leaf)
        .When(Cue::LeadingSpace | Cue::PrevIsOpenBracket)
        .Unless(Cue::FirstOnLine)
        .Cost(4, "Open_AfterOpenGap"),
    // A wide whitespace gap before a token that could start a group suggests
    // an opener was deleted in the gap.
    Rule(Ins::ParenLike, Kind::Opener | Kind::Leaf | Kind::Period)
        .When(Cue::WideGap)
        .Unless(Cue::FirstOnLine)
        .Cost(CostCloseAtWideGap, "Open_AtWideGap"),
    Rule(Ins::ParenLike)
        .When(Cue::WideGap | Cue::ModifierKeyword)
        .Unless(Cue::FirstOnLine)
        .Cost(CostCloseAtWideGap, "Open_AtWideGap"),
    // An empty group directly after a name: `Op()`. Only applies after a
    // leaf (a call of a just-computed value, `f(x)()`, is much rarer than a
    // call of a name), and not when the name is a type after `as` or `->`,
    // where a parenthesized group is more plausible than an empty call.
    Rule(Ins::Paren, Kind::CloseParen)
        .When(Cue::PrevItemIsLeaf | Cue::LeadingSpace)
        .Unless(Cue::PrevItemIsTypeName)
        .Cost(5, "Open_EmptyParens"),
    // An empty `[]` is rarer than empty parens.
    Rule(Ins::Square, Kind::CloseSquare)
        .When(Cue::PrevItemIsLeaf | Cue::LeadingSpace)
        .Unless(Cue::PrevItemIsTypeName)
        .Cost(12, "Open_EmptySquares"),
    // An empty call of a just-computed value: `T.(Default.Op)()`. Only
    // trusted when the `)` is spaced, marking the deletion gap. This looks at
    // the last token of the previous item, so it fires after a collapsed
    // `(...)` block too.
    Rule(Ins::Paren, Kind::CloseParen)
        .When(Cue::PrevIsCloseParen | Cue::LeadingSpace)
        .Unless(Cue::FirstOnLine)
        .Cost(8, "Open_EmptyParensAfterClose"),
    // A `(` or `[` anywhere else an expression could start.
    Rule(Ins::ParenLike).Cost(35, "Open_ParenBaseline"),
    // A scope `{` is never inserted directly before another opener: the body
    // it would open starts with that opener, so the `{` belongs before it only
    // via one of the rules above.
    Rule(Ins::ScopeBrace, Kind::Opener).Decline(),
    // A scope `{` between an unbraced declaration or statement header and its
    // body, as in `if (c) return;`.
    Rule(Ins::ScopeBrace)
        .When(Cue::FollowsStatementHeader)
        .Unless(Cue::HeaderHasOpenCurly)
        .Cost(8, "Open_ScopeAfterHeader"),
    Rule(Ins::ScopeBrace).Cost(60, "Open_ScopeBaseline"),
    // A struct `{` before a `.` designator that isn't a member access.
    Rule(Ins::StructBrace, Kind::Period)
        .Unless(Cue::PrevValueEnding)
        .Cost(5, "Open_StructBeforeDesignator"),
    // A struct literal `{...}` that lost its `{`, leaving content directly
    // before the `}`. Real content is required, so a stray `}` is reported as
    // an error instead. Priced above Open_ScopeAfterHeader: when a single-line
    // body lost its `{`, inserting it before the body beats making empty
    // braces at the `}`.
    Rule(Ins::StructBrace, Kind::CloseCurly)
        .Unless(Cue::FirstOnLine)
        .AnyOf(Cue::PrevValueEnding | Cue::PrevIsCloseCurly |
               Cue::PrevIsCloseSquare | Cue::PrevIsComma)
        .Cost(10, "Open_StructEmptyBraces"),
    // Any other struct brace has no cue at all, and isn't worth proposing.
    Rule(Ins::StructBrace).Decline(),
};

static_assert(std::size(OpenerRules) <= 64);

constexpr auto OpenerRuleIndex = BuildRuleIndex(OpenerRules);

// Computes the cost of inserting a synthetic opener directly before `item`, or
// nullopt if this insertion isn't worth exploring. Sets `origin` to the rule
// that fired.
auto ClassifyOpenerInsertion(BracketTokenKind kind, bool is_struct_brace,
                             const Item& item, const char*& origin)
    -> std::optional<int32_t> {
  uint64_t candidates =
      OpenerRuleIndex[InsClassOf(kind, is_struct_brace) * NumBracketTokenKinds +
                      static_cast<int32_t>(item.token.kind)];
  while (candidates != 0) {
    const auto& rule = OpenerRules[std::countr_zero(candidates)];
    candidates &= candidates - 1;
    if (Matches(rule, item.cues)) {
      if (rule.cost == DeclineCost) {
        break;
      }
      origin = rule.origin;
      return rule.cost;
    }
  }
  return std::nullopt;
}

// Penalties for advancing over an item in a context where it doesn't belong.
// These make "close the group before this token" win over "swallow this token
// into the group". Unlike the tables above, this one is additive: every
// matching rule contributes, and the penalties sum.
constexpr BracketRule AdvanceRules[] = {
    // A line that dedents to at-or-before the indentation of the enclosing
    // brace's statement header, or of the enclosing paren's own line, while
    // the bracket is still open. Closing brackets are excluded, since a `}`
    // that closes its group is expected to dedent, and `FileEnd` isn't
    // content at all.
    Rule(Top::Struct | Top::Scope, Kind::Dedentable)
        .When(Cue::FirstOnLine | Cue::DedentToHeader)
        .Cost(40, "Adv_DedentInScope"),
    Rule(Top::ParenLike, Kind::Dedentable)
        .When(Cue::FirstOnLine | Cue::DedentToOpenerLine)
        .Cost(25, "Adv_DedentInParen"),
    // A scope `{` inside parens or a struct brace is a lambda, which is rare;
    // a struct `{` there is a struct literal argument, which is common. The
    // first rule covers a collapsed block that contains a scope brace.
    Rule(Top::ParenLike | Top::Struct, Kind::Opener)
        .When(Cue::CollapsedBlock | Cue::ContainsScopeBrace)
        .Cost(40, "Adv_ScopeBlockInParen"),
    Rule(Top::ParenLike | Top::Struct, Kind::OpenCurly)
        .Unless(Cue::CollapsedBlock | Cue::StructBrace)
        .Cost(40, "Adv_ScopeBraceInParen"),
    Rule(Top::ParenLike | Top::Struct, Kind::OpenCurly)
        .When(Cue::StructBrace)
        .Unless(Cue::CollapsedBlock)
        .Cost(5, "Adv_StructBraceInParen"),
    // A spaced `(` or `[` directly after a value: calls and indexing are
    // written without spaces, so a bracket was likely deleted in the gap —
    // unless this path already inserted a closer there.
    Rule(Top::Any, Kind::OpenGroup)
        .When(Cue::LeadingSpace | Cue::PrevValueLike)
        .Unless(Cue::FirstOnLine | Cue::CollapsedBlock | Cue::CloserInserted)
        .Cost(10, "Adv_SpacedOpenAfterValue"),
    // Formatted code has no space before a closer either.
    Rule(Top::Any, Kind::CloseGroup)
        .When(Cue::LeadingSpace | Cue::PrevValueLike)
        .Unless(Cue::FirstOnLine | Cue::BracketInsertedHere)
        .Cost(8, "Adv_SpacedCloserUnexplained"),
    // A `;` can't appear inside parens, square brackets, or a struct brace.
    Rule(Top::ParenLike | Top::Struct, Kind::Semi).Cost(100, "Adv_SemiInParen"),
    // A `,` at statement level: directly in a scope brace or at the top level.
    Rule(Top::None | Top::Scope, Kind::Comma)
        .Cost(50, "Adv_CommaAtStatementLevel"),
    // A `,` directly following a still-open `(`/`[` is illegal.
    Rule(Top::ParenLike | Top::Struct, Kind::Comma)
        .When(Cue::AfterOpenTop)
        .Cost(50, "Adv_CommaAfterOpen"),
    // Formatted code has no space before a `,`, so a bracket was likely
    // deleted in the gap.
    Rule(Top::ParenLike | Top::Struct, Kind::Comma)
        .When(Cue::LeadingSpace | Cue::PrevValueLike)
        .Unless(Cue::FirstOnLine | Cue::CloserInserted | Cue::AfterOpenTop)
        .Cost(8, "Adv_SpacedCommaInParen"),
    // A statement introducer keyword inside parens or a struct brace.
    Rule(Top::ParenLike | Top::Struct, Kind::Introducer)
        .Cost(60, "Adv_IntroducerInParen"),
    // A leaf, or a binding modifier keyword, directly following a value-ending
    // token is an illegal adjacency — unless an opener synthesized here, or a
    // `]`/`}` inserted here, repairs it.
    Rule(Top::Any, Kind::Leaf)
        .When(Cue::PrevValueEnding)
        .Unless(Cue::OpenerHere | Cue::CloserFixesAdjacency)
        .Cost(60, "Adv_LeafAdjacency"),
    Rule(Top::Any, Kind::Other)
        .When(Cue::ModifierKeyword | Cue::PrevValueEnding)
        .Unless(Cue::OpenerHere | Cue::CloserFixesAdjacency)
        .Cost(60, "Adv_LeafAdjacency"),
    // `=`, `->`, or `as` inside parens or square brackets. These *can* occur
    // there (default arguments, function types, casts), so this is mild; it
    // serves to prefer the earliest sensible close point. Casts `(x as T)` are
    // common enough that `as` keeps only a nominal preference.
    Rule(Top::ParenLike, Kind::Other)
        .When(Cue::StructuralOp | Cue::LeadingSpace)
        .Unless(Cue::AsOp)
        .Cost(10, "Adv_StructuralOpInParen"),
    Rule(Top::ParenLike, Kind::Other)
        .When(Cue::StructuralOp | Cue::LeadingSpace | Cue::AsOp)
        .Cost(1, "Adv_AsOpInParen"),
    // A comparison or logical operator inside square brackets or a call.
    Rule(Top::Square, Kind::Other)
        .When(Cue::ComparisonOp)
        .Cost(8, "Adv_ComparisonInSquare"),
    Rule(Top::Paren, Kind::Other)
        .When(Cue::ComparisonOp | Cue::CallParenTop)
        .Cost(8, "Adv_ComparisonInSquare"),
    // A wide mid-line whitespace gap suggests a deleted bracket that this path
    // hasn't repaired.
    Rule(Top::Any, Kind::Other)
        .When(Cue::WideGap)
        .Unless(Cue::FirstOnLine | Cue::BracketInsertedHere)
        .Cost(10, "Adv_WideGapUnexplained"),
    // A mid-line `.` with whitespace before it suggests a deleted bracket:
    // member access is written without spaces. Prefer closing an open group
    // before it, or opening one.
    Rule(Top::Any, Kind::Period)
        .When(Cue::LeadingSpace)
        .Unless(Cue::FirstOnLine | Cue::BracketInsertedHere)
        .AnyOf(Cue::PrevValueLike | Cue::PrevIsOpenBracket |
               Cue::PrevIsOpenCurly)
        .Cost(10, "Adv_SpacedPeriodInParen"),
};

static_assert(std::size(AdvanceRules) <= 64);

// Declining is meaningless in an additive table: every matching rule
// contributes its cost, so a declining rule would subtract from the penalty.
// This also catches a rule that forgot its `Cost`.
static_assert([] {
  for (const BracketRule& rule : AdvanceRules) {
    if (rule.cost == DeclineCost) {
      return false;
    }
  }
  return true;
}());

constexpr auto AdvanceRuleIndex = BuildRuleIndex(AdvanceRules);

// Computes the cues for advancing over `item` in search state `node`.
auto ComputeAdvanceCues(const SearchState& node, const Item& item) -> uint64_t {
  bool opener_here = OpenerSynthesizedHere(node.stack, item);
  bool closer_here = node.closer_inserted != BracketTokenKind::Other;
  uint64_t cues = item.cues | CueIf(closer_here, Cue::CloserInserted) |
                  CueIf(CloserFixesLeafAdjacency(node.closer_inserted),
                        Cue::CloserFixesAdjacency) |
                  CueIf(opener_here, Cue::OpenerHere) |
                  CueIf(closer_here || opener_here, Cue::BracketInsertedHere);
  if (node.stack.empty()) {
    return cues;
  }
  const auto& top = node.stack.back();
  return cues | CueIf(top.is_call_paren, Cue::CallParenTop) |
         CueIf(top.token_pos == item.token_start_index - 1, Cue::AfterOpenTop) |
         CueIf(item.token.line_indent <= top.effective_header_indent,
               Cue::DedentToHeader) |
         CueIf(item.token.line_indent <= top.line_indent,
               Cue::DedentToOpenerLine);
}

// Computes the total penalty for advancing over `item` in search state `node`,
// summing every `AdvanceRules` entry that matches.
auto AdvancePenalty(const SearchState& node, const Item& item) -> int32_t {
  uint64_t cues = ComputeAdvanceCues(node, item);
  int32_t penalty = 0;
  uint64_t candidates =
      AdvanceRuleIndex[TopClassOfStack(node.stack) * NumBracketTokenKinds +
                       static_cast<int32_t>(item.token.kind)];
  while (candidates != 0) {
    const auto& rule = AdvanceRules[std::countr_zero(candidates)];
    candidates &= candidates - 1;
    if (Matches(rule, cues)) {
      penalty += rule.cost;
    }
  }
  return penalty;
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
    if (a.fix_action != b.fix_action || a.fix_token_kind != b.fix_token_kind) {
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
            if (llvm::none_of(
                    exist_node.parent_edges,
                    [&](const ParentEdge& e) { return EdgesEqual(e, edge); })) {
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
          bool spaced_suspicious = kind != BracketTokenKind::CloseCurlyBrace &&
                                   !item.is_first_on_line &&
                                   item.token.has_leading_space &&
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

        // Propose each flavor of synthetic opener the table allows here.
        auto propose_opener = [&](BracketTokenKind open_kind,
                                  bool is_struct_brace) {
          const char* origin = "";
          if (auto cost = ClassifyOpenerInsertion(open_kind, is_struct_brace,
                                                  item, origin)) {
            push_synthetic(open_kind, is_struct_brace, *cost, origin);
          }
        };
        propose_opener(BracketTokenKind::OpenParen, /*is_struct_brace=*/false);
        propose_opener(BracketTokenKind::OpenSquareBracket,
                       /*is_struct_brace=*/false);
        propose_opener(BracketTokenKind::OpenCurlyBrace,
                       /*is_struct_brace=*/false);
        propose_opener(BracketTokenKind::OpenCurlyBrace,
                       /*is_struct_brace=*/true);
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

      // What it costs to advance over this item in this state, per the
      // `AdvanceRules` table.
      int32_t penalty = AdvancePenalty(current, item);

      if (item.is_collapsed_block) {
        try_enqueue_advance(current.stack, penalty);
        continue;
      }

      if (IsOpeningBracket(kind)) {
        // Advance and push opener onto stack.
        if (current.stack.size() < MaxSearchStackDepth) {
          auto next_stack = current.stack;
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
            ReplaceWithError(item.token,
                             BracketDiagnosticKind::UnmatchedOpening,
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
          if (kind == BracketTokenKind::CloseCurlyBrace &&
              !top.is_struct_brace && item.token.line != top.line) {
            // A multi-line scope close must not be dedented past its header,
            // and pays for indentation disagreement with its header.
            if (item.token.line_indent < top.effective_header_indent) {
              allow_match = false;
            } else if (item.is_first_on_line &&
                       item.token.line_indent != top.effective_header_indent) {
              penalty += CostBraceIndentMismatchBase +
                         CostBraceIndentMismatchPerColumn *
                             std::abs(top.effective_header_indent -
                                      item.token.line_indent);
            }
          }
          if (allow_match) {
            auto next_stack = current.stack;
            auto popped = next_stack.pop_back_val();
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
            ReplaceWithError(item.token,
                             BracketDiagnosticKind::UnmatchedClosing,
                             "Adv_ReplaceCloser"),
            /*has_correction=*/true);
        continue;
      }

      // Non-bracket token.
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
                      .diagnostic_kind =
                          BracketDiagnosticKind::UnmatchedOpening,
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
                   tokens[close_idx].line_indent < effective_header_indent[i]) {
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
      const auto& unmatched_openers = kind == BracketTokenKind::OpenParen
                                          ? unmatched_open_parens
                                          : unmatched_open_squares;
      const auto& unmatched_closers = kind == BracketTokenKind::OpenParen
                                          ? unmatched_close_parens
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
        if (it != unmatched_closers.end() && seg_id[*it] == seg_id[close_idx]) {
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
  for (int32_t i = 0; i < static_cast<int32_t>(items.size()); ++i) {
    items[i].cues = ComputeItemCues(items[i], i > 0 ? &items[i - 1] : nullptr);
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

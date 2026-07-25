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
#include "common/hashing.h"
#include "common/map.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"

namespace Carbon::Lex {
namespace {

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

// Maximum number of collapsed items in a damaged region before falling back to
// naive greedy recovery. The beam search below is linear in the region size,
// so this is mostly a defense against pathological inputs.
constexpr int32_t MaxRegionItemsForSearch = 1500;

// Layered beam search width limit.
constexpr size_t MaxBeamWidth = 16;

// Maximum stack depth allowed during search before capping.
constexpr size_t MaxSearchStackDepth = 12;

// Maximum number of distinct optimal repair paths enumerated when checking
// whether the optimal repairs agree about each correction. Paths beyond this
// are not examined, so a correction they alone would dispute stays untied.
constexpr size_t MaxOptimalPaths = 100;

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
// the `Adv_SpacedPeriodInParen` penalty, so that closing here beats closing
// earlier and leaving the spaced `.` unexplained.
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

// A contextual cue a rule can test, beyond the bucket's context class and token
// kind. These are bit indexes into a `CueSet`; no cue's value is ever
// significant, so cues can be added, removed, and reordered freely.
enum class Cue : uint8_t {
  // Properties of the current token.
  //
  // The current token is the first on its line.
  FirstOnLine,
  // There is whitespace (or a comment) directly before the current token.
  LeadingSpace,
  // The current token is mid-line with two or more bytes of space before it.
  WideGap,
  // The current token is a `{` with struct-literal cues.
  StructBrace,
  // The current token is `else`.
  ElseKeyword,

  // Properties of the token directly before the current one. All are false at
  // the start of the input.
  //
  // The previous token is value-ending, an adjacency that is illegal before a
  // leaf.
  PrevValueEnding,
  // As `PrevValueEnding`, but also counting `]` and `}`: the previous token
  // ends a value, whether or not a leaf may follow it.
  PrevValueLike,
  // The previous token itself has whitespace before it.
  PrevHasLeadingSpace,
  // The previous token is a keyword that must be followed by `(` (`if`,
  // `while`, `for`, `match`), or one that must be followed by `[` (`forall`).
  PrevKeywordWantsParen,
  PrevKeywordWantsSquare,
  // The previous token is `as`, `->`, or `where`, so what follows it is a type
  // rather than something callable. (Not `=`, which is followed by a value.)
  PrevIntroducesType,
  // The previous token's kind, for the kinds any rule distinguishes. At most
  // one of these holds at a time.
  PrevIsPeriod,
  PrevIsComma,
  // `PrevIsOpenBracket` covers `(` and `[` together, which no rule separates.
  PrevIsOpenBracket,
  PrevIsOpenCurly,
  PrevIsCloseParen,
  PrevIsCloseSquare,
  PrevIsCloseCurly,

  // Properties of the item directly before the current one. Unlike the `Prev`
  // cues above, which look at the last token of that item, these look at the
  // item as a whole, so they say nothing about a collapsed `(...)` block's
  // contents.
  //
  // The previous item is a leaf.
  PrevItemIsLeaf,
  // The previous item is a name directly following `as` or `->`, so it is a
  // type rather than something callable.
  PrevItemIsTypeName,

  // Properties of the item, from the analysis of the surrounding tokens.
  //
  // This item is a collapsed well-bracketed block rather than a single token.
  CollapsedBlock,
  // This collapsed block contains a scope `{`.
  ContainsScopeBrace,
  // The current token starts the body of a statement or declaration whose
  // header is complete, and whose header didn't contain a `{`.
  FollowsStatementHeader,
  HeaderHasOpenCurly,

  // Properties of the innermost open bracket, and of its relationship to the
  // current token. All are false when no bracket is open.
  //
  // The innermost open bracket is a call or index `(`/`[`, rather than one
  // following a keyword such as `if` or `forall`.
  CallParenTop,
  // The innermost open bracket is the token directly before the current one, so
  // closing here would make an empty group.
  AfterOpenTop,
  // The current token is indented no further than the header of the statement
  // or declaration containing the innermost open bracket.
  DedentToHeader,
  // The current token is indented no further than the line of the innermost
  // open bracket itself, as opposed to that of its statement header.
  DedentToOpenerLine,
  // The current token is on a later line than the innermost open bracket.
  NewLineFromTop,
  // The current token is a closer matching an opener further out on the stack,
  // so the innermost group has to close first.
  Cascade,

  // Properties of the repair this search path has made so far.
  //
  // This path inserted a closer directly before the current token.
  CloserInserted,
  // The closer this path inserted can be directly followed by a leaf (`]` or
  // `}`), so it repairs an otherwise illegal adjacency.
  CloserFixesAdjacency,
  // This path synthesized an opener directly before the current token.
  OpenerHere,
  // This path inserted some bracket directly before the current token, so
  // suspicious whitespace before it is already explained.
  BracketInsertedHere,

  // Must stay last: it counts the cues.
  Count,
};

// A set of cues.
enum class CueSet : uint64_t {
  None = 0,
  LLVM_MARK_AS_BITMASK_ENUM(uint64_t{1}
                            << (static_cast<uint8_t>(Cue::Count) - 1))
};

// The set containing exactly `cues`.
template <std::same_as<Cue>... CueT>
constexpr auto CueSetOf(CueT... cues) -> CueSet {
  return (CueSet::None | ... |
          static_cast<CueSet>(uint64_t{1} << static_cast<int>(cues)));
}

// Convenience synonym for our token kinds.
using Kind = BracketTokenKind;

// Internal representation of an item after clean subrange collapsing.
struct Item {
  int32_t token_start_index;
  int32_t token_end_index;
  MismatchedBracketToken token;
  int32_t effective_header_indent = 0;
  // Every boolean property of this item and its neighbours, computed once by
  // `ComputeItemCues`. This is the single record of them: the rule tables match
  // against these (combined with the few cues that depend on the search state),
  // and the search itself tests them through `Has`.
  CueSet cues = CueSet::None;

  // Whether every one of `wanted` holds for this item.
  template <typename... CueT>
  constexpr auto Has(CueT... wanted) const -> bool {
    CueSet wanted_set = CueSetOf(wanted...);
    return (cues & wanted_set) == wanted_set;
  }
};

// The parts of an unclosed opening bracket that make a search state distinct.
// Compared and hashed as a whole, so everything here must matter to the search:
// two stacks that agree on these are interchangeable.
//
// Fields are ordered so the type has no padding, which
// `has_unique_object_representations` requires and `Carbon::Map` relies on to
// hash and compare every byte.
struct OpenBracketKey {
  // The real opener token, or None for a synthetic opener.
  TokenIndex token_index = TokenIndex::None;
  // For a synthetic opener, where it would be inserted.
  TokenIndex insertion_token_index = TokenIndex::None;
  // Index of the real opener in the input token array, or -1 if synthetic.
  int32_t token_pos = -1;
  int32_t line = -1;
  // Indentation of the line containing the opener.
  int32_t line_indent = 0;
  // Indentation of the start of the statement or declaration containing the
  // opener.
  int32_t effective_header_indent = 0;
  BracketTokenKind kind;
  bool is_synthetic = false;
  bool is_struct_brace = false;
  // Whether this is a paren/square bracket directly following a value-ending
  // token (a call or index), rather than following a keyword like `if`. This is
  // a function of the opener token, so comparing it is redundant with
  // `token_index` for a real opener, and it is always false for a synthetic
  // one; it's included only because excluding it would cost more than it saves.
  bool is_call_paren = false;

  friend auto operator==(const OpenBracketKey& lhs, const OpenBracketKey& rhs)
      -> bool = default;
};

static_assert(std::has_unique_object_representations_v<OpenBracketKey>,
              "Padding would leave indeterminate bytes for hashing.");

// An unclosed opening bracket on the search stack, plus the bookkeeping that
// says nothing about which state this is.
struct OpenBracketInfo : OpenBracketKey {
  // For a synthetic opener, the rule that proposed it, for diagnostics. Two
  // openers that differ only in which rule proposed them are the same state, so
  // this is deliberately outside the key.
  const char* origin = "";
};

// Computes the associated line indentation for a token by scanning backwards,
// skipping matched parens/brackets, looking for a statement introducer.
auto GetOuterStatementIntroducerIndent(
    llvm::ArrayRef<MismatchedBracketToken> tokens,
    llvm::ArrayRef<int32_t> match_partner, int32_t j) -> int32_t {
  int32_t result_indent = tokens[j].line_indent;
  while (j > 0) {
    int32_t p = j - 1;
    if ((tokens[p].kind == Kind::CloseParen ||
         tokens[p].kind == Kind::CloseSquareBracket) &&
        match_partner[p] != -1 && match_partner[p] < p) {
      p = match_partner[p];
      if (p <= 0) {
        break;
      }
      --p;
    }
    if (tokens[p].kind == Kind::StatementIntroducer) {
      result_indent = tokens[p].line_indent;
      // Keep walking out only while the introducers stack up on one line or
      // dedent; an indented token belongs to this introducer's body.
      if (tokens[j].line == tokens[p].line ||
          tokens[j].line_indent <= tokens[p].line_indent) {
        j = p;
        continue;
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

// Computes the indentation that a token's statement or declaration starts at,
// by scanning back over the current statement to its introducer. This is what
// a `{` opened by that statement should line its `}` up with, which is often
// not the indentation of the `{` itself.
auto ComputeAssociatedLineIndent(llvm::ArrayRef<MismatchedBracketToken> tokens,
                                 llvm::ArrayRef<int32_t> match_partner,
                                 int32_t token_index) -> int32_t {
  if (token_index < 0 || token_index >= static_cast<int32_t>(tokens.size())) {
    return 0;
  }
  if (tokens[token_index].kind == Kind::FileEnd) {
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

    if (kind == Kind::Semi || kind == Kind::OpenCurlyBrace ||
        kind == Kind::CloseCurlyBrace) {
      break;
    }

    earliest_indent = tokens[j].line_indent;

    if (kind == Kind::StatementIntroducer) {
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
  if (curr_kind == Kind::Semi || curr_kind == Kind::OpenCurlyBrace ||
      IsClosingBracket(curr_kind) || curr_kind == Kind::FileEnd) {
    return false;
  }
  // Only tokens that could start a body are considered: the first token on a
  // line, a statement introducer (`if (c) return;`), or a token directly
  // after a `)` ending a header. (Not after `]`: declaration headers always
  // continue after implicit parameter lists and `forall` clauses.)
  bool is_first_on_line =
      tokens[token_index].line != tokens[token_index - 1].line;
  auto prev_kind = tokens[token_index - 1].kind;
  if (!is_first_on_line && curr_kind != Kind::StatementIntroducer &&
      prev_kind != Kind::CloseParen) {
    return false;
  }
  // A body can't start with an operator like `as` or `==`, or with a `.`
  // designator (a `where`-clause continuation line).
  if (IsStructuralOpKind(curr_kind) || curr_kind == Kind::ComparisonOp ||
      curr_kind == Kind::Period) {
    return false;
  }
  for (int32_t j = token_index - 1; j >= 0; --j) {
    auto kind = tokens[j].kind;
    if ((kind == Kind::CloseParen || kind == Kind::CloseSquareBracket) &&
        match_partner[j] != -1 && match_partner[j] < j) {
      j = match_partner[j];
      continue;
    }
    if (kind == Kind::Semi || kind == Kind::OpenCurlyBrace ||
        kind == Kind::CloseCurlyBrace || kind == Kind::OpenParen ||
        kind == Kind::OpenSquareBracket) {
      return false;
    }
    if (kind == Kind::StatementIntroducer) {
      if (curr_kind == Kind::StatementIntroducer) {
        if (tokens[token_index].line == tokens[j].line) {
          // On the same line, a chain of adjacent introducers (`private fn`)
          // is a single header; but with other tokens in between, this is a
          // body after a single-line header (`if (c) return x;`).
          bool all_introducers = true;
          for (int32_t k = j + 1; k < token_index; ++k) {
            if (tokens[k].kind != Kind::StatementIntroducer) {
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
    if (kind == Kind::OpenCurlyBrace) {
      return true;
    }
    if ((kind == Kind::OpenParen || kind == Kind::OpenSquareBracket) &&
        match_partner[j] != -1 && match_partner[j] > j) {
      j = match_partner[j];
      continue;
    }
    if (kind == Kind::Semi || kind == Kind::CloseCurlyBrace ||
        kind == Kind::FileEnd || kind == Kind::StatementIntroducer) {
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
  Kind closer_inserted = Kind::Other;
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
  Kind closer_inserted;
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
      .origin = origin,
  };
}

// Solve a damaged region using the simple greedy fallback algorithm.
auto SolveNaive(llvm::ArrayRef<Item> items,
                llvm::SmallVectorImpl<BracketCorrection>& corrections) -> void {
  llvm::SmallVector<MismatchedBracketToken> open_stack;
  for (const auto& item : items) {
    if (item.Has(Cue::CollapsedBlock)) {
      continue;
    }
    auto kind = item.token.kind;
    if (kind == Kind::Semi || kind == Kind::StatementIntroducer ||
        kind == Kind::OpenCurlyBrace || kind == Kind::FileEnd) {
      while (!open_stack.empty() &&
             (open_stack.back().kind == Kind::OpenParen ||
              open_stack.back().kind == Kind::OpenSquareBracket)) {
        corrections.push_back(ReplaceWithError(
            open_stack.pop_back_val(), BracketDiagnosticKind::UnmatchedOpening,
            "Naive_UnclosedParenBracket"));
      }
    }

    if (IsOpeningBracket(kind)) {
      open_stack.push_back(item.token);
    } else if (IsClosingBracket(kind)) {
      auto search_range = llvm::reverse(open_stack);
      size_t lookback = 0;
      auto match_it = search_range.end();
      for (auto it = search_range.begin();
           it != search_range.end() && lookback < 16; ++it, ++lookback) {
        if (MatchingClosingKind(it->kind) == kind) {
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
          corrections.push_back(
              ReplaceWithError(*it, BracketDiagnosticKind::UnmatchedOpening,
                               "Naive_PoppedOpener"));
        }
        open_stack.erase(match_it.base() - 1, open_stack.end());
      }
    }
  }

  for (const auto& open : llvm::reverse(open_stack)) {
    corrections.push_back(ReplaceWithError(
        open, BracketDiagnosticKind::UnmatchedOpening, "Naive_UnclosedAtEnd"));
  }
}

// Hashes the whole of each entry's key, so that equal stacks always hash equal:
// the search relies on that to find a state to merge into.
auto HashStack(llvm::ArrayRef<OpenBracketInfo> stack) -> uint64_t {
  auto hash = static_cast<uint64_t>(stack.size());
  for (const OpenBracketKey& key : stack) {
    hash = static_cast<uint64_t>(HashValue(key, hash));
  }
  return hash;
}

// A search state is identified by its open-bracket stack plus which closer (if
// any) was just inserted before the current token; this hash keys the
// per-layer dedup map.
auto StateHash(llvm::ArrayRef<OpenBracketInfo> stack, Kind closer_inserted)
    -> uint64_t {
  return static_cast<uint64_t>(HashValue(closer_inserted, HashStack(stack)));
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
                         Kind closing_kind) -> bool {
  if (!IsClosingBracket(closing_kind) || stack.size() < 2) {
    return false;
  }
  auto req = MatchingOpeningKind(closing_kind);
  for (const OpenBracketKey& entry : stack.drop_back()) {
    if (entry.kind == req) {
      return true;
    }
  }
  return false;
}

// Whether the token before the current one ends a value. Unlike
// `IsValueEndingKind`, which the leaf-adjacency rules use, this also counts `]`
// and `}`, which end a value but can be followed by a leaf. `prev_token` is
// null at the start of the input.
auto PrevIsValueLike(const MismatchedBracketToken* prev_token) -> bool {
  return prev_token != nullptr &&
         (IsValueEndingKind(prev_token->kind) ||
          prev_token->kind == Kind::CloseSquareBracket ||
          prev_token->kind == Kind::CloseCurlyBrace);
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
auto CloserFixesLeafAdjacency(Kind closer_inserted) -> bool {
  return closer_inserted == Kind::CloseSquareBracket ||
         closer_inserted == Kind::CloseCurlyBrace;
}

// The bracket-insertion rules below are expressed as data: each rule states
// the context it applies in and what that context costs, and the rules are
// tried in order so that the first (most specific) match wins. Two small
// categorical facts — a context class and the kind of the current token —
// form a bucket index, and a constexpr-built table maps each bucket to the
// bit-set of rules that can possibly apply there, so a lookup only tests the
// handful of rules relevant to the situation.

// Context classes for the closer and advance tables: the class of the innermost
// open bracket. `Struct` and `Scope` distinguish the two kinds of `{`. Each
// class has an index, which selects its row of bucket entries, and a bit, which
// is how a rule names it — so a rule can apply to several classes at once.
namespace Top {
enum Index : int32_t {
  ParenIndex,
  SquareIndex,
  StructIndex,
  ScopeIndex,
  // No open bracket at all. Only the advance table, which also scores tokens
  // at the top level, uses this.
  NoneIndex,
  Count,
};
constexpr uint8_t Paren = 1 << ParenIndex;
constexpr uint8_t Square = 1 << SquareIndex;
constexpr uint8_t Struct = 1 << StructIndex;
constexpr uint8_t Scope = 1 << ScopeIndex;
constexpr uint8_t None = 1 << NoneIndex;
constexpr uint8_t ParenLike = Paren | Square;
constexpr uint8_t Any = ParenLike | Struct | Scope | None;
}  // namespace Top

// Context classes for the opener table: which bracket would be inserted.
namespace Ins {
enum Index : int32_t {
  ParenIndex,
  SquareIndex,
  ScopeBraceIndex,
  StructBraceIndex,
  Count,
};
constexpr uint8_t Paren = 1 << ParenIndex;
constexpr uint8_t Square = 1 << SquareIndex;
constexpr uint8_t ScopeBrace = 1 << ScopeBraceIndex;
constexpr uint8_t StructBrace = 1 << StructBraceIndex;
constexpr uint8_t ParenLike = Paren | Square;
}  // namespace Ins

// Bucket rows are shared by all the tables, so there must be room for whichever
// set of context classes is larger.
constexpr int32_t NumContextClasses = std::max<int32_t>(Top::Count, Ins::Count);

// The class of the innermost open bracket, which is always a real bracket.
auto TopClassOf(const OpenBracketInfo& top) -> int32_t {
  switch (top.kind) {
    case Kind::OpenParen:
      return Top::ParenIndex;
    case Kind::OpenSquareBracket:
      return Top::SquareIndex;
    default:
      return top.is_struct_brace ? Top::StructIndex : Top::ScopeIndex;
  }
}

// As `TopClassOf`, but for a stack that may be empty.
auto TopClassOfStack(llvm::ArrayRef<OpenBracketInfo> stack) -> int32_t {
  return stack.empty() ? Top::NoneIndex : TopClassOf(stack.back());
}

// The class of a synthetic opener of kind `kind`.
auto InsClassOf(Kind kind, bool is_struct_brace) -> int32_t {
  switch (kind) {
    case Kind::OpenParen:
      return Ins::ParenIndex;
    case Kind::OpenSquareBracket:
      return Ins::SquareIndex;
    default:
      return is_struct_brace ? Ins::StructBraceIndex : Ins::ScopeBraceIndex;
  }
}

// Number of distinct `Kind` values.
constexpr int32_t NumKinds = static_cast<int32_t>(Kind::Other) + 1;

// A set of token kinds a rule can apply to. `Kind` values are the
// bit indexes, so there are no per-kind enumerators here to keep in sync with
// them; `Rule` takes the kinds themselves.
enum class KindSet : uint32_t {
  None = 0,
  LLVM_MARK_AS_BITMASK_ENUM(uint32_t{1} << (NumKinds - 1))
};

// The set containing exactly `kinds`.
template <std::same_as<Kind>... KindT>
constexpr auto KindSetOf(KindT... kinds) -> KindSet {
  return (KindSet::None | ... |
          static_cast<KindSet>(uint32_t{1} << static_cast<int>(kinds)));
}

// Sets of kinds that name a concept more than one rule shares. A rule wanting
// just one or two particular kinds names them directly instead.
namespace Kinds {

// Every kind: the default, for a rule that doesn't care.
constexpr KindSet Any = ~KindSet::None;

// A `(` or `[`, opening or closing: the bracket kinds written without a
// preceding space in formatted code.
constexpr KindSet OpenGroup =
    KindSetOf(Kind::OpenParen, Kind::OpenSquareBracket);
constexpr KindSet CloseGroup =
    KindSetOf(Kind::CloseParen, Kind::CloseSquareBracket);

// Any opening bracket, and everything that isn't one.
constexpr KindSet Opener = OpenGroup | KindSetOf(Kind::OpenCurlyBrace);
constexpr KindSet NonOpener = ~Opener;

// The statement-structuring operators together.
constexpr KindSet AnyStructural =
    KindSetOf(Kind::Assignment, Kind::As, Kind::StructuralOp);

// A binary connector, which needs a left operand and so can't start a group.
constexpr KindSet BinaryConnector =
    AnyStructural | KindSetOf(Kind::ComparisonOp);

// Everything that carries no bracket structure of its own: `Other` plus the
// operator and keyword classifications split out of it.
constexpr KindSet AnyOther =
    AnyStructural |
    KindSetOf(Kind::Other, Kind::ComparisonOp, Kind::ModifierKeyword);

// A token that could start a bracketed group.
constexpr KindSet GroupStarter =
    Opener | KindSetOf(Kind::Leaf, Kind::Period, Kind::ModifierKeyword);

// A leaf, or a binding modifier keyword, which like a leaf can't directly
// follow a value-ending token.
constexpr KindSet LeafLike = KindSetOf(Kind::Leaf, Kind::ModifierKeyword);

// The kinds a dedent penalty can apply to: a closer is expected to dedent, and
// `FileEnd` isn't content.
constexpr KindSet Dedentable =
    ~(CloseGroup | KindSetOf(Kind::CloseCurlyBrace, Kind::FileEnd));

}  // namespace Kinds

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
  // The token kinds this rule applies to.
  KindSet kinds = Kinds::Any;
  // Every cue listed here must hold.
  CueSet when = CueSet::None;
  // No cue listed here may hold.
  CueSet unless = CueSet::None;
  // The cues listed here must not all hold together.
  CueSet not_all = CueSet::None;
  // At least one cue listed here must hold, if any are listed.
  CueSet any_of = CueSet::None;
  // Cost of the insertion this rule proposes, or `DeclineCost`.
  int32_t cost;
  // Name of the rule, reported for diagnostics and evaluation.
  const char* origin = "";

  // Builders, so a rule reads as one sentence. Each returns an updated copy,
  // and `Cost` or `Decline` finishes the rule.
  template <typename... CueT>
  constexpr auto When(CueT... cues) const -> BracketRule {
    auto result = *this;
    result.when = CueSetOf(cues...);
    return result;
  }
  template <typename... CueT>
  constexpr auto Unless(CueT... cues) const -> BracketRule {
    auto result = *this;
    result.unless = CueSetOf(cues...);
    return result;
  }
  template <typename... CueT>
  constexpr auto NotAll(CueT... cues) const -> BracketRule {
    auto result = *this;
    result.not_all = CueSetOf(cues...);
    return result;
  }
  template <typename... CueT>
  constexpr auto AnyOf(CueT... cues) const -> BracketRule {
    auto result = *this;
    result.any_of = CueSetOf(cues...);
    return result;
  }
  constexpr auto Cost(int32_t rule_cost, const char* rule_origin) const
      -> BracketRule {
    auto result = *this;
    result.cost = rule_cost;
    result.origin = rule_origin;
    return result;
  }
  constexpr auto Decline() const -> BracketRule {
    auto result = *this;
    result.cost = DeclineCost;
    return result;
  }
};

// Starts a rule that applies to context classes `ctx` and token kinds `kinds`.
constexpr auto Rule(uint8_t ctx, KindSet kinds = Kinds::Any) -> BracketRule {
  return BracketRule{.ctx = ctx, .kinds = kinds, .cost = DeclineCost};
}

// As above, for a rule that applies to a few particular kinds. Taking the first
// kind separately keeps the no-kinds call unambiguous.
template <std::same_as<Kind>... RestT>
constexpr auto Rule(uint8_t ctx, Kind kind, RestT... rest) -> BracketRule {
  return Rule(ctx, KindSetOf(kind, rest...));
}

// Whether `rule`'s cue conditions hold for the cue bit-set `cues`.
constexpr auto Matches(const BracketRule& rule, CueSet cues) -> bool {
  return (cues & rule.when) == rule.when &&
         (cues & rule.unless) == CueSet::None &&
         (rule.not_all == CueSet::None ||
          (cues & rule.not_all) != rule.not_all) &&
         (rule.any_of == CueSet::None || (cues & rule.any_of) != CueSet::None);
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
    Rule(Top::ParenLike, Kinds::BinaryConnector)
        .When(Cue::AfterOpenTop)
        .Cost(8, "Close_EmptyGroup"),
    Rule(Top::ParenLike, Kind::Period)
        .When(Cue::AfterOpenTop, Cue::LeadingSpace)
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
    Rule(Top::Paren, Kind::OpenCurlyBrace)
        .NotAll(Cue::StructBrace, Cue::CallParenTop)
        .Cost(8, "Close_ParenBeforeBrace"),
    // `=` can directly follow both `)` and `]`, but `->` and `as` only
    // plausibly follow `)`. An unspaced structural operator is not a cue:
    // formatted code spaces these operators, and an unspaced `->` is a
    // pointer member access (`p->x`).
    Rule(Top::Paren, Kinds::AnyStructural)
        .When(Cue::LeadingSpace)
        .Cost(8, "Close_ParenBeforeStructuralOp"),
    Rule(Top::Square, Kind::Assignment)
        .When(Cue::LeadingSpace)
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
        .When(Cue::LeadingSpace, Cue::PrevValueLike)
        .Unless(Cue::FirstOnLine)
        .Cost(6, "Close_ParenBeforeSpacedPeriod"),
    // Similarly, a `(` or `[` with whitespace before it directly following a
    // value-ending token: calls and indexing are written without spaces.
    Rule(Top::ParenLike, Kinds::OpenGroup)
        .When(Cue::LeadingSpace, Cue::PrevValueLike)
        .Unless(Cue::FirstOnLine)
        .Cost(6, "Close_ParenBeforeSpacedOpen"),
    // A comparison or logical operator is unlikely inside square brackets or
    // call/index argument lists (but common in `if (...)` etc.).
    Rule(Top::Square, Kind::ComparisonOp)
        .Cost(8, "Close_ParenBeforeComparison"),
    Rule(Top::Paren, Kind::ComparisonOp)
        .When(Cue::CallParenTop)
        .Cost(8, "Close_ParenBeforeComparison"),
    // A `,` with whitespace before it: formatted code has no space before a
    // comma, so a closer was likely deleted in the gap.
    Rule(Top::ParenLike, Kind::Comma)
        .When(Cue::LeadingSpace, Cue::PrevValueLike)
        .Unless(Cue::FirstOnLine)
        .Cost(6, "Close_BeforeSpacedComma"),
    // Likewise a `)` or `]` with whitespace before it: formatted code has no
    // space before closers either.
    Rule(Top::ParenLike, Kinds::CloseGroup)
        .When(Cue::LeadingSpace, Cue::PrevValueLike)
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
        .When(Cue::FirstOnLine, Cue::NewLineFromTop)
        .Cost(10, "Close_SquareAtContinuation"),
    // A block `{` can't be content of a header/grouping `[` (only an index
    // `arr[...]` could hold a lambda block, but a `[` after a keyword can't):
    // the `]` must close before it. A last-resort bound, below the precise
    // cues above, that stops the `[` from swallowing the block. Priced above
    // the precise cues (so they win) but below closing at the region end, so
    // an unclosed group can't swallow a whole block.
    Rule(Top::Square, Kind::OpenCurlyBrace)
        .Unless(Cue::StructBrace, Cue::CallParenTop)
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
    Rule(Top::Struct, Kind::OpenCurlyBrace)
        .Unless(Cue::StructBrace)
        .Cost(14, "Close_StructBeforeBlock"),
    Rule(Top::Struct)
        .When(Cue::WideGap)
        .Unless(Cue::FirstOnLine)
        .Cost(CostCloseAtWideGap, "Close_StructAtWideGap"),
    Rule(Top::Struct, Kind::FileEnd)
        .Cost(CostCloseAtEnd, "Close_StructAtFileEnd"),
    Rule(Top::Struct)
        .When(Cue::FirstOnLine, Cue::DedentToHeader)
        .Cost(12, "Close_StructAtDedent"),
    Rule(Top::Struct).Cost(40, "Close_StructBaseline"),

    // Scope `{`.
    Rule(Top::Scope)
        .When(Cue::FirstOnLine, Cue::DedentToHeader)
        .Cost(6, "Close_ScopeAtDedent"),
    // A first-on-line `else` normally directly follows a `}` on the same line,
    // so one was likely deleted before it. Priced below Close_ScopeAtDedent so
    // this wins over closing the else block early.
    Rule(Top::Scope)
        .When(Cue::ElseKeyword, Cue::FirstOnLine)
        .Cost(4, "Close_ScopeBeforeElse"),
    Rule(Top::Scope).When(Cue::Cascade).Cost(6, "Close_ScopeCascade"),
    Rule(Top::Scope, Kind::FileEnd)
        .Cost(CostCloseAtEnd, "Close_ScopeAtFileEnd"),
    Rule(Top::Scope).Cost(45, "Close_ScopeBaseline"),
};

// A bucket index: rules are identified by a bit in a `uint64_t`.
static_assert(std::size(CloserRules) <= 64);

// Maps each (context class, token kind) bucket to the bit-set of rules that can
// apply in it, so a lookup tests only those rules, in table order.
using RuleIndex = std::array<std::array<uint64_t, NumKinds>, NumContextClasses>;

template <size_t N>
constexpr auto BuildRuleIndex(const BracketRule (&rules)[N]) -> RuleIndex {
  RuleIndex index = {};
  for (size_t r = 0; r != N; ++r) {
    for (int32_t t = 0; t != NumContextClasses; ++t) {
      if ((rules[r].ctx & (1 << t)) == 0) {
        continue;
      }
      for (int32_t k = 0; k != NumKinds; ++k) {
        if ((rules[r].kinds & KindSetOf(static_cast<Kind>(k))) ==
            KindSet::None) {
          continue;
        }
        index[t][k] |= uint64_t{1} << r;
      }
    }
  }
  return index;
}

// The bit-set of rules that could apply in a bucket. A rule is identified by
// its bit position, which is its position in the table, so scanning the bits
// from lowest to highest visits the rules in table order.
constexpr auto CandidateRules(const RuleIndex& index, int32_t ctx_class,
                              Kind kind) -> uint64_t {
  return index[ctx_class][static_cast<int32_t>(kind)];
}

// Returns the first rule that applies, for a first-match table, or null if
// none does.
template <size_t N>
auto FindMatchingRule(const BracketRule (&rules)[N], const RuleIndex& index,
                      int32_t ctx_class, Kind kind, CueSet cues)
    -> const BracketRule* {
  uint64_t candidates = CandidateRules(index, ctx_class, kind);
  while (candidates != 0) {
    const auto& rule = rules[std::countr_zero(candidates)];
    candidates &= candidates - 1;
    if (Matches(rule, cues)) {
      return &rule;
    }
  }
  return nullptr;
}

// Returns the total cost of every rule that applies, for an additive table.
template <size_t N>
auto SumMatchingRules(const BracketRule (&rules)[N], const RuleIndex& index,
                      int32_t ctx_class, Kind kind, CueSet cues) -> int32_t {
  int32_t total = 0;
  uint64_t candidates = CandidateRules(index, ctx_class, kind);
  while (candidates != 0) {
    const auto& rule = rules[std::countr_zero(candidates)];
    candidates &= candidates - 1;
    if (Matches(rule, cues)) {
      total += rule.cost;
    }
  }
  return total;
}

constexpr auto CloserRuleIndex = BuildRuleIndex(CloserRules);

// The cue for the previous token having kind `kind`, if any.
constexpr auto PrevKindCue(Kind kind) -> CueSet {
  switch (kind) {
    case Kind::Period:
      return CueSetOf(Cue::PrevIsPeriod);
    case Kind::OpenParen:
    case Kind::OpenSquareBracket:
      return CueSetOf(Cue::PrevIsOpenBracket);
    case Kind::OpenCurlyBrace:
      return CueSetOf(Cue::PrevIsOpenCurly);
    case Kind::CloseParen:
      return CueSetOf(Cue::PrevIsCloseParen);
    case Kind::CloseCurlyBrace:
      return CueSetOf(Cue::PrevIsCloseCurly);
    case Kind::CloseSquareBracket:
      return CueSetOf(Cue::PrevIsCloseSquare);
    case Kind::Comma:
      return CueSetOf(Cue::PrevIsComma);
    default:
      return CueSet::None;
  }
}

// Returns the set holding `cue` if `holds`, and the empty set otherwise.
constexpr auto CueIf(bool holds, Cue cue) -> CueSet {
  return holds ? CueSetOf(cue) : CueSet::None;
}

// Computes every cue that depends only on an item and its neighbours.
// `prev_token` is the token directly before the item and `prev_item` the item
// directly before it, both null at the start of the input. `context_cues` holds
// the cues the caller has already determined from the surrounding token arrays
// (`Cue::CollapsedBlock`, `Cue::FirstOnLine`, and so on).
auto ComputeItemCues(const MismatchedBracketToken& token,
                     const MismatchedBracketToken* prev_token,
                     const Item* prev_item, CueSet context_cues) -> CueSet {
  CueSet cues = context_cues |
                CueIf(token.has_leading_space, Cue::LeadingSpace) |
                CueIf(token.has_wide_leading_space, Cue::WideGap) |
                CueIf(token.is_else_keyword, Cue::ElseKeyword) |
                CueIf(token.is_struct_brace, Cue::StructBrace) |
                CueIf(PrevIsValueLike(prev_token), Cue::PrevValueLike);
  if (prev_token != nullptr) {
    // `forall` requires a following `[`; the other paren keywords (`if`,
    // `while`, `for`, `match`) are statement introducers and require a `(`.
    bool wants_paren = prev_token->kind == Kind::StatementIntroducer;
    cues |= CueIf(prev_token->is_paren_keyword && wants_paren,
                  Cue::PrevKeywordWantsParen) |
            CueIf(prev_token->is_paren_keyword && !wants_paren,
                  Cue::PrevKeywordWantsSquare) |
            CueIf(prev_token->has_leading_space, Cue::PrevHasLeadingSpace) |
            CueIf(IsValueEndingKind(prev_token->kind), Cue::PrevValueEnding) |
            CueIf(prev_token->kind == Kind::As ||
                      prev_token->kind == Kind::StructuralOp,
                  Cue::PrevIntroducesType) |
            PrevKindCue(prev_token->kind);
  }
  if (prev_item != nullptr) {
    // A name directly following `as` or `->` is a type, not something
    // callable, so empty parens after it are implausible.
    cues |=
        CueIf(prev_item->token.kind == Kind::Leaf, Cue::PrevItemIsLeaf) |
        CueIf(prev_item->Has(Cue::PrevIntroducesType), Cue::PrevItemIsTypeName);
  }
  return cues;
}

// Computes the cues that depend on the innermost open bracket `top` and the
// search state, to combine with `item.cues`.
auto ComputeTopCues(const OpenBracketInfo& top, const Item& item,
                    llvm::ArrayRef<OpenBracketInfo> stack) -> CueSet {
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
  CueSet cues = item.cues | ComputeTopCues(top, item, stack);
  const auto* rule = FindMatchingRule(CloserRules, CloserRuleIndex,
                                      TopClassOf(top), item.token.kind, cues);
  if (rule == nullptr || rule->cost == DeclineCost) {
    return std::nullopt;
  }
  origin = rule->origin;
  return rule->cost;
}

// Where to insert a synthetic opening bracket, and what that costs. A `(` or
// `[` can always be synthesized, just expensively without a cue, so those end
// in a baseline; a brace is only proposed where a cue supports it.
constexpr BracketRule OpenerRules[] = {
    // `if`/`while`/`for`/`match` (statement introducers) require a following
    // `(`; `forall` (an Other token) requires a following `[`.
    Rule(Ins::Paren, Kinds::NonOpener)
        .When(Cue::PrevKeywordWantsParen)
        .Cost(3, "Open_AfterParenKeyword"),
    Rule(Ins::Square, Kinds::NonOpener)
        .When(Cue::PrevKeywordWantsSquare)
        .Cost(3, "Open_AfterParenKeyword"),
    // A leaf or binding modifier directly following a value-ending token is
    // illegal; an opener here fixes the adjacency.
    Rule(Ins::ParenLike, Kinds::LeafLike)
        .When(Cue::PrevValueEnding)
        .Cost(3, "Open_AtLeafAdjacency"),
    // A `.` with whitespace before it directly following a value-ending
    // token: likely a designator argument that lost its `(`, as in
    // `ImplicitAs(.Self)`.
    Rule(Ins::Paren, Kind::Period)
        .When(Cue::LeadingSpace, Cue::PrevValueEnding)
        .Unless(Cue::FirstOnLine)
        .Cost(CostCloseParenBeforeSpacedPeriod, "Open_BeforeSpacedPeriod"),
    // A mid-line leaf with whitespace before it directly following an
    // unspaced `.`: member access is written without spaces, `x.y`, so a
    // bracket was likely deleted in the gap.
    Rule(Ins::ParenLike, Kind::Leaf)
        .When(Cue::LeadingSpace, Cue::PrevIsPeriod)
        .Unless(Cue::FirstOnLine, Cue::PrevHasLeadingSpace)
        .Cost(4, "Open_AfterPeriodGap"),
    // A mid-line leaf with whitespace before it directly following an
    // opener: formatted code has no space after `(` or `[`, so a bracket was
    // likely deleted in the gap.
    Rule(Ins::ParenLike, Kind::Leaf)
        .When(Cue::LeadingSpace, Cue::PrevIsOpenBracket)
        .Unless(Cue::FirstOnLine)
        .Cost(4, "Open_AfterOpenGap"),
    // A wide whitespace gap before a token that could start a group suggests
    // an opener was deleted in the gap.
    Rule(Ins::ParenLike, Kinds::GroupStarter)
        .When(Cue::WideGap)
        .Unless(Cue::FirstOnLine)
        .Cost(CostCloseAtWideGap, "Open_AtWideGap"),
    // An empty group directly after a name: `Op()`. Only applies after a
    // leaf (a call of a just-computed value, `f(x)()`, is much rarer than a
    // call of a name), and not when the name is a type after `as` or `->`,
    // where a parenthesized group is more plausible than an empty call.
    Rule(Ins::Paren, Kind::CloseParen)
        .When(Cue::PrevItemIsLeaf, Cue::LeadingSpace)
        .Unless(Cue::PrevItemIsTypeName)
        .Cost(5, "Open_EmptyParens"),
    // An empty `[]` is rarer than empty parens.
    Rule(Ins::Square, Kind::CloseSquareBracket)
        .When(Cue::PrevItemIsLeaf, Cue::LeadingSpace)
        .Unless(Cue::PrevItemIsTypeName)
        .Cost(12, "Open_EmptySquares"),
    // An empty call of a just-computed value: `T.(Default.Op)()`. Only
    // trusted when the `)` is spaced, marking the deletion gap. This looks at
    // the last token of the previous item, so it fires after a collapsed
    // `(...)` block too.
    Rule(Ins::Paren, Kind::CloseParen)
        .When(Cue::PrevIsCloseParen, Cue::LeadingSpace)
        .Unless(Cue::FirstOnLine)
        .Cost(8, "Open_EmptyParensAfterClose"),
    // A `(` or `[` anywhere else an expression could start.
    Rule(Ins::ParenLike).Cost(35, "Open_ParenBaseline"),
    // A scope `{` is never inserted directly before another opener: the body
    // it would open starts with that opener, so the `{` belongs before it only
    // via one of the rules above.
    Rule(Ins::ScopeBrace, Kinds::Opener).Decline(),
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
    Rule(Ins::StructBrace, Kind::CloseCurlyBrace)
        .Unless(Cue::FirstOnLine)
        .AnyOf(Cue::PrevValueEnding, Cue::PrevIsCloseCurly,
               Cue::PrevIsCloseSquare, Cue::PrevIsComma)
        .Cost(10, "Open_StructEmptyBraces"),
    // Any other struct brace has no cue at all, and isn't worth proposing.
    Rule(Ins::StructBrace).Decline(),
};

static_assert(std::size(OpenerRules) <= 64);

constexpr auto OpenerRuleIndex = BuildRuleIndex(OpenerRules);

// Computes the cost of inserting a synthetic opener directly before `item`, or
// nullopt if this insertion isn't worth exploring. Sets `origin` to the rule
// that fired.
auto ClassifyOpenerInsertion(Kind kind, bool is_struct_brace, const Item& item,
                             const char*& origin) -> std::optional<int32_t> {
  const auto* rule = FindMatchingRule(OpenerRules, OpenerRuleIndex,
                                      InsClassOf(kind, is_struct_brace),
                                      item.token.kind, item.cues);
  if (rule == nullptr || rule->cost == DeclineCost) {
    return std::nullopt;
  }
  origin = rule->origin;
  return rule->cost;
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
    Rule(Top::Struct | Top::Scope, Kinds::Dedentable)
        .When(Cue::FirstOnLine, Cue::DedentToHeader)
        .Cost(40, "Adv_DedentInScope"),
    Rule(Top::ParenLike, Kinds::Dedentable)
        .When(Cue::FirstOnLine, Cue::DedentToOpenerLine)
        .Cost(25, "Adv_DedentInParen"),
    // A scope `{` inside parens or a struct brace is a lambda, which is rare;
    // a struct `{` there is a struct literal argument, which is common. The
    // first rule covers a collapsed block that contains a scope brace.
    Rule(Top::ParenLike | Top::Struct, Kinds::Opener)
        .When(Cue::CollapsedBlock, Cue::ContainsScopeBrace)
        .Cost(40, "Adv_ScopeBlockInParen"),
    Rule(Top::ParenLike | Top::Struct, Kind::OpenCurlyBrace)
        .Unless(Cue::CollapsedBlock, Cue::StructBrace)
        .Cost(40, "Adv_ScopeBraceInParen"),
    Rule(Top::ParenLike | Top::Struct, Kind::OpenCurlyBrace)
        .When(Cue::StructBrace)
        .Unless(Cue::CollapsedBlock)
        .Cost(5, "Adv_StructBraceInParen"),
    // A spaced `(` or `[` directly after a value: calls and indexing are
    // written without spaces, so a bracket was likely deleted in the gap —
    // unless this path already inserted a closer there.
    Rule(Top::Any, Kinds::OpenGroup)
        .When(Cue::LeadingSpace, Cue::PrevValueLike)
        .Unless(Cue::FirstOnLine, Cue::CollapsedBlock, Cue::CloserInserted)
        .Cost(10, "Adv_SpacedOpenAfterValue"),
    // Formatted code has no space before a closer either.
    Rule(Top::Any, Kinds::CloseGroup)
        .When(Cue::LeadingSpace, Cue::PrevValueLike)
        .Unless(Cue::FirstOnLine, Cue::BracketInsertedHere)
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
        .When(Cue::LeadingSpace, Cue::PrevValueLike)
        .Unless(Cue::FirstOnLine, Cue::CloserInserted, Cue::AfterOpenTop)
        .Cost(8, "Adv_SpacedCommaInParen"),
    // A statement introducer keyword inside parens or a struct brace.
    Rule(Top::ParenLike | Top::Struct, Kind::StatementIntroducer)
        .Cost(60, "Adv_IntroducerInParen"),
    // A leaf, or a binding modifier keyword, directly following a value-ending
    // token is an illegal adjacency — unless an opener synthesized here, or a
    // `]`/`}` inserted here, repairs it.
    Rule(Top::Any, Kinds::LeafLike)
        .When(Cue::PrevValueEnding)
        .Unless(Cue::OpenerHere, Cue::CloserFixesAdjacency)
        .Cost(60, "Adv_LeafAdjacency"),
    // `=`, `->`, or `as` inside parens or square brackets. These *can* occur
    // there (default arguments, function types, casts), so this is mild; it
    // serves to prefer the earliest sensible close point. Casts `(x as T)` are
    // common enough that `as` keeps only a nominal preference.
    Rule(Top::ParenLike, Kind::Assignment, Kind::StructuralOp)
        .When(Cue::LeadingSpace)
        .Cost(10, "Adv_StructuralOpInParen"),
    Rule(Top::ParenLike, Kind::As)
        .When(Cue::LeadingSpace)
        .Cost(1, "Adv_AsOpInParen"),
    // A comparison or logical operator inside square brackets or a call.
    Rule(Top::Square, Kind::ComparisonOp).Cost(8, "Adv_ComparisonInSquare"),
    Rule(Top::Paren, Kind::ComparisonOp)
        .When(Cue::CallParenTop)
        .Cost(8, "Adv_ComparisonInCall"),
    // A wide mid-line whitespace gap suggests a deleted bracket that this path
    // hasn't repaired.
    Rule(Top::Any, Kinds::AnyOther)
        .When(Cue::WideGap)
        .Unless(Cue::FirstOnLine, Cue::BracketInsertedHere)
        .Cost(10, "Adv_WideGapUnexplained"),
    // A mid-line `.` with whitespace before it suggests a deleted bracket:
    // member access is written without spaces. Prefer closing an open group
    // before it, or opening one.
    Rule(Top::Any, Kind::Period)
        .When(Cue::LeadingSpace)
        .Unless(Cue::FirstOnLine, Cue::BracketInsertedHere)
        .AnyOf(Cue::PrevValueLike, Cue::PrevIsOpenBracket, Cue::PrevIsOpenCurly)
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
auto ComputeAdvanceCues(const SearchState& node, const Item& item) -> CueSet {
  bool opener_here = OpenerSynthesizedHere(node.stack, item);
  bool closer_here = node.closer_inserted != Kind::Other;
  CueSet cues = item.cues | CueIf(closer_here, Cue::CloserInserted) |
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
  return SumMatchingRules(AdvanceRules, AdvanceRuleIndex,
                          TopClassOfStack(node.stack), item.token.kind,
                          ComputeAdvanceCues(node, item));
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
    if (all_paths.size() >= MaxOptimalPaths) {
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
  Map<int32_t, int32_t> token_to_item;
  for (auto [idx, region_item] : llvm::enumerate(items)) {
    token_to_item.Update(region_item.token.token_index.index,
                         static_cast<int32_t>(idx));
  }
  token_to_item.Update(region_end_token.index,
                       static_cast<int32_t>(items.size()));
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
    int32_t* a_item = token_to_item[a.fix_token_index.index];
    int32_t* b_item = token_to_item[b.fix_token_index.index];
    if (a_item == nullptr || b_item == nullptr) {
      return false;
    }
    auto [lo, hi] = std::minmax(*a_item, *b_item);
    for (const Item& between : items.slice(lo, hi - lo)) {
      if (between.Has(Cue::CollapsedBlock) ||
          ToTokenKind(between.token.kind) != a.fix_token_kind) {
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
// `region_end_token` is the token directly after the region, where any
// still-unclosed brackets are closed.
auto SolveRegionCostBased(llvm::ArrayRef<Item> items,
                          TokenIndex region_end_token,
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
          Map<uint64_t, int32_t>& layer_map, int32_t next_item_idx,
          llvm::SmallVector<OpenBracketInfo, 4> next_stack,
          Kind closer_inserted, int32_t next_cost, ParentEdge edge,
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
        int32_t* hash_match = layer_map[stack_hash];
        if (hash_match != nullptr) {
          if (arena[*hash_match].stack == next_stack &&
              arena[*hash_match].closer_inserted == closer_inserted) {
            merge_into(*hash_match);
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
        layer_map.Update(stack_hash, new_idx);
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
  Map<uint64_t, int32_t> layer_map;

  for (auto [item_index, item] : llvm::enumerate(items)) {
    auto i = static_cast<int32_t>(item_index);
    auto kind = item.token.kind;

    // Step 1: Epsilon moves within layer `i` (insertions before token `i`).
    if (kind != Kind::FileEnd) {
      for (int32_t idx : current_layer) {
        layer_map.Update(
            StateHash(arena[idx].stack, arena[idx].closer_inserted), idx);
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
          if (kind == Kind::CloseCurlyBrace && !top.is_struct_brace &&
              item.token.line != top.line &&
              item.token.line_indent < top.effective_header_indent) {
            direct_match_ok = false;
          }
          bool spaced_suspicious =
              kind != Kind::CloseCurlyBrace && !item.Has(Cue::FirstOnLine) &&
              item.Has(Cue::LeadingSpace, Cue::PrevValueLike);
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
      // The bound is taken before the loop because it appends to the layer.
      for (size_t idx : llvm::seq<size_t>(0, current_layer.size())) {
        int32_t node_idx = current_layer[idx];
        const SearchState current = Snapshot(arena[node_idx]);
        if (current.cost > min_goal_cost ||
            current.stack.size() >= MaxSearchStackDepth) {
          continue;
        }

        auto push_synthetic = [&](Kind open_kind, bool is_struct_brace,
                                  int32_t add_cost, const char* origin) {
          auto next_stack = current.stack;
          next_stack.push_back(OpenBracketInfo{
              OpenBracketKey{
                  .insertion_token_index = item.token.token_index,
                  .line = item.token.line,
                  .line_indent = item.token.line_indent,
                  .effective_header_indent = item.effective_header_indent,
                  .kind = open_kind,
                  .is_synthetic = true,
                  .is_struct_brace = is_struct_brace,
              },
              origin,
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
        auto propose_opener = [&](Kind open_kind, bool is_struct_brace) {
          const char* origin = "";
          if (auto cost = ClassifyOpenerInsertion(open_kind, is_struct_brace,
                                                  item, origin)) {
            push_synthetic(open_kind, is_struct_brace, *cost, origin);
          }
        };
        propose_opener(Kind::OpenParen, /*is_struct_brace=*/false);
        propose_opener(Kind::OpenSquareBracket,
                       /*is_struct_brace=*/false);
        propose_opener(Kind::OpenCurlyBrace,
                       /*is_struct_brace=*/false);
        propose_opener(Kind::OpenCurlyBrace,
                       /*is_struct_brace=*/true);
      }

      layer_map.Clear();
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
                             std::move(next_stack), Kind::Other,
                             current.cost + add_cost, edge, nullptr);
          };

      // What it costs to advance over this item in this state, per the
      // `AdvanceRules` table.
      int32_t penalty = AdvancePenalty(current, item);

      if (item.Has(Cue::CollapsedBlock)) {
        try_enqueue_advance(current.stack, penalty);
        continue;
      }

      if (IsOpeningBracket(kind)) {
        // Advance and push opener onto stack.
        if (current.stack.size() < MaxSearchStackDepth) {
          auto next_stack = current.stack;
          next_stack.push_back(OpenBracketInfo{OpenBracketKey{
              .token_index = item.token.token_index,
              .token_pos = item.token_start_index,
              .line = item.token.line,
              .line_indent = item.token.line_indent,
              .effective_header_indent = item.effective_header_indent,
              .kind = kind,
              .is_struct_brace = item.token.is_struct_brace,
              .is_call_paren = item.Has(Cue::PrevValueEnding),
          }});
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
          if (kind == Kind::CloseCurlyBrace && !top.is_struct_brace &&
              item.token.line != top.line) {
            // A multi-line scope close must not be dedented past its header,
            // and pays for indentation disagreement with its header.
            if (item.token.line_indent < top.effective_header_indent) {
              allow_match = false;
            } else if (item.Has(Cue::FirstOnLine) &&
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
    layer_map.Clear();
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
      if (entry.kind == Kind::OpenCurlyBrace) {
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

  for (int32_t i : llvm::seq(0, num_tokens)) {
    auto kind = tokens[i].kind;
    if (IsOpeningBracket(kind)) {
      open_stack.push_back(i);
    } else if (IsClosingBracket(kind)) {
      int32_t match_s = -1;
      if (kind == Kind::CloseCurlyBrace) {
        for (int32_t s = static_cast<int32_t>(open_stack.size()) - 1; s >= 0;
             --s) {
          int32_t cand = open_stack[s];
          if (tokens[cand].kind != Kind::OpenCurlyBrace) {
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
  for (int32_t i : llvm::seq(1, num_tokens)) {
    auto prev_kind = tokens[i - 1].kind;
    bool new_seg = prev_kind == Kind::Semi ||
                   prev_kind == Kind::OpenCurlyBrace ||
                   prev_kind == Kind::CloseCurlyBrace;
    seg_id[i] = seg_id[i - 1] + (new_seg ? 1 : 0);
    seg_first[i] = new_seg ? i : seg_first[i - 1];
  }

  // Sorted lists of unmatched openers and closers, by kind, for the
  // cleanliness checks below.
  llvm::SmallVector<int32_t> unmatched_open_parens;
  llvm::SmallVector<int32_t> unmatched_open_squares;
  llvm::SmallVector<int32_t> unmatched_close_parens;
  llvm::SmallVector<int32_t> unmatched_close_squares;
  for (int32_t i : llvm::seq(0, num_tokens)) {
    if (match_partner[i] != -1) {
      continue;
    }
    switch (tokens[i].kind) {
      case Kind::OpenParen:
        unmatched_open_parens.push_back(i);
        break;
      case Kind::OpenSquareBracket:
        unmatched_open_squares.push_back(i);
        break;
      case Kind::CloseParen:
        unmatched_close_parens.push_back(i);
        break;
      case Kind::CloseSquareBracket:
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

  for (int32_t i : llvm::seq(0, num_tokens)) {
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
  for (int32_t i : llvm::reverse(llvm::seq(0, num_tokens))) {
    auto kind = tokens[i].kind;
    if (match_partner[i] == -1 || match_partner[i] <= i) {
      continue;
    }
    int32_t close_idx = match_partner[i];
    bool clean = true;

    if (kind == Kind::OpenCurlyBrace) {
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
        auto bad_kind = tokens[i].is_struct_brace ? Kind::Semi : Kind::Comma;
        int32_t depth = 0;
        for (int32_t j : llvm::seq(i + 1, close_idx)) {
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
      const auto& unmatched_openers = kind == Kind::OpenParen
                                          ? unmatched_open_parens
                                          : unmatched_open_squares;
      const auto& unmatched_closers = kind == Kind::OpenParen
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
      for (int32_t j : llvm::seq(i + 1, close_idx)) {
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
        if (kind == Kind::OpenCurlyBrace && !tokens[i].is_struct_brace &&
            is_first_on_line[j] && tokens[j].line != tokens[close_idx].line &&
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
    // Cues the surrounding analysis has already determined; the rest follow
    // from the tokens themselves.
    CueSet context_cues =
        CueIf(collapsed, Cue::CollapsedBlock) |
        CueIf(has_scope, Cue::ContainsScopeBrace) |
        CueIf(is_first_on_line[start], Cue::FirstOnLine) |
        CueIf(follows_statement_header[start], Cue::FollowsStatementHeader) |
        CueIf(header_has_open_curly_brace[start], Cue::HeaderHasOpenCurly);
    items.push_back(Item{
        .token_start_index = start,
        .token_end_index = end,
        .token = tokens[start],
        .effective_header_indent = effective_header_indent[start],
        .cues = ComputeItemCues(
            tokens[start], start > 0 ? &tokens[start - 1] : nullptr,
            items.empty() ? nullptr : &items.back(), context_cues),
    });
  };
  for (int32_t i = 0; i < num_tokens;) {
    if (is_clean_range[i] && match_partner[i] != -1) {
      int32_t close_idx = match_partner[i];
      bool has_scope = false;
      for (int32_t j : llvm::seq(i, close_idx + 1)) {
        if (tokens[j].kind == Kind::OpenCurlyBrace &&
            !tokens[j].is_struct_brace) {
          has_scope = true;
          break;
        }
      }
      make_item(i, close_idx, /*collapsed=*/true, has_scope);
      i = close_idx + 1;
    } else {
      make_item(i, i, /*collapsed=*/false,
                tokens[i].kind == Kind::OpenCurlyBrace ||
                    tokens[i].kind == Kind::CloseCurlyBrace);
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

  for (auto [item_index, item] : llvm::enumerate(items)) {
    auto i = static_cast<int32_t>(item_index);
    // Note: line_indent is a 1-based column number, so top-level tokens have
    // line_indent 1.
    if (i > 0 && !item.Has(Cue::CollapsedBlock) &&
        item.token.kind == Kind::StatementIntroducer &&
        !item.token.is_else_keyword && item.token.line_indent <= 1 &&
        item.Has(Cue::FirstOnLine)) {
      auto prev_end_kind = tokens[items[i - 1].token_end_index].kind;
      if ((prev_end_kind == Kind::Semi ||
           prev_end_kind == Kind::CloseCurlyBrace) &&
          region_boundaries.back() != i) {
        region_boundaries.push_back(i);
      }
    }
  }
  if (region_boundaries.back() != static_cast<int32_t>(items.size())) {
    region_boundaries.push_back(static_cast<int32_t>(items.size()));
  }

  // Each region runs between consecutive boundaries.
  for (auto [start, end] :
       llvm::zip(region_boundaries, llvm::drop_begin(region_boundaries))) {
    if (start >= end) {
      continue;
    }

    // A region needs solving only if its loose (non-collapsed) brackets don't
    // already form a balanced, well-nested sequence. A balanced region has no
    // unmatched bracket, so the search would simply match everything and emit
    // no corrections; skipping it avoids running the beam search over the many
    // regions whose matched pairs merely weren't collapsed.
    bool balanced = true;
    llvm::SmallVector<Kind> open_kinds;
    for (int32_t i = start; i < end && balanced; ++i) {
      if (items[i].Has(Cue::CollapsedBlock)) {
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
      auto slice = llvm::ArrayRef<Item>(items).slice(start, end - start);
      SolveRegionCostBased(slice, region_end_token, corrections);
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

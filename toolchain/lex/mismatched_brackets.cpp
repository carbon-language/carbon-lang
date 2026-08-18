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
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon::Lex {

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
// The individual numbers here and in the rule tables below carry no meaning
// beyond how they order the repairs they price. They were not derived; they
// were hill-climbed by coordinate descent against `mismatched_brackets_eval`,
// which deletes brackets from real Carbon source and scores what recovery puts
// back, including a per-rule precision breakdown. So a cost is best changed the
// same way it was chosen: adjust it, rerun the evaluation, and keep the change
// only if the scores improve. Rebalancing is worth doing when the evaluation
// reports poor precision for a rule, and is required when adding a rule, whose
// cost only means something relative to the rules it competes with. See
// /toolchain/docs/lex/mismatched_brackets.md for how to run it.
//
// Costs of replacing a real bracket with an error token. These are the
// "give up on this bracket" fallbacks; a good targeted repair should beat
// them, and a dubious one should lose to them.
constexpr int32_t CostReplaceClosing = 30;
constexpr int32_t CostReplaceOpening = 50;

// Costs of inserting a synthetic closing bracket in front of the current
// token live in the `CloserRules` table below, which is keyed by how strongly
// the context suggests the group ends here. These few are shared with the
// region-end handling, which isn't part of that table.
// Closing anything at the end of the file or region.
constexpr int32_t CostCloseAtEnd = 12;
constexpr int32_t CostCloseParenAtEnd = 22;
constexpr int32_t CostCloseStructAtEnd = 20;
// Closing a paren/square bracket before a mid-line `.` that has whitespace
// before it: member access is normally written without spaces. Priced below the
// `Adv_SpacedPeriodInParen` rule in `AdvanceRules` below, which penalizes
// stepping over such a `.` instead, so that closing here beats closing earlier
// and leaving the spaced `.` unexplained.
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

namespace {

// A contextual cue a rule can test, beyond the bucket's context category and
// token kind. These are bit indexes into a `CueSet`; no cue's value is ever
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

}  // namespace

// The set containing exactly `cues`.
template <std::same_as<Cue>... CueT>
static constexpr auto CueSetOf(CueT... cues) -> CueSet {
  return (CueSet::None | ... |
          static_cast<CueSet>(uint64_t{1} << static_cast<int>(cues)));
}

// Convenience synonym for our token kinds.
using Kind = BracketTokenKind;

namespace {

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
  constexpr auto HasAll(CueT... wanted) const -> bool {
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
  // For a synthetic opener, the name of the rule that proposed it, for
  // debugging. Two openers that differ only in which rule proposed them are the
  // same state, so this is deliberately outside the key.
  llvm::StringLiteral rule_name = "";
};

// Names a stack of open brackets held in an `OpenStackStore`.
using OpenStackId = int32_t;

// The empty stack, which every search starts from. Interning gives it no cell,
// so it needs an id of its own.
constexpr OpenStackId EmptyOpenStack = -1;

// What distinguishes one stack from another: the bracket on top, and the stack
// below it. Only the `OpenBracketKey` part of the bracket takes part, since
// that is all that distinguishes one search state from another.
struct OpenStackCellKey {
  OpenStackId parent;
  OpenBracketKey entry;

  friend auto operator==(const OpenStackCellKey& lhs,
                         const OpenStackCellKey& rhs) -> bool = default;
};

static_assert(std::has_unique_object_representations_v<OpenStackCellKey>,
              "Padding would leave indeterminate bytes for hashing.");

inline auto CarbonHashValue(const OpenStackCellKey& key, uint64_t seed)
    -> HashCode {
  Hasher hasher(seed);
  hasher.HashRaw(key);
  return static_cast<HashCode>(hasher);
}

// Holds the open-bracket stacks the search reaches.
//
// A stack only ever changes at the top, so it is stored as the bracket on top
// plus the id of the stack below it, which lets the many states sharing a
// prefix share its cells instead of each carrying a copy. Equal stacks are
// interned to one id, so comparing two stacks is comparing two integers, and a
// `BeamNode` holds no bracket storage of its own — both of which matter, since
// the search creates far more states than it keeps and never frees them.
//
// Each cell also carries the answers to the questions the rules ask about the
// whole stack, so that asking them doesn't mean walking it.
class OpenStackStore {
 public:
  // The stack that is `stack` with `entry` pushed onto it.
  auto Push(OpenStackId stack, const OpenBracketInfo& entry) -> OpenStackId {
    auto result =
        intern_.Insert(OpenStackCellKey{.parent = stack, .entry = entry},
                       static_cast<OpenStackId>(cells_.size()));
    if (result.is_inserted()) {
      cells_.push_back({
          .parent = stack,
          .entry = entry,
          .depth = Depth(stack) + 1,
          .open_kinds =
              static_cast<uint8_t>(OpenKinds(stack) | KindBit(entry.kind)),
          .has_synthetic = HasSynthetic(stack) || entry.is_synthetic,
      });
    }
    return result.value();
  }

  // The stack below the top of `stack`, which must not be empty.
  auto Pop(OpenStackId stack) const -> OpenStackId {
    return cells_[stack].parent;
  }

  // The bracket on top of `stack`, which must not be empty.
  auto Top(OpenStackId stack) const -> const OpenBracketInfo& {
    return cells_[stack].entry;
  }

  auto IsEmpty(OpenStackId stack) const -> bool {
    return stack == EmptyOpenStack;
  }

  auto Depth(OpenStackId stack) const -> int32_t {
    return IsEmpty(stack) ? 0 : cells_[stack].depth;
  }

  // Whether any bracket in `stack` is synthetic.
  auto HasSynthetic(OpenStackId stack) const -> bool {
    return !IsEmpty(stack) && cells_[stack].has_synthetic;
  }

  // Whether `stack` holds an opener of kind `kind`.
  auto Contains(OpenStackId stack, BracketTokenKind kind) const -> bool {
    return !IsEmpty(stack) && (cells_[stack].open_kinds & KindBit(kind)) != 0;
  }

 private:
  struct Cell {
    OpenStackId parent;
    OpenBracketInfo entry;
    int32_t depth;
    // The kinds of every bracket in this stack, as `KindBit`s.
    uint8_t open_kinds;
    bool has_synthetic;
  };

  static auto KindBit(BracketTokenKind kind) -> uint8_t {
    return static_cast<uint8_t>(1 << static_cast<int>(kind));
  }

  auto OpenKinds(OpenStackId stack) const -> uint8_t {
    return IsEmpty(stack) ? 0 : cells_[stack].open_kinds;
  }

  llvm::SmallVector<Cell, 0> cells_;
  Map<OpenStackCellKey, OpenStackId> intern_;
};

}  // namespace

// The `{` opening the branch that the first-on-line `else` at `else_index`
// continues. Such an `else` should have been preceded by the `}` closing that
// branch, so with the `}` missing the `else` is still inside it: the brace is
// the innermost one that nothing has closed yet. Returns -1 if there is no
// enclosing brace.
static auto FindBranchBrace(llvm::ArrayRef<MismatchedBracketToken> tokens,
                            int32_t else_index) -> int32_t {
  int32_t depth = 0;
  for (int32_t j : llvm::reverse(llvm::seq(0, else_index))) {
    if (tokens[j].kind == Kind::CloseCurlyBrace) {
      ++depth;
    } else if (tokens[j].kind == Kind::OpenCurlyBrace) {
      if (depth == 0) {
        return j;
      }
      --depth;
    }
  }
  return -1;
}

// Computes the associated line indentation for a token by scanning backwards,
// skipping matched parens/brackets, looking for a statement introducer.
static auto GetOuterStatementIntroducerIndent(
    llvm::ArrayRef<MismatchedBracketToken> tokens,
    llvm::ArrayRef<int32_t> match_partner, int32_t j) -> int32_t {
  int32_t result_indent = tokens[j].line_indent;
  while (j > 0) {
    // An `else` continues the statement its `if` introduced, and the `}`
    // closing its block lines up with that `if`, so keep walking out from the
    // brace of the branch before it. A first-on-line `else`'s own column says
    // nothing about where the statement starts: the `}` that should precede it
    // is missing, so the `else` sits wherever the author left it.
    if (tokens[j].is_else_keyword && tokens[j].line != tokens[j - 1].line) {
      int32_t brace = FindBranchBrace(tokens, j);
      if (brace > 0) {
        j = brace;
        result_indent = tokens[j].line_indent;
        continue;
      }
    }
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
  return result_indent;
}

// Computes, for every token, the indentation that its statement or declaration
// starts at. This is what a `{` opened by that statement should line its `}` up
// with, which is often not the indentation of the `{` itself.
//
// The answer for one token is a walk backwards to the start of the statement it
// belongs to: it steps over tokens, jumps over matched bracket pairs, stops at
// a `;` or an unmatched brace, and answers with the indentation of the last
// token it stepped on, or of the enclosing statement introducer if it reaches
// one.
//
// Every step of that walk depends only on the token it is standing on, so the
// walks from all the tokens share their tails and can be solved in one pass.
// `walk_result[j]` is what the walk starting at token `j` answers, or
// `NeedsCaller` when the walk stops before it steps on anything, in which case
// the answer is the caller's own indentation. Each entry depends only on
// entries below it, so a single forward pass fills the table, and the answer
// for token `i` reads the entry for `i - 1`.
static auto ComputeAssociatedLineIndents(
    llvm::ArrayRef<MismatchedBracketToken> tokens,
    llvm::ArrayRef<int32_t> match_partner) -> llvm::SmallVector<int32_t> {
  auto num_tokens = static_cast<int32_t>(tokens.size());

  // A walk that stopped without stepping on a token, so it has no indentation
  // of its own to report.
  constexpr int32_t NeedsCaller = std::numeric_limits<int32_t>::min();

  llvm::SmallVector<int32_t> walk_result(num_tokens, NeedsCaller);
  // The walk from `j` continues at `next`, so it answers what that walk
  // answers, falling back to `indent` when that walk has nothing to report.
  auto continue_walk = [&](int32_t next, int32_t indent) {
    return next >= 0 && walk_result[next] != NeedsCaller ? walk_result[next]
                                                         : indent;
  };

  for (int32_t j : llvm::seq(0, num_tokens)) {
    auto kind = tokens[j].kind;

    // A matched closer jumps to its opener, which the walk passes over without
    // testing it, and continues from the token before it.
    if (IsClosingBracket(kind) && match_partner[j] != -1 &&
        match_partner[j] < j) {
      int32_t open = match_partner[j];
      walk_result[j] = continue_walk(open - 1, tokens[open].line_indent);
      continue;
    }

    // The statement ends here, so the walk stops without stepping on this
    // token. `walk_result[j]` stays `NeedsCaller`.
    if (kind == Kind::Semi || kind == Kind::OpenCurlyBrace ||
        kind == Kind::CloseCurlyBrace) {
      continue;
    }

    // The statement introducer this token belongs to answers for it directly.
    if (kind == Kind::StatementIntroducer) {
      walk_result[j] =
          GetOuterStatementIntroducerIndent(tokens, match_partner, j);
      continue;
    }

    walk_result[j] = continue_walk(j - 1, tokens[j].line_indent);
  }

  llvm::SmallVector<int32_t> indents(num_tokens, 0);
  for (int32_t i : llvm::seq(0, num_tokens)) {
    if (tokens[i].kind == Kind::FileEnd) {
      continue;
    }
    indents[i] = continue_walk(i - 1, tokens[i].line_indent);
  }
  return indents;
}

// Determines if a token follows a statement/declaration header, so that a
// scope `{` could naturally be inserted directly before it. Only tokens that
// could start a body are considered: the first token on a line, a statement
// introducer (e.g. `return` in `if (c) return;`), or a token directly
// following a `)`/`]` that ends a header.
static auto ComputeFollowsStatementHeader(
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
static auto ComputeHeaderHasOpenCurlyBrace(
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

namespace {

struct ParentEdge {
  int32_t parent_node_index;
  BracketCorrection correction;
  bool has_correction = false;
};

// Node in the beam search tree.
//
// The search creates far more nodes than any one layer keeps, and never frees
// them, so how big a node is decides how much of the search is spent waiting on
// memory rather than deciding anything. That is why its stack is an id into a
// shared `OpenStackStore` rather than a stack of its own, and why
// `parent_edges` has inline room for the one edge a node usually has rather
// than for the worst case.
struct BeamNode {
  int32_t item_index;
  OpenStackId stack;
  int32_t cost;
  // The kind of synthetic closer inserted directly before the current item,
  // or Other if none; an inserted closer repairs illegal adjacency with the
  // preceding token.
  Kind closer_inserted = Kind::Other;
  llvm::SmallVector<ParentEdge, 1> parent_edges;
};

// The parts of a `BeamNode` the search reads while expanding it. A copy — not a
// reference — is needed because expanding a node appends to the node list,
// which may reallocate.
struct SearchState {
  OpenStackId stack;
  int32_t cost;
  Kind closer_inserted;
};

// What makes a search state distinct: the open-bracket stack, plus which
// closer, if any, was just inserted before the current token. Two nodes in a
// layer that agree on this are interchangeable and get merged, which is what
// `RegionSearch::layer_dedup_` is keyed on. Stacks are interned, so this is two
// integers.
struct StateKey {
  OpenStackId stack;
  Kind closer_inserted;

  friend auto operator==(const StateKey& lhs, const StateKey& rhs)
      -> bool = default;

  friend auto CarbonHashValue(const StateKey& key, uint64_t seed) -> HashCode {
    Hasher hasher(seed);
    hasher.Hash(key.stack, key.closer_inserted);
    return static_cast<HashCode>(hasher);
  }
};

}  // namespace

static auto Snapshot(const BeamNode& node) -> SearchState {
  return {node.stack, node.cost, node.closer_inserted};
}

// A correction that replaces a bracket token with an error token (the "give
// up on this bracket" repair).
static auto ReplaceWithError(const MismatchedBracketToken& token,
                             BracketDiagnosticKind diagnostic_kind,
                             llvm::StringLiteral rule_name)
    -> BracketCorrection {
  return BracketCorrection{
      .diagnostic_kind = diagnostic_kind,
      .diagnostic_token_index = token.token_index,
      .fix_action = BracketFixAction::ReplaceWithError,
      .fix_token_index = token.token_index,
      .fix_token_kind = ToTokenKind(token.kind),
      .rule_name = rule_name,
  };
}

// Solve a damaged region using the simple greedy fallback algorithm.
static auto SolveNaive(llvm::ArrayRef<Item> items,
                       llvm::SmallVectorImpl<BracketCorrection>& corrections)
    -> void {
  llvm::SmallVector<MismatchedBracketToken> open_stack;
  for (const auto& item : items) {
    if (item.HasAll(Cue::CollapsedBlock)) {
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

// Whether two parent edges represent the same predecessor and the same repair,
// so that keeping both would be a redundant duplicate.
static auto EdgesEqual(const ParentEdge& a, const ParentEdge& b) -> bool {
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
static auto MatchesDeeperOpener(const OpenStackStore& stacks, OpenStackId stack,
                                Kind closing_kind) -> bool {
  if (!IsClosingBracket(closing_kind) || stacks.Depth(stack) < 2) {
    return false;
  }
  return stacks.Contains(stacks.Pop(stack), MatchingOpeningKind(closing_kind));
}

// Whether the token before the current one ends a value. Unlike
// `IsValueEndingKind`, which the leaf-adjacency rules use, this also counts `]`
// and `}`, which end a value but can be followed by a leaf. `prev_token` is
// null at the start of the input.
static auto PrevIsValueLike(const MismatchedBracketToken* prev_token) -> bool {
  return prev_token != nullptr &&
         (IsValueEndingKind(prev_token->kind) ||
          prev_token->kind == Kind::CloseSquareBracket ||
          prev_token->kind == Kind::CloseCurlyBrace);
}

// Whether the top of `stack` is a synthetic opener inserted directly before
// `item` — i.e. this path just opened a bracket at this position.
static auto OpenerSynthesizedHere(const OpenStackStore& stacks,
                                  OpenStackId stack, const Item& item) -> bool {
  return !stacks.IsEmpty(stack) && stacks.Top(stack).is_synthetic &&
         stacks.Top(stack).insertion_token_index == item.token.token_index;
}

// Whether a `]` or `}` was inserted directly before the current token. Such a
// closer repairs an illegal leaf adjacency (unlike `)`, these don't end a
// value). `Other` means no closer was inserted here.
static auto CloserFixesLeafAdjacency(Kind closer_inserted) -> bool {
  return closer_inserted == Kind::CloseSquareBracket ||
         closer_inserted == Kind::CloseCurlyBrace;
}

// The bracket-insertion rules below are expressed as data: each rule states
// the context it applies in and what that context costs, and the rules are
// tried in order so that the first (most specific) match wins. Two small
// categorical facts — a context category and the kind of the current token —
// form a bucket index, and a constexpr-built table maps each bucket to the
// bit-set of rules that can possibly apply there, so a lookup only tests the
// handful of rules relevant to the situation.

// Context categories for the closer and advance tables: the category of the
// innermost open bracket. `Struct` and `Scope` distinguish the two kinds of
// `{`. Each category has an index, which selects its row of bucket entries, and
// a bit, which is how a rule names it — so a rule can apply to several
// categories at once.
namespace {

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

// Context categories for the opener table: which bracket would be inserted.
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

}  // namespace

// Bucket rows are shared by all the tables, so there must be room for whichever
// set of context categories is larger.
constexpr int32_t NumContextCategories =
    std::max<int32_t>(Top::Count, Ins::Count);

// The category of the innermost open bracket, which is always a real bracket.
static auto TopCategoryOf(const OpenBracketInfo& top) -> int32_t {
  switch (top.kind) {
    case Kind::OpenParen:
      return Top::ParenIndex;
    case Kind::OpenSquareBracket:
      return Top::SquareIndex;
    default:
      return top.is_struct_brace ? Top::StructIndex : Top::ScopeIndex;
  }
}

// As `TopCategoryOf`, but for a stack that may be empty.
static auto TopCategoryOfStack(const OpenStackStore& stacks, OpenStackId stack)
    -> int32_t {
  return stacks.IsEmpty(stack) ? Top::NoneIndex
                               : TopCategoryOf(stacks.Top(stack));
}

// The category of a synthetic opener of kind `kind`.
static auto InsCategoryOf(Kind kind, bool is_struct_brace) -> int32_t {
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
namespace {

enum class KindSet : uint32_t {
  None = 0,
  LLVM_MARK_AS_BITMASK_ENUM(uint32_t{1} << (NumKinds - 1))
};

}  // namespace

// The set containing exactly `kinds`.
template <std::same_as<Kind>... KindT>
static constexpr auto KindSetOf(KindT... kinds) -> KindSet {
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

// A rule in a bracket-insertion table. The rule applies when the context
// category is in `ctx`, the current token's kind is in `kinds`, and all four
// cue conditions hold. Rules are tried in order and the first match wins, so
// earlier rules express stronger, more specific cues.
namespace {

struct BracketRule {
  // Bit-set of context categories: `Top::` for the closer table (the category
  // of the innermost open bracket), `Ins::` for the opener table (which bracket
  // would be inserted).
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
  // See `BracketCorrection::rule_name`.
  llvm::StringLiteral rule_name = "";

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
  constexpr auto Cost(int32_t rule_cost, llvm::StringLiteral name) const
      -> BracketRule {
    auto result = *this;
    result.cost = rule_cost;
    result.rule_name = name;
    return result;
  }
  constexpr auto Decline() const -> BracketRule {
    auto result = *this;
    result.cost = DeclineCost;
    return result;
  }
};

}  // namespace

// Starts a rule that applies to context categories `ctx` and token kinds
// `kinds`.
static constexpr auto Rule(uint8_t ctx, KindSet kinds = Kinds::Any)
    -> BracketRule {
  return BracketRule{.ctx = ctx, .kinds = kinds, .cost = DeclineCost};
}

// As above, for a rule that applies to a few particular kinds. Taking the first
// kind separately keeps the no-kinds call unambiguous.
template <std::same_as<Kind>... RestT>
static constexpr auto Rule(uint8_t ctx, Kind kind, RestT... rest)
    -> BracketRule {
  return Rule(ctx, KindSetOf(kind, rest...));
}

// Whether `rule`'s cue conditions hold for the cue bit-set `cues`.
static constexpr auto Matches(const BracketRule& rule, CueSet cues) -> bool {
  return (cues & rule.when) == rule.when &&
         (cues & rule.unless) == CueSet::None &&
         (rule.not_all == CueSet::None ||
          (cues & rule.not_all) != rule.not_all) &&
         (rule.any_of == CueSet::None || (cues & rule.any_of) != CueSet::None);
}

// Where to insert a synthetic closing bracket, and what that costs. All costs
// are relative; see the cost model comment at the top of this file for what
// they mean and how to change one.
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
        .Cost(4, "Close_EmptyGroup"),
    Rule(Top::ParenLike, Kinds::BinaryConnector)
        .When(Cue::AfterOpenTop)
        .Cost(4, "Close_EmptyGroup"),
    Rule(Top::ParenLike, Kind::Period)
        .When(Cue::AfterOpenTop, Cue::LeadingSpace)
        .Cost(4, "Close_EmptyGroup"),
    // A `;` can't appear inside parens or square brackets at all.
    Rule(Top::ParenLike, Kind::Semi).Cost(6, "Close_ParenBeforeSemi"),
    Rule(Top::ParenLike).When(Cue::Cascade).Cost(3, "Close_ParenCascade"),
    // A `{` starting a block means the paren should have closed: `if (c) {`,
    // `while (c) {`. A struct-literal `{...}` can legitimately sit inside a
    // *call* paren (`f({.x = 1})`), but not inside a keyword or grouping paren
    // (whose `{` — even an empty `{}` misread as a struct — is a block). This
    // is not a cue for `[`: a `]` is essentially never immediately followed by
    // `{` (only in `fn [captures] {...}`, which is unimplemented), so a `[`
    // should close at an earlier cue, or not here at all.
    Rule(Top::Paren, Kind::OpenCurlyBrace)
        .NotAll(Cue::StructBrace, Cue::CallParenTop)
        .Cost(11, "Close_ParenBeforeBrace"),
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
        .Cost(11, "Close_ParenBeforeComparison"),
    Rule(Top::Paren, Kind::ComparisonOp)
        .When(Cue::CallParenTop)
        .Cost(11, "Close_ParenBeforeComparison"),
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
    // A first-on-line `else` must have been preceded by the `}` closing the
    // branch before it. Ordered ahead of Close_ScopeAtDedent, which would
    // otherwise match first and charge more for the same insertion.
    Rule(Top::Scope)
        .When(Cue::ElseKeyword, Cue::FirstOnLine)
        .Cost(4, "Close_ScopeBeforeElse"),
    Rule(Top::Scope)
        .When(Cue::FirstOnLine, Cue::DedentToHeader)
        .Cost(6, "Close_ScopeAtDedent"),
    Rule(Top::Scope).When(Cue::Cascade).Cost(6, "Close_ScopeCascade"),
    Rule(Top::Scope, Kind::FileEnd)
        .Cost(CostCloseAtEnd, "Close_ScopeAtFileEnd"),
    Rule(Top::Scope).Cost(45, "Close_ScopeBaseline"),
};

// A bucket index: rules are identified by a bit in a `uint64_t`.
static_assert(std::size(CloserRules) <= 64);

// Maps each (context category, token kind) bucket to the bit-set of rules that
// can apply in it, so a lookup tests only those rules, in table order.
using RuleIndex =
    std::array<std::array<uint64_t, NumKinds>, NumContextCategories>;

template <size_t N>
static constexpr auto BuildRuleIndex(const BracketRule (&rules)[N])
    -> RuleIndex {
  RuleIndex index = {};
  for (size_t r = 0; r != N; ++r) {
    for (int32_t t = 0; t != NumContextCategories; ++t) {
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
static constexpr auto CandidateRules(const RuleIndex& index,
                                     int32_t ctx_category, Kind kind)
    -> uint64_t {
  return index[ctx_category][static_cast<int32_t>(kind)];
}

// Returns the first rule that applies, for a first-match table, or null if
// none does.
template <size_t N>
static auto FindMatchingRule(const BracketRule (&rules)[N],
                             const RuleIndex& index, int32_t ctx_category,
                             Kind kind, CueSet cues) -> const BracketRule* {
  uint64_t candidates = CandidateRules(index, ctx_category, kind);
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
static auto SumMatchingRules(const BracketRule (&rules)[N],
                             const RuleIndex& index, int32_t ctx_category,
                             Kind kind, CueSet cues) -> int32_t {
  int32_t total = 0;
  uint64_t candidates = CandidateRules(index, ctx_category, kind);
  while (candidates != 0) {
    const auto& rule = rules[std::countr_zero(candidates)];
    candidates &= candidates - 1;
    if (Matches(rule, cues)) {
      total += rule.cost;
    }
  }
  return total;
}

static constexpr auto CloserRuleIndex = BuildRuleIndex(CloserRules);

// The cue for the previous token having kind `kind`, if any.
static constexpr auto PrevKindCue(Kind kind) -> CueSet {
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
static constexpr auto CueIf(bool holds, Cue cue) -> CueSet {
  return holds ? CueSetOf(cue) : CueSet::None;
}

// Computes every cue that depends only on an item and its neighbours.
// `prev_token` is the token directly before the item and `prev_item` the item
// directly before it, both null at the start of the input. `context_cues` holds
// the cues the caller has already determined from the surrounding token arrays
// (`Cue::CollapsedBlock`, `Cue::FirstOnLine`, and so on).
static auto ComputeItemCues(const MismatchedBracketToken& token,
                            const MismatchedBracketToken* prev_token,
                            const Item* prev_item, CueSet context_cues)
    -> CueSet {
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
    cues |= CueIf(prev_item->token.kind == Kind::Leaf, Cue::PrevItemIsLeaf) |
            CueIf(prev_item->HasAll(Cue::PrevIntroducesType),
                  Cue::PrevItemIsTypeName);
  }
  return cues;
}

// Computes the cues that depend on the innermost open bracket `top` and the
// search state, to combine with `item.cues`.
static auto ComputeTopCues(const OpenBracketInfo& top, const Item& item,
                           const OpenStackStore& stacks, OpenStackId stack)
    -> CueSet {
  const auto& token = item.token;
  return CueIf(top.is_call_paren, Cue::CallParenTop) |
         CueIf(MatchesDeeperOpener(stacks, stack, token.kind), Cue::Cascade) |
         CueIf(top.token_pos == item.token_start_index - 1, Cue::AfterOpenTop) |
         CueIf(token.line_indent <= top.effective_header_indent,
               Cue::DedentToHeader) |
         CueIf(token.line != top.line, Cue::NewLineFromTop);
}

// Computes the cost of inserting a synthetic closer for `top` directly before
// `item`, or nullopt if this insertion isn't worth exploring. Sets `rule_name`
// to the name of the rule that fired.
static auto ClassifyCloserInsertion(const OpenBracketInfo& top,
                                    const Item& item,
                                    const OpenStackStore& stacks,
                                    OpenStackId stack,
                                    llvm::StringLiteral& rule_name)
    -> std::optional<int32_t> {
  CueSet cues = item.cues | ComputeTopCues(top, item, stacks, stack);
  const auto* rule = FindMatchingRule(
      CloserRules, CloserRuleIndex, TopCategoryOf(top), item.token.kind, cues);
  if (rule == nullptr || rule->cost == DeclineCost) {
    return std::nullopt;
  }
  rule_name = rule->rule_name;
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
    Rule(Ins::ParenLike).Cost(70, "Open_ParenBaseline"),
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
    Rule(Ins::ScopeBrace).Cost(30, "Open_ScopeBaseline"),
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

static constexpr auto OpenerRuleIndex = BuildRuleIndex(OpenerRules);

// Computes the cost of inserting a synthetic opener directly before `item`, or
// nullopt if this insertion isn't worth exploring. Sets `rule_name` to the name
// of the rule that fired.
static auto ClassifyOpenerInsertion(Kind kind, bool is_struct_brace,
                                    const Item& item,
                                    llvm::StringLiteral& rule_name)
    -> std::optional<int32_t> {
  const auto* rule = FindMatchingRule(OpenerRules, OpenerRuleIndex,
                                      InsCategoryOf(kind, is_struct_brace),
                                      item.token.kind, item.cues);
  if (rule == nullptr || rule->cost == DeclineCost) {
    return std::nullopt;
  }
  rule_name = rule->rule_name;
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
        .Cost(5, "Adv_StructuralOpInParen"),
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

static constexpr auto AdvanceRuleIndex = BuildRuleIndex(AdvanceRules);

// Computes the cues for advancing over `item` in search state `node`.
static auto ComputeAdvanceCues(const OpenStackStore& stacks,
                               const SearchState& node, const Item& item)
    -> CueSet {
  bool opener_here = OpenerSynthesizedHere(stacks, node.stack, item);
  bool closer_here = node.closer_inserted != Kind::Other;
  CueSet cues = item.cues | CueIf(closer_here, Cue::CloserInserted) |
                CueIf(CloserFixesLeafAdjacency(node.closer_inserted),
                      Cue::CloserFixesAdjacency) |
                CueIf(opener_here, Cue::OpenerHere) |
                CueIf(closer_here || opener_here, Cue::BracketInsertedHere);
  if (stacks.IsEmpty(node.stack)) {
    return cues;
  }
  const auto& top = stacks.Top(node.stack);
  return cues | CueIf(top.is_call_paren, Cue::CallParenTop) |
         CueIf(top.token_pos == item.token_start_index - 1, Cue::AfterOpenTop) |
         CueIf(item.token.line_indent <= top.effective_header_indent,
               Cue::DedentToHeader) |
         CueIf(item.token.line_indent <= top.line_indent,
               Cue::DedentToOpenerLine);
}

// Computes the total penalty for advancing over `item` in search state `node`,
// summing every `AdvanceRules` entry that matches.
static auto AdvancePenalty(const OpenStackStore& stacks,
                           const SearchState& node, const Item& item)
    -> int32_t {
  return SumMatchingRules(
      AdvanceRules, AdvanceRuleIndex, TopCategoryOfStack(stacks, node.stack),
      item.token.kind, ComputeAdvanceCues(stacks, node, item));
}

// Enumerates the distinct optimal repairs, by walking parent edges from each
// goal node back to the root and collecting the corrections along the way. At
// most `MaxOptimalPaths` are returned, so a correction that only a path beyond
// that cap would dispute stays untied.
static auto EnumerateOptimalPaths(llvm::ArrayRef<BeamNode> nodes,
                                  llvm::ArrayRef<int32_t> goal_node_indices)
    -> llvm::SmallVector<llvm::SmallVector<BracketCorrection>> {
  llvm::SmallVector<llvm::SmallVector<BracketCorrection>> all_paths;
  llvm::SmallVector<BracketCorrection> current_path;
  struct StackFrame {
    int32_t node_index;
    int32_t edge_index = 0;
  };
  llvm::SmallVector<StackFrame> stack;
  for (int32_t goal_idx : goal_node_indices) {
    stack.push_back({.node_index = goal_idx, .edge_index = 0});
  }
  while (!stack.empty()) {
    auto& frame = stack.back();
    const auto& node = nodes[frame.node_index];
    if (node.parent_edges.empty()) {
      // The root, so `current_path` is now a complete repair, in reverse.
      all_paths.push_back(current_path);
      std::reverse(all_paths.back().begin(), all_paths.back().end());
      stack.pop_back();
      if (all_paths.size() >= MaxOptimalPaths) {
        break;
      }
      continue;
    }
    if (frame.edge_index > 0) {
      const auto& prev_edge = node.parent_edges[frame.edge_index - 1];
      if (prev_edge.has_correction) {
        current_path.pop_back();
      }
    }
    if (static_cast<size_t>(frame.edge_index) < node.parent_edges.size()) {
      const auto& edge = node.parent_edges[frame.edge_index];
      ++frame.edge_index;
      if (edge.has_correction) {
        current_path.push_back(edge.correction);
      }
      stack.push_back({.node_index = edge.parent_node_index, .edge_index = 0});
    } else {
      stack.pop_back();
    }
  }
  return all_paths;
}

// Finds which of the first optimal path's corrections the optimal repairs
// disagree about. Each of its corrections must be answered by one making the
// same repair, named by `repair_key`, in every other path; one with no
// counterpart in some path is tied.
//
// Repairs are matched by key rather than pairwise, which works because making
// the same repair is an equivalence relation: the baseline's `n`th correction
// with a given key is answered by a path exactly when that path has at least
// `n` corrections with that key. So all that matters is, for each key the
// baseline uses, the fewest corrections any path has with it.
static auto FindTiedCorrections(
    llvm::ArrayRef<llvm::SmallVector<BracketCorrection>> all_paths,
    llvm::function_ref<uint64_t(const BracketCorrection&)> repair_key)
    -> llvm::BitVector {
  llvm::ArrayRef<BracketCorrection> baseline_path = all_paths.front();

  // Number the keys the baseline uses, and count how many of its corrections
  // share each one. A key no baseline correction uses is never asked about, so
  // it goes unnumbered.
  Map<uint64_t, int32_t> key_ids;
  llvm::SmallVector<int32_t> baseline_key_id(baseline_path.size());
  // Which of the baseline's corrections with the same key this one is, counting
  // from one: it needs the path to have at least this many.
  llvm::SmallVector<int32_t> baseline_rank(baseline_path.size());
  llvm::SmallVector<int32_t> baseline_count;
  for (auto [corr_idx, corr] : llvm::enumerate(baseline_path)) {
    auto result = key_ids.Insert(repair_key(corr),
                                 static_cast<int32_t>(baseline_count.size()));
    if (result.is_inserted()) {
      baseline_count.push_back(0);
    }
    int32_t key_id = result.value();
    baseline_key_id[corr_idx] = key_id;
    baseline_rank[corr_idx] = ++baseline_count[key_id];
  }

  // The fewest corrections with each key that any path makes.
  llvm::SmallVector<int32_t> min_count = baseline_count;
  llvm::SmallVector<int32_t> path_count(baseline_count.size());
  for (const auto& path : llvm::drop_begin(all_paths)) {
    std::fill(path_count.begin(), path_count.end(), 0);
    for (const auto& corr : path) {
      if (int32_t* key_id = key_ids[repair_key(corr)]) {
        ++path_count[*key_id];
      }
    }
    for (auto [key_id, count] : llvm::enumerate(path_count)) {
      min_count[key_id] = std::min(min_count[key_id], count);
    }
  }

  llvm::BitVector tied(baseline_path.size());
  for (auto [corr_idx, key_id] : llvm::enumerate(baseline_key_id)) {
    if (min_count[key_id] < baseline_rank[corr_idx]) {
      tied.set(corr_idx);
    }
  }
  return tied;
}

// From the optimal goal nodes, reconstructs the repair corrections and appends
// them to `corrections`. Every optimal repair is enumerated (up to a cap); a
// correction that the optimal repairs disagree about is marked tied, so the
// caller downgrades it to an error token rather than guessing. Falls back to
// naive recovery if no path can be reconstructed.
static auto ReconstructCorrections(
    llvm::ArrayRef<BeamNode> nodes, llvm::ArrayRef<int32_t> goal_node_indices,
    llvm::ArrayRef<Item> items, TokenIndex region_end_token,
    llvm::SmallVectorImpl<BracketCorrection>& corrections) -> void {
  auto all_paths = EnumerateOptimalPaths(nodes, goal_node_indices);
  if (all_paths.empty()) {
    SolveNaive(items, corrections);
    return;
  }

  Map<int32_t, int32_t> token_to_item;
  for (auto [idx, region_item] : llvm::enumerate(items)) {
    token_to_item.Update(region_item.token.token_index.index,
                         static_cast<int32_t>(idx));
  }
  token_to_item.Update(region_end_token.index,
                       static_cast<int32_t>(items.size()));

  // For each item, the first item of the run of consecutive items with its own
  // kind that ends at it. Inserting a bracket anywhere within a run of that
  // same bracket produces the same token sequence — a `)` on either side of an
  // existing `)` reads the same — so an insertion point slides back to the
  // start of such a run, and two that slide to the same place are the same
  // repair. A collapsed item ends a run, since it stands for a whole bracketed
  // block rather than one token.
  llvm::SmallVector<int32_t> run_start(items.size(), 0);
  for (auto [i, item] : llvm::enumerate(items)) {
    auto idx = static_cast<int32_t>(i);
    bool continues_run = idx > 0 && !item.HasAll(Cue::CollapsedBlock) &&
                         !items[idx - 1].HasAll(Cue::CollapsedBlock) &&
                         items[idx - 1].token.kind == item.token.kind;
    run_start[idx] = continues_run ? run_start[idx - 1] : idx;
  }

  // Names the repair a correction makes, so that two corrections making the
  // same repair get the same name. This reads only the fix, not the diagnosed
  // bracket: two paths that blame different brackets but repair the token
  // stream identically don't disagree about the repair.
  //
  // An insertion whose target is in this region is named by the item its
  // insertion point slides back to; anything else is named by the token it
  // fixes. Those are separate numbering spaces, so the name records which it
  // used.
  auto repair_key = [&](const BracketCorrection& c) -> uint64_t {
    int32_t position = c.fix_token_index.index;
    bool position_is_item = false;
    if (c.fix_action == BracketFixAction::InsertBefore) {
      if (int32_t* item = token_to_item[c.fix_token_index.index]) {
        position_is_item = true;
        position = *item;
        if (position > 0 && !items[position - 1].HasAll(Cue::CollapsedBlock) &&
            ToTokenKind(items[position - 1].token.kind) == c.fix_token_kind) {
          position = run_start[position - 1];
        }
      }
    }
    return (static_cast<uint64_t>(c.fix_action) << 40) |
           (static_cast<uint64_t>(c.fix_token_kind.AsInt()) << 33) |
           (static_cast<uint64_t>(position_is_item) << 32) |
           static_cast<uint32_t>(position);
  };

  llvm::BitVector tied = FindTiedCorrections(all_paths, repair_key);
  for (auto [corr_idx, corr] : llvm::enumerate(all_paths.front())) {
    corrections.push_back(corr);
    corrections.back().is_tied = tied[corr_idx];
  }
}

// Whether matching the real closer `item` directly against `top` is strictly
// better than synthesizing a duplicate closer in front of it. That's normally
// the case, but not when the closer isn't allowed to match `top` directly, and
// not when it has suspicious whitespace before it suggesting a closer was
// deleted in the gap.
static auto DirectMatchPreferred(const OpenBracketInfo& top, const Item& item)
    -> bool {
  auto kind = item.token.kind;
  bool direct_match_ok = kind != Kind::CloseCurlyBrace || top.is_struct_brace ||
                         item.token.line == top.line ||
                         item.token.line_indent >= top.effective_header_indent;
  bool spaced_suspicious = kind != Kind::CloseCurlyBrace &&
                           !item.HasAll(Cue::FirstOnLine) &&
                           item.HasAll(Cue::LeadingSpace, Cue::PrevValueLike);
  return direct_match_ok && !spaced_suspicious;
}

namespace {

// A flavor of synthetic opening bracket the search can insert.
struct SyntheticOpener {
  Kind kind;
  bool is_struct_brace;
};

}  // namespace

// The synthetic openers the search considers inserting before a token, in the
// order they're proposed. Each is offered to `ClassifyOpenerInsertion`, which
// decides whether the context supports it.
constexpr SyntheticOpener SyntheticOpeners[] = {
    {Kind::OpenParen, /*is_struct_brace=*/false},
    {Kind::OpenSquareBracket, /*is_struct_brace=*/false},
    {Kind::OpenCurlyBrace, /*is_struct_brace=*/false},
    {Kind::OpenCurlyBrace, /*is_struct_brace=*/true},
};

// The stack entry for the real opening bracket `item`.
static auto RealOpener(const Item& item) -> OpenBracketInfo {
  return OpenBracketInfo{OpenBracketKey{
      .token_index = item.token.token_index,
      .token_pos = item.token_start_index,
      .line = item.token.line,
      .line_indent = item.token.line_indent,
      .effective_header_indent = item.effective_header_indent,
      .kind = item.token.kind,
      .is_struct_brace = item.token.is_struct_brace,
      .is_call_paren = item.HasAll(Cue::PrevValueEnding),
  }};
}

// The extra cost of matching the real closer `item` against `top`, or nullopt
// if the match isn't allowed. A multi-line scope close must not be dedented
// past its header, and pays for indentation disagreement with it.
static auto MatchClosePenalty(const OpenBracketInfo& top, const Item& item)
    -> std::optional<int32_t> {
  if (item.token.kind != Kind::CloseCurlyBrace || top.is_struct_brace ||
      item.token.line == top.line) {
    return 0;
  }
  if (item.token.line_indent < top.effective_header_indent) {
    return std::nullopt;
  }
  if (item.HasAll(Cue::FirstOnLine) &&
      item.token.line_indent != top.effective_header_indent) {
    return CostBraceIndentMismatchBase +
           CostBraceIndentMismatchPerColumn *
               std::abs(top.effective_header_indent - item.token.line_indent);
  }
  return 0;
}

// The correction that records where the synthetic opener `opener`, which the
// real closer `closer` has just matched, would be inserted.
static auto InsertOpenerCorrection(const OpenBracketInfo& opener,
                                   const MismatchedBracketToken& closer)
    -> BracketCorrection {
  return BracketCorrection{
      .diagnostic_kind = BracketDiagnosticKind::UnmatchedClosing,
      .diagnostic_token_index = closer.token_index,
      .fix_action = BracketFixAction::InsertBefore,
      .fix_token_index = opener.insertion_token_index,
      .fix_token_kind = ToTokenKind(opener.kind),
      .rule_name = opener.rule_name,
  };
}

// The cost of closing an open bracket at the end of the file or region.
static auto CostToCloseAtEnd(const OpenBracketInfo& open) -> int32_t {
  if (open.kind != Kind::OpenCurlyBrace) {
    return CostCloseParenAtEnd;
  }
  return open.is_struct_brace ? CostCloseStructAtEnd : CostCloseAtEnd;
}

// A layered beam search over one damaged region.
//
// Each layer holds the states that survive just before one item. Expanding a
// layer first proposes bracket insertions before the item — epsilon moves,
// which stay within the layer — and then advances over the item into the next
// layer, pruning each layer back to `MaxBeamWidth`. States in a layer that
// agree on their open-bracket stack merge into one node, which keeps every
// cheapest way of reaching it so that ties can be found afterwards.
namespace {

class RegionSearch {
 public:
  // `region_end_token` is the token directly after the region, where any
  // still-unclosed brackets are closed.
  RegionSearch(llvm::ArrayRef<Item> items, TokenIndex region_end_token)
      : items_(items), region_end_token_(region_end_token) {
    nodes_.reserve(256);
  }

  // Runs the search and appends the corrections for the cheapest repair to
  // `corrections`.
  auto Solve(llvm::SmallVectorImpl<BracketCorrection>& corrections) -> void;

 private:
  // Adds the state reached by `edge` at total cost `next_cost` to `layer`,
  // merging it into an equal state already there if there is one.
  // `next_item_idx` is the item the state sits before. If `worklist` is given,
  // any node that was added or became cheaper is appended to it, so that a
  // further epsilon move can be applied to it.
  auto AddToLayer(llvm::SmallVectorImpl<int32_t>& layer, int32_t next_item_idx,
                  OpenStackId next_stack, Kind closer_inserted,
                  int32_t next_cost, ParentEdge edge,
                  llvm::SmallVectorImpl<int32_t>* worklist = nullptr) -> void;

  // Merges a newly-found way of reaching the state already in `nodes_[idx]`: a
  // cheaper cost replaces what's recorded there, and an equal cost adds another
  // parent edge, so that every cheapest path stays available.
  auto MergeIntoNode(int32_t idx, int32_t cost, const ParentEdge& edge,
                     llvm::SmallVectorImpl<int32_t>* worklist) -> void;

  // Keeps a layer within the beam width by discarding the costliest states.
  auto PruneBeam(llvm::SmallVectorImpl<int32_t>& layer) -> void;

  // Adds to the current layer every state reachable by inserting brackets
  // directly before item `item_idx`.
  auto InsertBracketsBefore(int32_t item_idx) -> void;
  auto InsertSyntheticClosers(int32_t item_idx) -> void;
  auto InsertSyntheticOpeners(int32_t item_idx) -> void;

  // Advances the current layer over item `item_idx`, returning the layer that
  // results.
  auto AdvanceOverItem(int32_t item_idx) -> llvm::SmallVector<int32_t>;
  auto AdvanceNode(int32_t node_idx, int32_t item_idx,
                   llvm::SmallVectorImpl<int32_t>& next_layer) -> void;

  // Closes whatever is still open at the end of the region, returning the goal
  // nodes of the cheapest complete repairs.
  auto CloseAtRegionEnd() -> llvm::SmallVector<int32_t>;

  llvm::ArrayRef<Item> items_;
  TokenIndex region_end_token_;

  // Every state the search has reached, in creation order; a layer names its
  // states by their index here. Nodes are never removed, so pruning a layer
  // leaves its states behind, unreachable.
  llvm::SmallVector<BeamNode, 0> nodes_;
  // The cost of the cheapest complete repair found so far, which bounds the
  // cost of any state still worth expanding.
  int32_t min_goal_cost_ = std::numeric_limits<int32_t>::max();
  llvm::SmallVector<int32_t> current_layer_;
  // The states in the layer currently being built, keyed on the state itself so
  // that reaching a state already in the layer merges into it, and giving that
  // state's index in `nodes_`. Kept across layers only to reuse its allocation.
  Map<StateKey, int32_t> layer_dedup_;
  // The open-bracket stacks every state in `nodes_` refers to.
  OpenStackStore stacks_;
};

}  // namespace

auto RegionSearch::AddToLayer(llvm::SmallVectorImpl<int32_t>& layer,
                              int32_t next_item_idx, OpenStackId next_stack,
                              Kind closer_inserted, int32_t next_cost,
                              ParentEdge edge,
                              llvm::SmallVectorImpl<int32_t>* worklist)
    -> void {
  if (next_cost > min_goal_cost_) {
    return;
  }
  StateKey key = {.stack = next_stack, .closer_inserted = closer_inserted};
  if (int32_t* existing = layer_dedup_[key]) {
    MergeIntoNode(*existing, next_cost, edge, worklist);
    return;
  }
  auto new_idx = static_cast<int32_t>(nodes_.size());
  nodes_.push_back(BeamNode{
      .item_index = next_item_idx,
      .stack = next_stack,
      .cost = next_cost,
      .closer_inserted = closer_inserted,
      .parent_edges = {edge},
  });
  layer.push_back(new_idx);
  layer_dedup_.Insert(key, new_idx);
  if (worklist) {
    worklist->push_back(new_idx);
  }
}

auto RegionSearch::MergeIntoNode(int32_t idx, int32_t cost,
                                 const ParentEdge& edge,
                                 llvm::SmallVectorImpl<int32_t>* worklist)
    -> void {
  BeamNode& node = nodes_[idx];
  if (cost < node.cost) {
    node.cost = cost;
    node.parent_edges.clear();
    node.parent_edges.push_back(edge);
    if (worklist) {
      worklist->push_back(idx);
    }
  } else if (cost == node.cost &&
             llvm::none_of(node.parent_edges, [&](const ParentEdge& e) {
               return EdgesEqual(e, edge);
             })) {
    node.parent_edges.push_back(edge);
  }
}

auto RegionSearch::PruneBeam(llvm::SmallVectorImpl<int32_t>& layer) -> void {
  if (layer.size() > MaxBeamWidth) {
    llvm::stable_sort(layer, [&](int32_t a, int32_t b) {
      return nodes_[a].cost < nodes_[b].cost;
    });
    layer.resize(MaxBeamWidth);
  }
}

auto RegionSearch::InsertBracketsBefore(int32_t item_idx) -> void {
  // Seed the dedup table with the states already in the layer, so that an
  // insertion reaching one of them merges into it instead of duplicating it.
  for (int32_t idx : current_layer_) {
    layer_dedup_.Insert(
        StateKey{.stack = nodes_[idx].stack,
                 .closer_inserted = nodes_[idx].closer_inserted},
        idx);
  }
  InsertSyntheticClosers(item_idx);
  InsertSyntheticOpeners(item_idx);
  layer_dedup_.Clear();
  PruneBeam(current_layer_);
}

auto RegionSearch::InsertSyntheticClosers(int32_t item_idx) -> void {
  const Item& item = items_[item_idx];
  // A worklist rather than one pass over the layer, so that several groups can
  // be closed at the same point.
  llvm::SmallVector<int32_t> worklist = current_layer_;
  for (size_t head = 0; head < worklist.size(); ++head) {
    const SearchState current = Snapshot(nodes_[worklist[head]]);
    if (current.cost > min_goal_cost_ || stacks_.IsEmpty(current.stack)) {
      continue;
    }
    const auto& top = stacks_.Top(current.stack);
    // Synthetic openers exist only to consume real closers; closing one
    // synthetically would insert a pointless empty pair.
    if (top.is_synthetic) {
      continue;
    }
    if (item.token.kind == MatchingClosingKind(top.kind) &&
        DirectMatchPreferred(top, item)) {
      continue;
    }
    llvm::StringLiteral rule_name = "";
    auto cost =
        ClassifyCloserInsertion(top, item, stacks_, current.stack, rule_name);
    if (!cost) {
      continue;
    }

    const OpenBracketInfo& popped = top;
    OpenStackId next_stack = stacks_.Pop(current.stack);
    auto closer_kind = MatchingClosingKind(popped.kind);
    AddToLayer(
        current_layer_, item_idx, next_stack, closer_kind, current.cost + *cost,
        ParentEdge{
            .parent_node_index = worklist[head],
            .correction =
                BracketCorrection{
                    .diagnostic_kind = BracketDiagnosticKind::UnmatchedOpening,
                    .diagnostic_token_index = popped.token_index,
                    .fix_action = BracketFixAction::InsertBefore,
                    .fix_token_index = item.token.token_index,
                    .fix_token_kind = ToTokenKind(closer_kind),
                    .rule_name = rule_name,
                },
            .has_correction = true,
        },
        &worklist);
  }
}

auto RegionSearch::InsertSyntheticOpeners(int32_t item_idx) -> void {
  const Item& item = items_[item_idx];
  // Iterate only over the states present after the closer phase, so that
  // synthetic openers don't chain onto each other. The bound is taken before
  // the loop because the loop appends to the layer.
  for (size_t idx : llvm::seq<size_t>(0, current_layer_.size())) {
    int32_t node_idx = current_layer_[idx];
    const SearchState current = Snapshot(nodes_[node_idx]);
    if (current.cost > min_goal_cost_ ||
        static_cast<size_t>(stacks_.Depth(current.stack)) >=
            MaxSearchStackDepth) {
      continue;
    }
    for (auto [open_kind, is_struct_brace] : SyntheticOpeners) {
      llvm::StringLiteral rule_name = "";
      auto cost =
          ClassifyOpenerInsertion(open_kind, is_struct_brace, item, rule_name);
      if (!cost) {
        continue;
      }
      OpenStackId next_stack = stacks_.Push(
          current.stack,
          OpenBracketInfo{
              OpenBracketKey{
                  .insertion_token_index = item.token.token_index,
                  .line = item.token.line,
                  .line_indent = item.token.line_indent,
                  .effective_header_indent = item.effective_header_indent,
                  .kind = open_kind,
                  .is_synthetic = true,
                  .is_struct_brace = is_struct_brace,
              },
              rule_name,
          });
      AddToLayer(current_layer_, item_idx, next_stack, current.closer_inserted,
                 current.cost + *cost,
                 ParentEdge{.parent_node_index = node_idx});
    }
  }
}

auto RegionSearch::AdvanceOverItem(int32_t item_idx)
    -> llvm::SmallVector<int32_t> {
  llvm::SmallVector<int32_t> next_layer;
  for (int32_t node_idx : current_layer_) {
    AdvanceNode(node_idx, item_idx, next_layer);
  }
  layer_dedup_.Clear();
  PruneBeam(next_layer);
  return next_layer;
}

auto RegionSearch::AdvanceNode(int32_t node_idx, int32_t item_idx,
                               llvm::SmallVectorImpl<int32_t>& next_layer)
    -> void {
  const SearchState current = Snapshot(nodes_[node_idx]);
  if (current.cost > min_goal_cost_) {
    return;
  }
  const Item& item = items_[item_idx];
  auto kind = item.token.kind;

  auto advance = [&](OpenStackId next_stack, int32_t add_cost,
                     BracketCorrection correction = {},
                     bool has_correction = false) {
    AddToLayer(next_layer, item_idx + 1, next_stack, Kind::Other,
               current.cost + add_cost,
               ParentEdge{.parent_node_index = node_idx,
                          .correction = correction,
                          .has_correction = has_correction});
  };

  // What it costs to advance over this item in this state, per the
  // `AdvanceRules` table.
  int32_t penalty = AdvancePenalty(stacks_, current, item);

  if (item.HasAll(Cue::CollapsedBlock)) {
    advance(current.stack, penalty);
    return;
  }

  if (IsOpeningBracket(kind)) {
    // Advance, pushing the opener onto the stack.
    if (static_cast<size_t>(stacks_.Depth(current.stack)) <
        MaxSearchStackDepth) {
      advance(stacks_.Push(current.stack, RealOpener(item)), penalty);
    }
    // Advance without pushing: replace the unmatched opener with an error
    // token.
    advance(
        current.stack, CostReplaceOpening,
        ReplaceWithError(item.token, BracketDiagnosticKind::UnmatchedOpening,
                         "Adv_ReplaceOpener"),
        /*has_correction=*/true);
    return;
  }

  if (IsClosingBracket(kind)) {
    // Advance, popping the opener this closer matches.
    if (!stacks_.IsEmpty(current.stack) &&
        stacks_.Top(current.stack).kind == MatchingOpeningKind(kind)) {
      const OpenBracketInfo& popped = stacks_.Top(current.stack);
      if (auto match_cost = MatchClosePenalty(popped, item)) {
        // Matching a synthetic opener finally pins down where it goes, so this
        // is where it becomes a correction.
        advance(stacks_.Pop(current.stack), penalty + *match_cost,
                popped.is_synthetic ? InsertOpenerCorrection(popped, item.token)
                                    : BracketCorrection{},
                popped.is_synthetic);
      }
    }
    // Advance without matching: replace the unmatched closer with an error
    // token.
    advance(
        current.stack, CostReplaceClosing,
        ReplaceWithError(item.token, BracketDiagnosticKind::UnmatchedClosing,
                         "Adv_ReplaceCloser"),
        /*has_correction=*/true);
    return;
  }

  // Any other token.
  advance(current.stack, penalty);
}

auto RegionSearch::CloseAtRegionEnd() -> llvm::SmallVector<int32_t> {
  llvm::SmallVector<int32_t> goal_node_indices;
  for (int32_t node_idx : current_layer_) {
    const SearchState current = Snapshot(nodes_[node_idx]);
    if (current.cost > min_goal_cost_) {
      continue;
    }
    // A synthetic opener that never matched a real closer is a meaningless
    // insertion; reject such states rather than dropping it silently.
    if (stacks_.HasSynthetic(current.stack)) {
      continue;
    }
    // Close what's left innermost-first, one node per closer, so that each is
    // reconstructed as its own correction.
    int32_t finish_cost = current.cost;
    int32_t parent = node_idx;
    for (OpenStackId open_stack = current.stack; !stacks_.IsEmpty(open_stack);
         open_stack = stacks_.Pop(open_stack)) {
      const OpenBracketInfo& open = stacks_.Top(open_stack);
      finish_cost += CostToCloseAtEnd(open);
      nodes_.push_back(BeamNode{
          .item_index = static_cast<int32_t>(items_.size()),
          .stack = EmptyOpenStack,
          .cost = finish_cost,
          .parent_edges = {{
              .parent_node_index = parent,
              .correction =
                  BracketCorrection{
                      .diagnostic_kind =
                          BracketDiagnosticKind::UnmatchedOpening,
                      .diagnostic_token_index = open.token_index,
                      .fix_action = BracketFixAction::InsertBefore,
                      .fix_token_index = region_end_token_,
                      .fix_token_kind =
                          ToTokenKind(MatchingClosingKind(open.kind)),
                      .rule_name = "Close_RegionEnd"},
              .has_correction = true,
          }},
      });
      parent = static_cast<int32_t>(nodes_.size()) - 1;
    }

    if (finish_cost < min_goal_cost_) {
      min_goal_cost_ = finish_cost;
      goal_node_indices.clear();
    }
    if (finish_cost == min_goal_cost_) {
      goal_node_indices.push_back(parent);
    }
  }
  return goal_node_indices;
}

auto RegionSearch::Solve(llvm::SmallVectorImpl<BracketCorrection>& corrections)
    -> void {
  nodes_.push_back(BeamNode{
      .item_index = 0,
      .stack = EmptyOpenStack,
      .cost = 0,
      .parent_edges = {},
  });
  current_layer_ = {0};

  for (auto [item_index, item] : llvm::enumerate(items_)) {
    auto i = static_cast<int32_t>(item_index);
    // Nothing is inserted before the end of the file; brackets still open there
    // are closed by `CloseAtRegionEnd` instead.
    if (item.token.kind != Kind::FileEnd) {
      InsertBracketsBefore(i);
    }
    current_layer_ = AdvanceOverItem(i);
  }

  llvm::SmallVector<int32_t> goal_node_indices = CloseAtRegionEnd();
  if (goal_node_indices.empty()) {
    SolveNaive(items_, corrections);
    return;
  }
  ReconstructCorrections(nodes_, goal_node_indices, items_, region_end_token_,
                         corrections);
}

// Solve a damaged region using layered beam search with tie detection.
// `region_end_token` is the token directly after the region, where any
// still-unclosed brackets are closed.
static auto SolveRegionCostBased(
    llvm::ArrayRef<Item> items, TokenIndex region_end_token,
    llvm::SmallVectorImpl<BracketCorrection>& corrections) -> void {
  if (items.size() > static_cast<size_t>(MaxRegionItemsForSearch)) {
    SolveNaive(items, corrections);
    return;
  }
  RegionSearch(items, region_end_token).Solve(corrections);
}

// The passes below run in order over the whole token sequence, each consuming
// the results of the ones before it: analyze the tokens, decide which matched
// bracket pairs can be trusted and collapsed, build the item sequence from
// that, then split it into regions and search the damaged ones.

// Finds the stack position of the opener in `open_stack` that the closing
// bracket `tokens[closer_index]` should pair with, or -1 if none is plausible.
// A `}` matches a `{` only when they're on the same line, the `{` is a struct
// brace, or their line indentation agrees.
static auto FindMatchingOpener(llvm::ArrayRef<MismatchedBracketToken> tokens,
                               llvm::ArrayRef<int32_t> open_stack,
                               int32_t closer_index) -> int32_t {
  const auto& closer = tokens[closer_index];
  auto num_open = static_cast<int32_t>(open_stack.size());
  for (int32_t s : llvm::reverse(llvm::seq(0, num_open))) {
    const auto& opener = tokens[open_stack[s]];
    if (closer.kind != Kind::CloseCurlyBrace) {
      if (opener.kind == MatchingOpeningKind(closer.kind)) {
        return s;
      }
    } else if (opener.kind == Kind::OpenCurlyBrace &&
               (opener.line == closer.line || opener.is_struct_brace ||
                opener.line_indent == closer.line_indent)) {
      return s;
    }
  }
  return -1;
}

// Pairs up brackets by a stack walk, returning for each token the index of the
// bracket it pairs with, or -1 if it has none. A closer that doesn't match the
// top of the stack pops through to a plausible match if one exists (leaving the
// popped brackets unmatched), and is otherwise left unmatched without
// disturbing the stack.
static auto MatchBracketPairs(llvm::ArrayRef<MismatchedBracketToken> tokens)
    -> llvm::SmallVector<int32_t> {
  auto num_tokens = static_cast<int32_t>(tokens.size());
  llvm::SmallVector<int32_t> match_partner(num_tokens, -1);
  llvm::SmallVector<int32_t> open_stack;
  for (int32_t i : llvm::seq(0, num_tokens)) {
    auto kind = tokens[i].kind;
    if (IsOpeningBracket(kind)) {
      open_stack.push_back(i);
    } else if (IsClosingBracket(kind)) {
      int32_t match_s = FindMatchingOpener(tokens, open_stack, i);
      if (match_s != -1) {
        match_partner[open_stack[match_s]] = i;
        match_partner[i] = open_stack[match_s];
        open_stack.resize(match_s);
      }
    }
  }
  return match_partner;
}

namespace {

// The statement segments of the token sequence, which are separated by `;`,
// `{`, and `}`.
struct Segments {
  // The id of each token's segment. Segments are numbered in order.
  llvm::SmallVector<int32_t> id;
  // The index of the first token of each token's segment.
  llvm::SmallVector<int32_t> first;
};

// The indexes of the unmatched `(`, `[`, `)`, and `]` tokens, each list in
// increasing order so it can be binary-searched.
struct UnmatchedGroupBrackets {
  llvm::SmallVector<int32_t> open_parens;
  llvm::SmallVector<int32_t> open_squares;
  llvm::SmallVector<int32_t> close_parens;
  llvm::SmallVector<int32_t> close_squares;

  // The openers and closers of the bracket kind that `open_kind` opens.
  auto OpenersOfKind(Kind open_kind) const -> llvm::ArrayRef<int32_t> {
    return open_kind == Kind::OpenParen ? open_parens : open_squares;
  }
  auto ClosersOfKind(Kind open_kind) const -> llvm::ArrayRef<int32_t> {
    return open_kind == Kind::OpenParen ? close_parens : close_squares;
  }
};

// Everything the later passes need to know about the token sequence as a whole,
// computed once up front by `AnalyzeTokens`. The vectors are all indexed by
// token index.
struct TokenAnalysis {
  // The index of the bracket each token pairs with, or -1. See
  // `MatchBracketPairs`.
  llvm::SmallVector<int32_t> match_partner;
  Segments segments;
  UnmatchedGroupBrackets unmatched;
  // See `ComputeAssociatedLineIndents`.
  llvm::SmallVector<int32_t> effective_header_indent;
  llvm::BitVector is_first_on_line;
  // See `ComputeFollowsStatementHeader` and `ComputeHeaderHasOpenCurlyBrace`.
  // The latter is only computed where the former holds.
  llvm::BitVector follows_statement_header;
  llvm::BitVector header_has_open_curly_brace;
};

}  // namespace

static auto ComputeSegments(llvm::ArrayRef<MismatchedBracketToken> tokens)
    -> Segments {
  auto num_tokens = static_cast<int32_t>(tokens.size());
  Segments segments = {.id = llvm::SmallVector<int32_t>(num_tokens, 0),
                       .first = llvm::SmallVector<int32_t>(num_tokens, 0)};
  for (int32_t i : llvm::seq(1, num_tokens)) {
    auto prev_kind = tokens[i - 1].kind;
    bool new_seg = prev_kind == Kind::Semi ||
                   prev_kind == Kind::OpenCurlyBrace ||
                   prev_kind == Kind::CloseCurlyBrace;
    segments.id[i] = segments.id[i - 1] + (new_seg ? 1 : 0);
    segments.first[i] = new_seg ? i : segments.first[i - 1];
  }
  return segments;
}

static auto FindUnmatchedGroupBrackets(
    llvm::ArrayRef<MismatchedBracketToken> tokens,
    llvm::ArrayRef<int32_t> match_partner) -> UnmatchedGroupBrackets {
  UnmatchedGroupBrackets unmatched;
  for (auto [i, token] : llvm::enumerate(tokens)) {
    if (match_partner[i] != -1) {
      continue;
    }
    auto index = static_cast<int32_t>(i);
    switch (token.kind) {
      case Kind::OpenParen:
        unmatched.open_parens.push_back(index);
        break;
      case Kind::OpenSquareBracket:
        unmatched.open_squares.push_back(index);
        break;
      case Kind::CloseParen:
        unmatched.close_parens.push_back(index);
        break;
      case Kind::CloseSquareBracket:
        unmatched.close_squares.push_back(index);
        break;
      default:
        break;
    }
  }
  return unmatched;
}

static auto AnalyzeTokens(llvm::ArrayRef<MismatchedBracketToken> tokens)
    -> TokenAnalysis {
  auto num_tokens = static_cast<int32_t>(tokens.size());
  auto match_partner = MatchBracketPairs(tokens);
  auto segments = ComputeSegments(tokens);
  auto unmatched = FindUnmatchedGroupBrackets(tokens, match_partner);

  auto effective_header_indent =
      ComputeAssociatedLineIndents(tokens, match_partner);
  llvm::BitVector is_first_on_line(num_tokens);
  llvm::BitVector follows_statement_header(num_tokens);
  llvm::BitVector header_has_open_curly_brace(num_tokens);
  for (int32_t i : llvm::seq(0, num_tokens)) {
    is_first_on_line[i] = (i == 0 || tokens[i].line != tokens[i - 1].line);
    follows_statement_header[i] =
        ComputeFollowsStatementHeader(tokens, match_partner, i);
    if (follows_statement_header[i]) {
      header_has_open_curly_brace[i] =
          ComputeHeaderHasOpenCurlyBrace(tokens, match_partner, i);
    }
  }

  return TokenAnalysis{
      .match_partner = std::move(match_partner),
      .segments = std::move(segments),
      .unmatched = std::move(unmatched),
      .effective_header_indent = std::move(effective_header_indent),
      .is_first_on_line = std::move(is_first_on_line),
      .follows_statement_header = std::move(follows_statement_header),
      .header_has_open_curly_brace = std::move(header_has_open_curly_brace),
  };
}

// Whether the sorted list `list` contains an element in [lo, hi].
static auto ContainsInRange(llvm::ArrayRef<int32_t> list, int32_t lo,
                            int32_t hi) -> bool {
  const auto* it = std::lower_bound(list.begin(), list.end(), lo);
  return it != list.end() && *it <= hi;
}

// Whether a matched `{`...`}` pair is trustworthy, judging by how the `}` lines
// up and by the separators directly inside the pair.
static auto BracePairIsClean(llvm::ArrayRef<MismatchedBracketToken> tokens,
                             const TokenAnalysis& analysis, int32_t open_idx,
                             int32_t close_idx) -> bool {
  const auto& open = tokens[open_idx];
  int32_t header_indent = analysis.effective_header_indent[open_idx];
  if (open.line != tokens[close_idx].line) {
    if (!open.is_struct_brace &&
        (!open.is_at_end_of_line ||
         header_indent != tokens[close_idx].line_indent)) {
      return false;
    }
    if (open.is_struct_brace && tokens[close_idx].line_indent < header_indent) {
      return false;
    }
  }
  // A `;` directly inside a struct brace, or a `,` directly inside a scope
  // brace, is illegal; the brace pairing has likely captured too much.
  auto bad_kind = open.is_struct_brace ? Kind::Semi : Kind::Comma;
  int32_t depth = 0;
  for (int32_t j : llvm::seq(open_idx + 1, close_idx)) {
    if (IsOpeningBracket(tokens[j].kind)) {
      ++depth;
    } else if (IsClosingBracket(tokens[j].kind)) {
      --depth;
    } else if (tokens[j].kind == bad_kind && depth == 0) {
      return false;
    }
  }
  return true;
}

// Whether a matched `(`...`)` or `[`...`]` pair is trustworthy. An unmatched
// opener of the same kind earlier in the same statement segment could really
// own our closer, and an unmatched closer of the same kind later in the same
// segment could really own our opener; both make the pairing suspect.
static auto GroupPairIsClean(llvm::ArrayRef<MismatchedBracketToken> tokens,
                             const TokenAnalysis& analysis, int32_t open_idx,
                             int32_t close_idx) -> bool {
  auto kind = tokens[open_idx].kind;
  if (ContainsInRange(analysis.unmatched.OpenersOfKind(kind),
                      analysis.segments.first[open_idx], open_idx - 1)) {
    return false;
  }
  // A scan for an unmatched closer would be bounded by the end of the closer's
  // segment, so it's enough to check whether the first one after `close_idx` is
  // still in that segment.
  llvm::ArrayRef<int32_t> closers = analysis.unmatched.ClosersOfKind(kind);
  const auto* it = std::upper_bound(closers.begin(), closers.end(), close_idx);
  return it == closers.end() ||
         analysis.segments.id[*it] != analysis.segments.id[close_idx];
}

// Whether everything between a matched pair is itself safe to collapse: every
// bracket inside pairs up within the pair and is clean, and for a scope brace,
// no line inside is dedented to or past the header.
static auto PairInteriorIsClean(llvm::ArrayRef<MismatchedBracketToken> tokens,
                                const TokenAnalysis& analysis,
                                const llvm::BitVector& is_clean_range,
                                int32_t open_idx, int32_t close_idx) -> bool {
  const auto& open = tokens[open_idx];
  for (int32_t j : llvm::seq(open_idx + 1, close_idx)) {
    int32_t partner = analysis.match_partner[j];
    if (partner == -1) {
      // An unmatched bracket inside means the pair can't be trusted.
      if (IsOpeningBracket(tokens[j].kind) ||
          IsClosingBracket(tokens[j].kind)) {
        return false;
      }
    } else if (partner < open_idx || partner > close_idx) {
      return false;
    }
    if (IsOpeningBracket(tokens[j].kind) && !is_clean_range[j]) {
      return false;
    }
    if (open.kind == Kind::OpenCurlyBrace && !open.is_struct_brace &&
        analysis.is_first_on_line[j] &&
        tokens[j].line != tokens[close_idx].line &&
        tokens[j].line_indent <= analysis.effective_header_indent[open_idx]) {
      return false;
    }
  }
  return true;
}

// Marks the matched pairs that can be trusted, so that the search can collapse
// each into a single item instead of reconsidering brackets that clearly pair
// up. Pairs are visited in reverse order, so an inner pair is decided before
// the pairs enclosing it.
//
// Note: an illegal leaf adjacency inside a *matched* pair is treated as invalid
// user code, not a bracket error; the pair is still trusted and collapsed. The
// adjacency cue only guides where to insert brackets in regions that are
// already unbalanced.
static auto MarkCleanRanges(llvm::ArrayRef<MismatchedBracketToken> tokens,
                            const TokenAnalysis& analysis) -> llvm::BitVector {
  auto num_tokens = static_cast<int32_t>(tokens.size());
  llvm::BitVector is_clean_range(num_tokens);
  for (int32_t i : llvm::reverse(llvm::seq(0, num_tokens))) {
    // Only consider a pair from its opener; this also skips the unmatched
    // tokens, whose partner is -1.
    int32_t close_idx = analysis.match_partner[i];
    if (close_idx <= i) {
      continue;
    }
    bool clean = tokens[i].kind == Kind::OpenCurlyBrace
                     ? BracePairIsClean(tokens, analysis, i, close_idx)
                     : GroupPairIsClean(tokens, analysis, i, close_idx);
    is_clean_range[i] =
        clean &&
        PairInteriorIsClean(tokens, analysis, is_clean_range, i, close_idx);
  }
  return is_clean_range;
}

// Builds the item sequence the search runs over: one item per token, except
// that each clean matched pair collapses into a single item spanning it.
static auto BuildItems(llvm::ArrayRef<MismatchedBracketToken> tokens,
                       const TokenAnalysis& analysis,
                       const llvm::BitVector& is_clean_range)
    -> llvm::SmallVector<Item> {
  auto num_tokens = static_cast<int32_t>(tokens.size());
  llvm::SmallVector<Item> items;
  auto make_item = [&](int32_t start, int32_t end, bool collapsed,
                       bool has_scope) {
    // Cues the surrounding analysis has already determined; the rest follow
    // from the tokens themselves.
    CueSet context_cues =
        CueIf(collapsed, Cue::CollapsedBlock) |
        CueIf(has_scope, Cue::ContainsScopeBrace) |
        CueIf(analysis.is_first_on_line[start], Cue::FirstOnLine) |
        CueIf(analysis.follows_statement_header[start],
              Cue::FollowsStatementHeader) |
        CueIf(analysis.header_has_open_curly_brace[start],
              Cue::HeaderHasOpenCurly);
    items.push_back(Item{
        .token_start_index = start,
        .token_end_index = end,
        .token = tokens[start],
        .effective_header_indent = analysis.effective_header_indent[start],
        .cues = ComputeItemCues(
            tokens[start], start > 0 ? &tokens[start - 1] : nullptr,
            items.empty() ? nullptr : &items.back(), context_cues),
    });
  };
  for (int32_t i = 0; i < num_tokens;) {
    if (!is_clean_range[i]) {
      make_item(i, i, /*collapsed=*/false,
                tokens[i].kind == Kind::OpenCurlyBrace ||
                    tokens[i].kind == Kind::CloseCurlyBrace);
      ++i;
      continue;
    }
    int32_t close_idx = analysis.match_partner[i];
    bool has_scope = llvm::any_of(llvm::seq(i, close_idx + 1), [&](int32_t j) {
      return tokens[j].kind == Kind::OpenCurlyBrace &&
             !tokens[j].is_struct_brace;
    });
    make_item(i, close_idx, /*collapsed=*/true, has_scope);
    i = close_idx + 1;
  }
  return items;
}

// Finds the item indexes where each independently-solved region starts,
// beginning with 0 and ending with the number of items, so that consecutive
// boundaries delimit the regions. A region ends at a top-level declaration
// boundary: a statement introducer at zero indentation whose predecessor ended
// a statement. This bounds how far a single mistake can smear, and gives
// unclosed brackets a natural place to be closed (the region end).
static auto FindRegionBoundaries(llvm::ArrayRef<MismatchedBracketToken> tokens,
                                 llvm::ArrayRef<Item> items)
    -> llvm::SmallVector<int32_t> {
  llvm::SmallVector<int32_t> boundaries = {0};
  for (auto [item_index, item] : llvm::enumerate(items)) {
    auto i = static_cast<int32_t>(item_index);
    // Note: line_indent is a 1-based column number, so top-level tokens have
    // line_indent 1.
    if (i == 0 || item.HasAll(Cue::CollapsedBlock) ||
        item.token.kind != Kind::StatementIntroducer ||
        item.token.is_else_keyword || item.token.line_indent > 1 ||
        !item.HasAll(Cue::FirstOnLine)) {
      continue;
    }
    auto prev_end_kind = tokens[items[i - 1].token_end_index].kind;
    if ((prev_end_kind == Kind::Semi ||
         prev_end_kind == Kind::CloseCurlyBrace) &&
        boundaries.back() != i) {
      boundaries.push_back(i);
    }
  }
  if (boundaries.back() != static_cast<int32_t>(items.size())) {
    boundaries.push_back(static_cast<int32_t>(items.size()));
  }
  return boundaries;
}

// Whether a region's loose (non-collapsed) brackets already form a balanced,
// well-nested sequence. Such a region needs no solving: it has no unmatched
// bracket, so the search would simply match everything and emit no corrections.
// Skipping it avoids running the beam search over the many regions whose
// matched pairs merely weren't collapsed.
static auto RegionIsBalanced(llvm::ArrayRef<Item> region) -> bool {
  llvm::SmallVector<Kind> open_kinds;
  for (const Item& item : region) {
    if (item.HasAll(Cue::CollapsedBlock)) {
      continue;
    }
    auto kind = item.token.kind;
    if (IsOpeningBracket(kind)) {
      open_kinds.push_back(kind);
    } else if (IsClosingBracket(kind)) {
      if (open_kinds.empty() ||
          MatchingClosingKind(open_kinds.back()) != kind) {
        return false;
      }
      open_kinds.pop_back();
    }
  }
  return open_kinds.empty();
}

auto FixMismatchedBrackets(llvm::ArrayRef<MismatchedBracketToken> tokens)
    -> llvm::SmallVector<BracketCorrection> {
  llvm::SmallVector<BracketCorrection> corrections;
  if (tokens.empty()) {
    return corrections;
  }

  TokenAnalysis analysis = AnalyzeTokens(tokens);
  llvm::BitVector is_clean_range = MarkCleanRanges(tokens, analysis);
  llvm::SmallVector<Item> items = BuildItems(tokens, analysis, is_clean_range);
  llvm::SmallVector<int32_t> region_boundaries =
      FindRegionBoundaries(tokens, items);

  // Each region runs between consecutive boundaries.
  for (auto [start, end] :
       llvm::zip(region_boundaries, llvm::drop_begin(region_boundaries))) {
    if (start >= end) {
      continue;
    }
    auto region = llvm::ArrayRef<Item>(items).slice(start, end - start);
    if (RegionIsBalanced(region)) {
      continue;
    }
    // Any bracket still open at the end of the region is closed before the
    // token the next region starts at, or before the final `FileEnd`.
    TokenIndex region_end_token = end < static_cast<int32_t>(items.size())
                                      ? items[end].token.token_index
                                      : tokens.back().token_index;
    SolveRegionCostBased(region, region_end_token, corrections);
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

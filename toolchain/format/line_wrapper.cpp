// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/line_wrapper.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <queue>
#include <utility>

#include "common/hashing.h"
#include "common/map.h"
#include "common/set.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/format/token_info.h"
#include "toolchain/lex/token_kind.h"

namespace Carbon::Format {

// An upper bound on the number of search states generated (roughly twice the
// number explored) before giving up and leaving the line unwrapped. Ordinary
// lines generate far fewer; this only bounds the cost of pathological inputs
// (very long, deeply nested lines).
constexpr int MaxStatesGenerated = 100'000;

// Whether the layout of a member-access chain is still open, committed to
// breaking every call/subscript boundary, or committed to staying packed. See
// `ChainState`.
enum class ChainBreak : int8_t { Undecided, Broken, Unbroken };

namespace {

// The live layout state of one member-access call chain. clang-format formats a
// chain all-or-nothing: either it fits packed on its line, or it breaks before
// *every* member access that follows a call/subscript (the "fluent"/builder
// shape). The first such break commits the chain, and the rest must match.
struct ChainState {
  // The chain's identity: its receiver-root token index (`member_chain_id`).
  int key;
  // The column member-access breaks in this chain indent to: the receiver's
  // start column plus the continuation indent (clang-format anchors a chain's
  // continuations under its root, not under the statement).
  int anchor;
  // Whether the all-or-nothing break decision has been made yet, and which way.
  ChainBreak decision;
};

// Per-token, line-local data about member-access chains, precomputed once
// before the search. Indexed by position within the line.
struct ChainInfo {
  // For a token that is the receiver root of a chain, the chain's key; else -1.
  // Placing such a token opens a `ChainState`.
  llvm::SmallVector<int> root_key;
  // For a member-access `.`/`->` token, whether it follows a `)`/`]` and so is
  // a fluent break point subject to the all-or-nothing coupling.
  llvm::SmallVector<bool> is_fluent;
  // For each chain key, the line position of its last member-access token,
  // after which the chain's state is no longer needed and is dropped.
  Map<int, int> last_member_pos;
  // For each chain key, whether it is a builder chain: one with at least one
  // fluent break point. In a builder chain, only fluent points break (the
  // first segment and any field accesses stay attached, clang-format's
  // builder-call shape); a chain with none is a plain field chain that breaks
  // by the usual minimum-break.
  Map<int, bool> is_builder;
};

}  // namespace

// Computes the per-token chain data for `line`; see `ChainInfo`.
static auto ComputeChainInfo(const Lex::TokenizedBuffer& tokens,
                             const TokenInfoStore& token_infos,
                             llvm::ArrayRef<Lex::TokenIndex> line)
    -> ChainInfo {
  int n = line.size();
  ChainInfo info;
  info.root_key.assign(n, -1);
  info.is_fluent.assign(n, false);

  // Map each token's index to its line position, to locate chain receiver
  // roots.
  Map<int, int> tok_to_pos;
  for (int i = 0; i < n; ++i) {
    tok_to_pos.Update(line[i].index, i);
  }

  for (int i = 0; i < n; ++i) {
    int key = token_infos.Get(line[i]).member_chain_id;
    if (key < 0) {
      continue;
    }
    info.last_member_pos.Update(key, i);
    bool fluent = i > 0 && tokens.GetKind(line[i - 1])
                               .IsOneOf({Lex::TokenKind::CloseParen,
                                         Lex::TokenKind::CloseSquareBracket});
    info.is_fluent[i] = fluent;
    if (fluent) {
      info.is_builder.Update(key, true);
    } else {
      info.is_builder.Insert(key, false);
    }
    // Mark the receiver root's position (the first time its chain is seen).
    if (int* root_pos = tok_to_pos[key]) {
      info.root_key[*root_pos] = key;
    }
  }
  return info;
}

// Updates the continuation-indent stack for the scopes that a token opens and
// closes, given that its text runs from `start_column` to `end_column`, with
// `kind` the token's kind and `token` the formatter's annotations for it.
// There are two kinds of scope, both stored as the column their continuation
// lines indent to:
//
//   - Operand-alignment scopes (the analog of clang-format's fake parentheses)
//     open at the first token of a binary-operator's operand span and align the
//     operands under it, but never tighter than the enclosing scope, so an
//     operator at the very start of a line indents by the continuation width
//     rather than aligning at column 0 (`AlignOperands = Align`).
//   - A real bracket aligns its contents just after the bracket
//     (`AlignAfterOpenBracket = Align`).
//
// The bottom entry (the statement-level continuation indent) is never popped,
// so unbalanced scopes from malformed input can't empty the stack.
static auto ApplyTokenScopes(Lex::TokenKind kind, const TokenInfo& token,
                             int start_column, int end_column,
                             llvm::SmallVectorImpl<int>& stack) -> void {
  for (int i = 0; i < token.open_scopes; ++i) {
    stack.push_back(std::max(start_column, stack.back()));
  }
  if (kind.IsOneOf({Lex::TokenKind::OpenParen,
                    Lex::TokenKind::OpenSquareBracket,
                    Lex::TokenKind::OpenCurlyBrace})) {
    // The anchor is the first element's column: just past `(` and `[`, whose
    // interiors have no padding, and one further for `{`, whose interior is
    // padded by a space.
    stack.push_back(end_column +
                    (kind == Lex::TokenKind::OpenCurlyBrace ? 1 : 0));
  } else if (kind.IsOneOf({Lex::TokenKind::CloseParen,
                           Lex::TokenKind::CloseSquareBracket,
                           Lex::TokenKind::CloseCurlyBrace}) &&
             stack.size() > 1) {
    stack.pop_back();
  }
  for (int i = 0; i < token.close_scopes && stack.size() > 1; ++i) {
    stack.pop_back();
  }
}

namespace {

// One state in the shortest-path search: a prefix of the line has been laid
// out, and `index` is the next token to place.
struct State {
  // The next token to place, as an index into the line.
  int index;
  // The column just past the last placed token, where the next token's leading
  // space would begin.
  int column;
  // The continuation-indent anchor for each currently-open bracket level, with
  // a bottom entry for the statement level. `back()` is the current level.
  llvm::SmallVector<int, 8> stack;
  // The currently-open member-access chains (innermost last). Tracked
  // separately from `stack` because a chain's anchor persists across the `()`
  // calls within it, which push and pop bracket levels.
  llvm::SmallVector<ChainState, 4> chains;
  // The state this was reached from, as an index into the node pool, or -1 for
  // the start state.
  int parent;
  // The break decision that produced this state: the indent column if a break
  // precedes token `index - 1`, or -1 if not.
  int newline_indent;
};

}  // namespace

// Finds the open chain with the given key, or null if none is open.
static auto FindChain(llvm::SmallVectorImpl<ChainState>& chains, int key)
    -> ChainState* {
  for (ChainState& chain : chains) {
    if (chain.key == key) {
      return &chain;
    }
  }
  return nullptr;
}

// Serializes the layout-relevant part of a state (everything that determines
// its future, excluding the path bookkeeping) into a deduplication key. Two
// states with equal keys have identical continuations, so only the cheaper need
// be explored.
//
// TODO: This builds a `SmallVector` key per explored state to store in the
// `settled` set (and each expansion copies the anchor stack into its successor
// states). If the solver ever needs to be faster, hashing the state into a
// fixed-width key would avoid the per-state allocations.
static auto StateKey(const State& state) -> llvm::SmallVector<int> {
  llvm::SmallVector<int> key;
  key.reserve(state.stack.size() + state.chains.size() * 3 + 4);
  key.push_back(state.index);
  key.push_back(state.column);
  key.push_back(state.stack.size());
  key.insert(key.end(), state.stack.begin(), state.stack.end());
  // Canonicalize chains by key so that two states differing only in the
  // (search- irrelevant) order of open chains dedupe against each other.
  llvm::SmallVector<const ChainState*, 4> sorted_chains;
  for (const ChainState& chain : state.chains) {
    sorted_chains.push_back(&chain);
  }
  llvm::sort(sorted_chains, [](const ChainState* a, const ChainState* b) {
    return a->key < b->key;
  });
  key.push_back(sorted_chains.size());
  for (const ChainState* chain : sorted_chains) {
    key.push_back(chain->key);
    key.push_back(chain->anchor);
    key.push_back(static_cast<int>(chain->decision));
  }
  return key;
}

// Returns the excess-character penalty for extending a line from `from_column`
// to `to_column`, charging `style.penalty_excess_character` for each column
// past the limit. Phrasing it as a delta over the line's prior extent counts
// each excess column exactly once across the tokens that share a line.
static auto ExcessPenalty(int from_column, int to_column, const Style& style)
    -> int64_t {
  int excess = std::max(0, to_column - style.column_limit) -
               std::max(0, from_column - style.column_limit);
  return style.penalty_excess_character * excess;
}

auto SolveLineBreaks(const Lex::TokenizedBuffer& tokens,
                     const TokenInfoStore& token_infos,
                     llvm::ArrayRef<Lex::TokenIndex> line, int indent,
                     const Style& style) -> llvm::SmallVector<int> {
  int token_count = line.size();
  llvm::SmallVector<int> breaks(token_count, -1);
  if (token_count <= 1) {
    return breaks;
  }

  ChainInfo chain_info = ComputeChainInfo(tokens, token_infos, line);

  // Opens a member-access chain when the token at `position`, placed starting
  // at `start_column`, is a chain's receiver root, recording the column its
  // members indent to.
  auto maybe_open_chain = [&](int position, int start_column,
                              llvm::SmallVectorImpl<ChainState>& chains) {
    int key = chain_info.root_key[position];
    if (key >= 0) {
      chains.push_back(
          {.key = key,
           .anchor = start_column + style.continuation_indent_width,
           .decision = ChainBreak::Undecided});
    }
  };
  // Drops a chain's state once its last member access has been placed.
  auto maybe_close_chain = [&](int position,
                               llvm::SmallVectorImpl<ChainState>& chains) {
    int key = token_infos.Get(line[position]).member_chain_id;
    int* last_pos = key >= 0 ? chain_info.last_member_pos[key] : nullptr;
    if (last_pos && *last_pos == position) {
      ChainState* chain = FindChain(chains, key);
      if (chain) {
        *chain = chains.back();
        chains.pop_back();
      }
    }
  };

  // The pool of explored states; `nodes[0]` is the start state, with the line's
  // first token already placed at `indent`.
  llvm::SmallVector<State> nodes;
  {
    llvm::SmallVector<int, 8> stack;
    // The statement-level continuation indent, used for breaks outside any
    // bracket.
    stack.push_back(indent + style.continuation_indent_width);
    int column = indent + token_infos.Get(line[0]).column_width;
    ApplyTokenScopes(tokens.GetKind(line[0]), token_infos.Get(line[0]),
                     /*start_column=*/indent, /*end_column=*/column, stack);
    llvm::SmallVector<ChainState, 4> chains;
    maybe_open_chain(0, /*start_column=*/indent, chains);
    nodes.push_back({.index = 1,
                     .column = column,
                     .stack = std::move(stack),
                     .chains = std::move(chains),
                     .parent = -1,
                     .newline_indent = -1});
  }

  // A min-priority queue keyed by penalty, plus the set of settled state keys.
  // With non-negative step costs, the first time a state is popped is along a
  // lowest-penalty path, so it can be settled and need never be revisited.
  using QueueEntry = std::pair<int64_t, int>;
  std::priority_queue<QueueEntry, llvm::SmallVector<QueueEntry>, std::greater<>>
      queue;
  queue.push({0, 0});
  Set<llvm::SmallVector<int>> settled;

  int goal = -1;
  while (!queue.empty()) {
    auto [penalty, node_index] = queue.top();
    queue.pop();

    if (!settled.Insert(StateKey(nodes[node_index])).is_inserted()) {
      // Reached earlier along a path that was at least as cheap.
      continue;
    }
    if (nodes[node_index].index == token_count) {
      goal = node_index;
      break;
    }
    if (static_cast<int>(nodes.size()) > MaxStatesGenerated) {
      // Pathounwrapped line; leave it unwrapped rather than searching on.
      return llvm::SmallVector<int>(token_count, -1);
    }

    // Copy out the fields used below: appending to `nodes` may move it.
    int index = nodes[node_index].index;
    int column = nodes[node_index].column;
    llvm::SmallVector<int, 8> stack(nodes[node_index].stack);
    llvm::SmallVector<ChainState, 4> chains(nodes[node_index].chains);

    Lex::TokenIndex previous = line[index - 1];
    Lex::TokenIndex token = line[index];
    Lex::TokenKind token_kind = tokens.GetKind(token);
    int token_width = token_infos.Get(token).column_width;

    // A member-access break indents to its chain's anchor rather than the
    // current bracket level, and may be constrained by the chain's all-or-
    // nothing decision.
    int member_chain_id = token_infos.Get(token).member_chain_id;
    bool is_member = member_chain_id >= 0;
    ChainState* chain =
        is_member ? FindChain(chains, member_chain_id) : nullptr;
    bool fluent = chain_info.is_fluent[index];
    bool* builder_entry =
        is_member ? chain_info.is_builder[member_chain_id] : nullptr;
    bool builder = builder_entry && *builder_entry;
    ChainBreak decision = chain ? chain->decision : ChainBreak::Undecided;

    // Whether a break before `token` is forbidden by the member-chain rules:
    //   - in a builder chain, only fluent points (those following a `)`/`]`)
    //     break, so the first segment and field accesses stay attached;
    //   - a fluent point whose chain committed to staying packed cannot break.
    //
    // TODO: The first rule is a hard block where clang-format uses a penalty,
    // so a builder chain whose receiver plus first segment alone overflow the
    // limit stays overflowing where clang-format could still break before the
    // first `.`. Rare in practice; revisit if it shows up in real code.
    bool member_break_blocked =
        (builder && !fluent) || (fluent && decision == ChainBreak::Unbroken);

    // Keep `token` on the current line. A fluent break point whose chain
    // already committed to breaking cannot stay on the line.
    if (!fluent || decision != ChainBreak::Broken) {
      int start_column =
          column + SpacesBefore(tokens, token_infos, previous, token);
      int end_column = start_column + token_width;
      int64_t step = ExcessPenalty(column, end_column, style);
      llvm::SmallVector<int, 8> next_stack(stack);
      ApplyTokenScopes(token_kind, token_infos.Get(token), start_column,
                       end_column, next_stack);
      llvm::SmallVector<ChainState, 4> next_chains(chains);
      if (fluent && decision == ChainBreak::Undecided) {
        if (ChainState* c = FindChain(next_chains, member_chain_id)) {
          c->decision = ChainBreak::Unbroken;
        }
      }
      maybe_open_chain(index, start_column, next_chains);
      maybe_close_chain(index, next_chains);
      nodes.push_back({.index = index + 1,
                       .column = end_column,
                       .stack = std::move(next_stack),
                       .chains = std::move(next_chains),
                       .parent = node_index,
                       .newline_indent = -1});
      queue.push({penalty + step, static_cast<int>(nodes.size()) - 1});
    }

    // Break before `token`. A member access indents to its chain anchor;
    // everything else to the current bracket level. The member-chain rules can
    // forbid the break (see `member_break_blocked`).
    if (CanBreakBefore(tokens, token_infos, previous, token) &&
        !member_break_blocked) {
      int break_indent = chain ? chain->anchor : stack.back();
      int end_column = break_indent + token_width;
      // A continuation line's excess telescope starts at its indent, clamped
      // to the limit so that when the anchor itself sits past the limit the
      // indentation columns beyond it are still charged and a deep-anchor
      // break is not undercharged.
      int64_t step = SplitPenalty(tokens, token_infos, previous, token, style) +
                     ExcessPenalty(std::min(break_indent, style.column_limit),
                                   end_column, style);
      llvm::SmallVector<int, 8> next_stack(std::move(stack));
      ApplyTokenScopes(token_kind, token_infos.Get(token), break_indent,
                       end_column, next_stack);
      llvm::SmallVector<ChainState, 4> next_chains(chains);
      if (fluent && decision == ChainBreak::Undecided) {
        if (ChainState* c = FindChain(next_chains, member_chain_id)) {
          c->decision = ChainBreak::Broken;
        }
      }
      maybe_open_chain(index, break_indent, next_chains);
      maybe_close_chain(index, next_chains);
      nodes.push_back({.index = index + 1,
                       .column = end_column,
                       .stack = std::move(next_stack),
                       .chains = std::move(next_chains),
                       .parent = node_index,
                       .newline_indent = break_indent});
      queue.push({penalty + step, static_cast<int>(nodes.size()) - 1});
    }
  }

  // A goal is always reachable, since option 1 is available except where a
  // committed chain forces a break (which is itself always available), but
  // guard against the state cap or an empty queue regardless.
  if (goal < 0) {
    return breaks;
  }

  // Replay the chosen path, recording each token's break decision.
  for (int node_index = goal; node_index >= 0;
       node_index = nodes[node_index].parent) {
    breaks[nodes[node_index].index - 1] = nodes[node_index].newline_indent;
  }
  return breaks;
}

}  // namespace Carbon::Format

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
#include "common/set.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/format/token_info.h"
#include "toolchain/lex/token_kind.h"

namespace Carbon::Format {

// Each column past the column limit costs this much penalty. It dwarfs every
// split penalty, which is what makes the limit "soft": the solver prefers any
// legal break over overrunning, yet can still exceed the limit when no break
// helps, producing best-effort output rather than failing. Matches
// clang-format's `PenaltyExcessCharacter`.
constexpr int64_t PenaltyExcessCharacter = 1'000'000;

// An upper bound on the number of search states generated (roughly twice the
// number explored) before giving up and leaving the line unwrapped. Ordinary
// lines generate far fewer; this only bounds the cost of pathological inputs
// (very long, deeply nested lines).
constexpr int MaxStatesGenerated = 100'000;

// Updates the continuation-indent stack for the bracket nesting that a token
// of kind `kind` introduces, given that the token's text ends at `end_column`.
// An opening bracket pushes the column just past it as the alignment anchor
// for its contents (`AlignAfterOpenBracket = Align`); a closing bracket pops
// back to the enclosing level. The bottom entry (the statement-level
// continuation indent) is never popped, so a stray closing bracket in
// malformed input can't empty the stack.
static auto ApplyBracket(Lex::TokenKind kind, int end_column,
                         llvm::SmallVectorImpl<int>& stack) -> void {
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
  // The state this was reached from, as an index into the node pool, or -1 for
  // the start state.
  int parent;
  // The break decision that produced this state: the indent column if a break
  // precedes token `index - 1`, or -1 if not.
  int newline_indent;
};

}  // namespace

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
  key.reserve(state.stack.size() + 2);
  key.push_back(state.index);
  key.push_back(state.column);
  key.insert(key.end(), state.stack.begin(), state.stack.end());
  return key;
}

// Returns the excess-character penalty for extending a line from `from_column`
// to `to_column`, charging `PenaltyExcessCharacter` for each column past the
// limit. Phrasing it as a delta over the line's prior extent counts each excess
// column exactly once across the tokens that share a line.
static auto ExcessPenalty(int from_column, int to_column) -> int64_t {
  int excess = std::max(0, to_column - ColumnLimit) -
               std::max(0, from_column - ColumnLimit);
  return PenaltyExcessCharacter * excess;
}

auto SolveLineBreaks(const Lex::TokenizedBuffer& tokens,
                     const TokenInfoStore& token_infos,
                     llvm::ArrayRef<Lex::TokenIndex> line, int indent)
    -> llvm::SmallVector<int> {
  int token_count = line.size();
  llvm::SmallVector<int> breaks(token_count, -1);
  if (token_count <= 1) {
    return breaks;
  }

  // The pool of explored states; `nodes[0]` is the start state, with the line's
  // first token already placed at `indent`.
  llvm::SmallVector<State> nodes;
  {
    llvm::SmallVector<int, 8> stack;
    // The statement-level continuation indent, used for breaks outside any
    // bracket.
    stack.push_back(indent + ContinuationIndentWidth);
    int column = indent + token_infos.Get(line[0]).column_width;
    ApplyBracket(tokens.GetKind(line[0]), column, stack);
    nodes.push_back({.index = 1,
                     .column = column,
                     .stack = std::move(stack),
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

    Lex::TokenIndex previous = line[index - 1];
    Lex::TokenIndex token = line[index];
    Lex::TokenKind token_kind = tokens.GetKind(token);
    int token_width = token_infos.Get(token).column_width;

    // Option 1: keep `token` on the current line.
    {
      int end_column = column +
                       SpacesBefore(tokens, token_infos, previous, token) +
                       token_width;
      int64_t step = ExcessPenalty(column, end_column);
      llvm::SmallVector<int, 8> next_stack(stack);
      ApplyBracket(token_kind, end_column, next_stack);
      nodes.push_back({.index = index + 1,
                       .column = end_column,
                       .stack = std::move(next_stack),
                       .parent = node_index,
                       .newline_indent = -1});
      queue.push({penalty + step, static_cast<int>(nodes.size()) - 1});
    }

    // Option 2: break before `token`, placing it at the current continuation
    // indent, where a break is allowed.
    if (CanBreakBefore(tokens, token_infos, previous, token)) {
      int break_indent = stack.back();
      int end_column = break_indent + token_width;
      // A continuation line's excess telescope starts at its indent, clamped
      // to the limit so that when the anchor itself sits past the limit the
      // indentation columns beyond it are still charged and a deep-anchor
      // break is not undercharged.
      int64_t step =
          SplitPenalty(tokens, token_infos, previous, token) +
          ExcessPenalty(std::min(break_indent, ColumnLimit), end_column);
      llvm::SmallVector<int, 8> next_stack(std::move(stack));
      ApplyBracket(token_kind, end_column, next_stack);
      nodes.push_back({.index = index + 1,
                       .column = end_column,
                       .stack = std::move(next_stack),
                       .parent = node_index,
                       .newline_indent = break_indent});
      queue.push({penalty + step, static_cast<int>(nodes.size()) - 1});
    }
  }

  // A goal is always reachable, since option 1 is always available, but guard
  // against the state cap or an empty queue regardless.
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

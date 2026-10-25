// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/line_wrapper.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <queue>
#include <utility>

#include "common/check.h"
#include "common/hashing.h"
#include "common/map.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/format/token_info.h"
#include "toolchain/lex/token_kind.h"

namespace Carbon::Format {

// Additional penalty per enclosing `(`, `[`, or inline `{` on the unwrapped
// line so outer breaks are preferred over splitting inside bracketed
// subexpressions.
constexpr int PenaltyPerBracketDepth = 20;

// Upper bound on search states generated before falling back to linear-time
// greedy packing on pathological inputs.
constexpr int MaxStatesGenerated = 100'000;

static auto IsInlineOpenBracket(Lex::TokenKind kind, TokenRole role) -> bool {
  return kind.IsOneOf(
             {Lex::TokenKind::OpenParen, Lex::TokenKind::OpenSquareBracket}) ||
         (kind == Lex::TokenKind::OpenCurlyBrace &&
          role == TokenRole::StructOpen);
}

static auto IsInlineCloseBracket(Lex::TokenKind kind, TokenRole role) -> bool {
  return kind.IsOneOf({Lex::TokenKind::CloseParen,
                       Lex::TokenKind::CloseSquareBracket}) ||
         (kind == Lex::TokenKind::CloseCurlyBrace &&
          role == TokenRole::StructClose);
}

namespace {

// Interns coalesced continuation-indent scope stacks in a persistent DAG so
// each solver `State` stores a single 32-bit scope ID. ID 0 is the root
// statement-level continuation scope.
class ScopeTable {
 public:
  explicit ScopeTable(int initial_column) {
    Intern({.parent = -1, .column = initial_column, .count = 1});
  }

  auto TopColumn(int32_t id) const -> int { return nodes_[id].column; }

  auto CanPop(int32_t id) const -> bool {
    return nodes_[id].parent != -1 || nodes_[id].count > 1;
  }

  auto Push(int32_t id, int column, int count) -> int32_t {
    if (count <= 0) {
      return id;
    }
    Node top = nodes_[id];
    if (top.column == column) {
      return Intern(
          {.parent = top.parent, .column = column, .count = top.count + count});
    }
    return Intern({.parent = id, .column = column, .count = count});
  }

  auto Pop(int32_t id, int count) -> int32_t {
    while (count > 0 && CanPop(id)) {
      Node top = nodes_[id];
      if (top.parent == -1) {
        return Intern({.parent = -1,
                       .column = top.column,
                       .count = std::max(1, top.count - count)});
      }
      if (count < top.count) {
        return Intern({.parent = top.parent,
                       .column = top.column,
                       .count = top.count - count});
      }
      count -= top.count;
      id = top.parent;
    }
    return id;
  }

 private:
  struct Node {
    int32_t parent;
    int32_t column;
    int32_t count;

    friend auto operator==(Node, Node) -> bool = default;

    friend auto CarbonHashValue(Node n) -> HashCode {
      Hasher hasher;
      hasher.Hash(n.parent, n.column, n.count);
      return static_cast<HashCode>(hasher);
    }
  };

  auto Intern(Node node) -> int32_t {
    int32_t next_id = static_cast<int32_t>(nodes_.size());
    auto insert = index_by_node_.Insert(node, next_id);
    if (insert.is_inserted()) {
      nodes_.push_back(node);
    }
    return insert.value();
  }

  llvm::SmallVector<Node> nodes_;
  Map<Node, int32_t> index_by_node_;
};

// Line-local metadata for one distinct member-access chain.
struct ChainMeta {
  int chain_id;
  int root_pos;
  int last_member_pos;
  int total_width = 0;
  bool seen_call = false;
  bool is_builder = false;
  bool has_forced_break = false;
};

// Precomputed line-local member-access chain metadata indexed by line position
// and dense chain slot. Empty when `line` contains no member-access chains.
struct ChainInfo {
  bool has_chains = false;
  // Dense slot opened when placing `line[pos]`, or -1 if `line[pos]` is not a
  // chain receiver root.
  llvm::SmallVector<int> root_slot;
  // Dense slot of the chain that `.` / `->` at `line[pos]` belongs to, or -1.
  llvm::SmallVector<int> member_slot;
  // Fluent call/subscript `.` / `->` positions coupled by a builder chain.
  llvm::BitVector is_fluent;
  // Final `.` / `->` positions (`last_member_pos`) that pop a chain's scope.
  llvm::BitVector closes_chain;
  llvm::SmallVector<ChainMeta, 4> chains;
};

// One state in the shortest-path search: tokens `0..index - 1` have been
// placed, and `index` is the next token to place.
struct State {
  int32_t index;
  // Column just past the last placed token.
  int32_t column;
  // Starting indent of the current physical line.
  int32_t line_indent;
  // Interned scope stack ID in `ScopeTable`.
  int32_t scope_id;
  // Predecessor index in `nodes`, or -1 for the initial state.
  int32_t parent;
  // Indent column if a break precedes `line[index - 1]`, or -1 otherwise.
  int32_t newline_indent;
};
static_assert(sizeof(State) == 24);

// Key for settled-state deduplication and Pareto column dominance. When
// `scope_open_ahead[index]` is true, `key_column` is the exact column; when
// false, `key_column` is -1 so states with larger columns are pruned against
// the minimum settled column for `(index, effective_line_indent, scope_id)`.
struct SettledKey {
  int32_t index;
  int32_t key_column;
  int32_t effective_line_indent;
  int32_t scope_id;

  friend auto operator==(SettledKey, SettledKey) -> bool = default;

  friend auto CarbonHashValue(SettledKey k) -> HashCode {
    Hasher hasher;
    hasher.Hash(k.index, k.key_column, k.effective_line_indent, k.scope_id);
    return static_cast<HashCode>(hasher);
  }
};

}  // namespace

// Precomputes `ChainInfo` for `line`. Returns an empty `ChainInfo` without
// allocating when `line` has no member-access links.
static auto ComputeChainInfo(const Lex::TokenizedBuffer& tokens,
                             const TokenInfoStore& token_infos,
                             llvm::ArrayRef<Lex::TokenIndex> line,
                             const LineBreakConstraints* constraints)
    -> ChainInfo {
  ChainInfo info;
  if (llvm::none_of(line, [&](Lex::TokenIndex token) {
        return token_infos.Get(token).member_chain_id >= 0;
      })) {
    return info;
  }

  int n = line.size();
  info.has_chains = true;
  info.root_slot.assign(n, -1);
  info.member_slot.assign(n, -1);
  info.is_fluent.resize(n);

  for (int i = 0; i < n; ++i) {
    const TokenInfo& tok_info = token_infos.Get(line[i]);
    int chain_id = tok_info.member_chain_id;
    if (chain_id < 0) {
      continue;
    }

    int slot = -1;
    for (int s = 0; s < static_cast<int>(info.chains.size()); ++s) {
      if (info.chains[s].chain_id == chain_id) {
        slot = s;
        break;
      }
    }
    if (slot < 0) {
      // `line` is strictly sorted by `TokenIndex`, so binary search locates the
      // receiver root without a hash map.
      auto it = llvm::lower_bound(line, Lex::TokenIndex(chain_id));
      if (it == line.end() || *it != Lex::TokenIndex(chain_id)) {
        continue;
      }
      int root_pos = static_cast<int>(it - line.begin());
      if (root_pos >= i || info.root_slot[root_pos] >= 0) {
        continue;
      }
      bool receiver_is_call = false;
      if (i > 0 && tokens.GetKind(line[i - 1])
                       .IsOneOf({Lex::TokenKind::CloseParen,
                                 Lex::TokenKind::CloseSquareBracket})) {
        Lex::TokenIndex open = tokens.GetMatchedOpeningToken(line[i - 1]);
        receiver_is_call = open.has_value() && token_infos.Get(open).role ==
                                                   TokenRole::PostfixBracket;
      }
      slot = static_cast<int>(info.chains.size());
      info.root_slot[root_pos] = slot;
      info.chains.push_back({.chain_id = chain_id,
                             .root_pos = root_pos,
                             .last_member_pos = i,
                             .seen_call = receiver_is_call});
    } else {
      info.chains[slot].last_member_pos = i;
    }

    info.member_slot[i] = slot;
    ChainMeta& meta = info.chains[slot];
    if (tok_info.is_call_link) {
      if (meta.seen_call) {
        info.is_fluent.set(i);
        meta.is_builder = true;
      }
      meta.seen_call = true;
    }
  }

  if (info.chains.empty()) {
    return ChainInfo();
  }

  info.closes_chain.resize(n);
  for (ChainMeta& meta : info.chains) {
    info.closes_chain.set(meta.last_member_pos);
    if (!meta.is_builder) {
      continue;
    }

    int end = meta.last_member_pos;
    auto advance_to_close = [&](int open_pos) -> bool {
      Lex::TokenIndex close = tokens.GetMatchedClosingToken(line[open_pos]);
      if (!close.has_value()) {
        return false;
      }
      auto close_it = llvm::lower_bound(line.slice(open_pos + 1), close);
      if (close_it == line.end() || *close_it != close) {
        end = n - 1;
        return false;
      }
      end = static_cast<int>(close_it - line.begin());
      return true;
    };

    if (end + 1 < n &&
        !CanBreakBefore(tokens, token_infos, line[end], line[end + 1])) {
      ++end;
      if (tokens.GetKind(line[end]) == Lex::TokenKind::OpenParen) {
        advance_to_close(end);
      }
    }
    while (end + 1 < n &&
           token_infos.Get(line[end + 1]).role == TokenRole::PostfixBracket &&
           tokens.GetKind(line[end + 1])
               .IsOneOf({Lex::TokenKind::OpenParen,
                         Lex::TokenKind::OpenSquareBracket})) {
      if (!advance_to_close(end + 1)) {
        break;
      }
    }
    while (end + 1 < n &&
           !CanBreakBefore(tokens, token_infos, line[end], line[end + 1])) {
      ++end;
    }

    if (constraints) {
      for (int k = meta.root_pos + 1; k <= end; ++k) {
        if (constraints->must_break_before.test(k)) {
          meta.has_forced_break = true;
          break;
        }
      }
    }
    meta.total_width =
        RenderedWidth(tokens, token_infos,
                      line.slice(meta.root_pos, end - meta.root_pos + 1)) +
        (constraints ? constraints->trailing_comment_widths[end] : 0);
  }
  return info;
}

// Updates `scope_id` in `scopes` for operand-alignment scopes, member-access
// chain receiver scopes, and `(`, `[`, or inline `{` brackets opened or closed
// at `line[pos]` spanning `[start_column, token_end)`. Because member access
// and postfix brackets bind tighter than infix and `where` operators and any
// bracketed subexpression inside a chain closes before its `last_member_pos`,
// all three scope kinds nest in LIFO order.
static auto ApplyTokenScopes(int pos, Lex::TokenKind kind,
                             const TokenInfo& info, const ChainInfo& chain_info,
                             int start_column, int token_end,
                             const Style& style, ScopeTable& scopes,
                             int32_t scope_id) -> int32_t {
  if (info.open_scopes > 0) {
    scope_id = scopes.Push(scope_id,
                           std::max(start_column, scopes.TopColumn(scope_id)),
                           info.open_scopes);
  }
  if (chain_info.has_chains && chain_info.root_slot[pos] >= 0) {
    scope_id = scopes.Push(scope_id,
                           start_column + style.continuation_indent_width, 1);
  }
  if (IsInlineOpenBracket(kind, info.role)) {
    scope_id = scopes.Push(scope_id, token_end, 1);
  } else if (IsInlineCloseBracket(kind, info.role)) {
    scope_id = scopes.Pop(scope_id, 1);
  }
  if (chain_info.has_chains && chain_info.closes_chain.test(pos)) {
    scope_id = scopes.Pop(scope_id, 1);
  }
  if (info.close_scopes > 0) {
    scope_id = scopes.Pop(scope_id, info.close_scopes);
  }
  return scope_id;
}

// Shared per-token break eligibility and trailing-comment width for `line[pos]`
// used by both `SolveLineBreaks` and `SolveLineBreaksGreedy`.
struct StepBreakInfo {
  int trailing_width = 0;
  bool forced_break = false;
  bool can_keep_same_line = true;
  bool allow_break = false;
};

static auto ComputeStepBreakInfo(
    const Lex::TokenizedBuffer& tokens, const TokenInfoStore& token_infos,
    llvm::ArrayRef<Lex::TokenIndex> line, int pos, const Style& style,
    const LineBreakConstraints* constraints, const ChainInfo& chain_info,
    const ScopeTable& scopes, int32_t scope_id) -> StepBreakInfo {
  Lex::TokenIndex previous = line[pos - 1];
  Lex::TokenIndex token = line[pos];
  bool forced_break = constraints && constraints->must_break_before.test(pos);
  int trailing_width =
      constraints ? constraints->trailing_comment_widths[pos] : 0;
  bool can_break =
      forced_break || CanBreakBefore(tokens, token_infos, previous, token);

  int slot = chain_info.has_chains ? chain_info.member_slot[pos] : -1;
  bool builder = slot >= 0 && chain_info.chains[slot].is_builder;
  bool fluent = slot >= 0 && chain_info.is_fluent.test(pos);
  bool chain_must_break = false;
  if (builder) {
    const ChainMeta& meta = chain_info.chains[slot];
    int recv_col = scopes.TopColumn(scope_id) - style.continuation_indent_width;
    chain_must_break = meta.has_forced_break ||
                       recv_col + meta.total_width > style.column_limit;
  }

  // In a builder chain, non-fluent links (the initial segment and plain field
  // accesses) stay attached to their base, and fluent links stay packed when
  // the chain fits on one line from `recv_col` without forced breaks.
  bool member_break_blocked =
      (builder && !fluent) || (fluent && !chain_must_break);
  return {
      .trailing_width = trailing_width,
      .forced_break = forced_break,
      .can_keep_same_line =
          !forced_break && (!can_break || !fluent || !chain_must_break),
      .allow_break = forced_break || (can_break && !member_break_blocked),
  };
}

// Computes the indentation column for breaking before the token after
// `previous`, demoting an opening `(`, `[`, or inline `{` anchor to
// `line_indent + style.continuation_indent_width` when breaking immediately
// after that bracket.
static auto ResolveBreakIndent(const Lex::TokenizedBuffer& tokens,
                               const TokenInfoStore& token_infos,
                               Lex::TokenIndex previous, int line_indent,
                               const Style& style, ScopeTable& scopes,
                               int32_t& scope_id) -> int {
  int break_indent = scopes.TopColumn(scope_id);
  if (IsInlineOpenBracket(tokens.GetKind(previous),
                          token_infos.Get(previous).role) &&
      token_infos.Get(previous).close_scopes == 0 && scopes.CanPop(scope_id)) {
    break_indent = line_indent + style.continuation_indent_width;
    scope_id = scopes.Pop(scope_id, 1);
    scope_id = scopes.Push(scope_id, break_indent, 1);
  }
  return break_indent;
}

// Greedy first-fit fallback used when `SolveLineBreaks` exceeds
// `MaxStatesGenerated` on a very long or deeply nested unwrapped line.
static auto SolveLineBreaksGreedy(
    const Lex::TokenizedBuffer& tokens, const TokenInfoStore& token_infos,
    llvm::ArrayRef<Lex::TokenIndex> line, int indent, const Style& style,
    const LineBreakConstraints* constraints, const ChainInfo& chain_info,
    ScopeTable& scopes) -> llvm::SmallVector<int> {
  int token_count = line.size();
  llvm::SmallVector<int> breaks(token_count, -1);
  int32_t scope_id = 0;
  int line_indent = indent;
  int token_end = indent + token_infos.Get(line[0]).column_width;
  scope_id = ApplyTokenScopes(
      0, tokens.GetKind(line[0]), token_infos.Get(line[0]), chain_info,
      /*start_column=*/indent, token_end, style, scopes, scope_id);
  int column =
      token_end + (constraints ? constraints->trailing_comment_widths[0] : 0);

  for (int i = 1; i < token_count; ++i) {
    Lex::TokenIndex previous = line[i - 1];
    Lex::TokenIndex token = line[i];
    const TokenInfo& info = token_infos.Get(token);
    StepBreakInfo step =
        ComputeStepBreakInfo(tokens, token_infos, line, i, style, constraints,
                             chain_info, scopes, scope_id);

    int start_column =
        column + SpacesBefore(tokens, token_infos, previous, token);
    token_end = start_column + info.column_width;
    if (!step.can_keep_same_line ||
        (step.allow_break &&
         token_end + step.trailing_width > style.column_limit)) {
      int break_indent = ResolveBreakIndent(
          tokens, token_infos, previous, line_indent, style, scopes, scope_id);
      breaks[i] = break_indent;
      line_indent = break_indent;
      start_column = break_indent;
      token_end = break_indent + info.column_width;
    }
    scope_id =
        ApplyTokenScopes(i, tokens.GetKind(token), info, chain_info,
                         start_column, token_end, style, scopes, scope_id);
    column = token_end + step.trailing_width;
  }
  return breaks;
}

// Incremental excess-column penalty when extending a line from `from_column` to
// `to_column`.
static auto ExcessPenalty(int from_column, int to_column, const Style& style)
    -> int64_t {
  int excess = std::max(0, to_column - style.column_limit) -
               std::max(0, from_column - style.column_limit);
  return style.penalty_excess_character * excess;
}

auto SolveLineBreaks(const Lex::TokenizedBuffer& tokens,
                     const TokenInfoStore& token_infos,
                     llvm::ArrayRef<Lex::TokenIndex> line, int indent,
                     const Style& style,
                     const LineBreakConstraints* constraints)
    -> llvm::SmallVector<int> {
  int token_count = line.size();
  CARBON_CHECK(
      !constraints ||
      (static_cast<int>(constraints->must_break_before.size()) == token_count &&
       static_cast<int>(constraints->trailing_comment_widths.size()) ==
           token_count));
  llvm::SmallVector<int> breaks(token_count, -1);
  if (token_count <= 1) {
    return breaks;
  }

  ChainInfo chain_info =
      ComputeChainInfo(tokens, token_infos, line, constraints);

  llvm::SmallVector<int> bracket_depths(token_count, 0);
  int depth = 0;
  for (int i = 0; i < token_count; ++i) {
    bracket_depths[i] = depth;
    Lex::TokenKind kind = tokens.GetKind(line[i]);
    TokenRole role = token_infos.Get(line[i]).role;
    if (IsInlineOpenBracket(kind, role)) {
      ++depth;
    } else if (IsInlineCloseBracket(kind, role)) {
      depth = std::max(0, depth - 1);
    }
  }

  llvm::BitVector line_indent_live(token_count);
  llvm::BitVector scope_open_ahead(token_count);
  bool indent_live = false;
  bool scope_ahead = false;
  for (int i = token_count - 1; i >= 1; --i) {
    if (constraints && constraints->must_break_before.test(i)) {
      indent_live = false;
      scope_ahead = false;
    } else {
      const TokenInfo& tok_info = token_infos.Get(line[i]);
      if (tok_info.open_scopes > 0 ||
          (chain_info.has_chains && chain_info.root_slot[i] >= 0) ||
          IsInlineOpenBracket(tokens.GetKind(line[i]), tok_info.role)) {
        scope_ahead = true;
      }
    }
    Lex::TokenIndex prev = line[i - 1];
    if (IsInlineOpenBracket(tokens.GetKind(prev), token_infos.Get(prev).role) &&
        token_infos.Get(prev).close_scopes == 0) {
      indent_live = true;
    }
    if (indent_live) {
      line_indent_live.set(i);
    }
    if (scope_ahead) {
      scope_open_ahead.set(i);
    }
  }

  ScopeTable scopes(indent + style.continuation_indent_width);
  llvm::SmallVector<State> nodes;
  {
    int token_end = indent + token_infos.Get(line[0]).column_width;
    int32_t scope_id = ApplyTokenScopes(
        0, tokens.GetKind(line[0]), token_infos.Get(line[0]), chain_info,
        /*start_column=*/indent, token_end, style, scopes, 0);
    int column =
        token_end + (constraints ? constraints->trailing_comment_widths[0] : 0);
    nodes.push_back({.index = 1,
                     .column = column,
                     .line_indent = indent,
                     .scope_id = scope_id,
                     .parent = -1,
                     .newline_indent = -1});
  }

  using QueueEntry = std::pair<int64_t, int>;
  std::priority_queue<QueueEntry, llvm::SmallVector<QueueEntry>, std::greater<>>
      queue;
  queue.push({0, 0});
  Map<SettledKey, int32_t> min_column;

  int goal = -1;
  while (!queue.empty()) {
    auto [penalty, node_index] = queue.top();
    queue.pop();

    const State s = nodes[node_index];
    if (s.index == token_count) {
      goal = node_index;
      break;
    }

    int32_t col = (constraints && constraints->must_break_before.test(s.index))
                      ? 0
                      : s.column;
    SettledKey key = {
        .index = s.index,
        .key_column = scope_open_ahead.test(s.index) ? col : -1,
        .effective_line_indent =
            line_indent_live.test(s.index) ? s.line_indent : 0,
        .scope_id = s.scope_id,
    };
    auto insert = min_column.Insert(key, col);
    if (!insert.is_inserted()) {
      if (col >= insert.value()) {
        continue;
      }
      insert.value() = col;
    }

    if (static_cast<int>(nodes.size()) > MaxStatesGenerated) {
      return SolveLineBreaksGreedy(tokens, token_infos, line, indent, style,
                                   constraints, chain_info, scopes);
    }

    Lex::TokenIndex previous = line[s.index - 1];
    Lex::TokenIndex token = line[s.index];
    Lex::TokenKind token_kind = tokens.GetKind(token);
    const TokenInfo& info = token_infos.Get(token);
    StepBreakInfo step_info =
        ComputeStepBreakInfo(tokens, token_infos, line, s.index, style,
                             constraints, chain_info, scopes, s.scope_id);

    // Keep `token` on the current line unless a preceding comment or an
    // overflowing builder chain forces a break here.
    if (step_info.can_keep_same_line) {
      int start_column =
          s.column + SpacesBefore(tokens, token_infos, previous, token);
      int token_end = start_column + info.column_width;
      int end_column = token_end + step_info.trailing_width;
      int64_t step = ExcessPenalty(s.column, end_column, style);
      int32_t next_scope_id =
          ApplyTokenScopes(s.index, token_kind, info, chain_info, start_column,
                           token_end, style, scopes, s.scope_id);
      nodes.push_back({.index = s.index + 1,
                       .column = end_column,
                       .line_indent = s.line_indent,
                       .scope_id = next_scope_id,
                       .parent = node_index,
                       .newline_indent = -1});
      queue.push({penalty + step, static_cast<int>(nodes.size()) - 1});
    }

    // Break before `token`, indenting to the active scope in `scopes` (or
    // demoting an immediately preceding inline open bracket).
    if (step_info.allow_break) {
      int32_t next_scope_id = s.scope_id;
      int break_indent =
          ResolveBreakIndent(tokens, token_infos, previous, s.line_indent,
                             style, scopes, next_scope_id);
      int token_end = break_indent + info.column_width;
      int end_column = token_end + step_info.trailing_width;
      // Clamp the starting column to `style.column_limit` so an anchor already
      // past the limit still pays for excess indentation columns. Mandatory
      // comment breaks have no same-line alternative, so they pay no split
      // penalty.
      int64_t split_penalty =
          step_info.forced_break
              ? 0
              : SplitPenalty(tokens, token_infos, previous, token, style) +
                    PenaltyPerBracketDepth * bracket_depths[s.index];
      int64_t step = split_penalty +
                     ExcessPenalty(std::min(break_indent, style.column_limit),
                                   end_column, style);
      next_scope_id =
          ApplyTokenScopes(s.index, token_kind, info, chain_info, break_indent,
                           token_end, style, scopes, next_scope_id);
      nodes.push_back({.index = s.index + 1,
                       .column = end_column,
                       .line_indent = break_indent,
                       .scope_id = next_scope_id,
                       .parent = node_index,
                       .newline_indent = break_indent});
      queue.push({penalty + step, static_cast<int>(nodes.size()) - 1});
    }
  }

  if (goal < 0) {
    return breaks;
  }

  for (int node_index = goal; node_index >= 0;
       node_index = nodes[node_index].parent) {
    breaks[nodes[node_index].index - 1] = nodes[node_index].newline_indent;
  }
  return breaks;
}

}  // namespace Carbon::Format

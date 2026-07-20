// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/mismatched_brackets.h"

#include <algorithm>
#include <cmath>
#include <queue>

#include "common/check.h"
#include "llvm/ADT/DenseMap.h"

namespace Carbon::Lex {
namespace {

// Maximum number of collapsed items in a damaged region before falling back to
// naive greedy recovery.
constexpr int32_t MaxRegionItemsForSearch = 40;

// Maximum number of state expansions in Dijkstra search before falling back.
constexpr int32_t MaxSearchExpansions = 500;

// Maximum stack depth allowed during search before capping.
constexpr size_t MaxSearchStackDepth = 8;

// Cost penalties for bracket recovery.
constexpr int32_t CostBaselineMatch = 0;
constexpr int32_t CostIndentMismatchMultiplier = 10;
constexpr int32_t CostScopeNotAtEol = 30;
constexpr int32_t CostDedentInsideScope = 40;
constexpr int32_t CostSemiInParen = 100;
constexpr int32_t CostScopeInParen = 100;

constexpr int32_t CostInsertScopeBraceAfterHeader = 25;
constexpr int32_t CostInsertScopeBraceIndented = 40;
constexpr int32_t CostInsertScopeBraceTopLevel = 200;
constexpr int32_t CostInsertCloseBrace = 25;
constexpr int32_t CostInsertParenOrBracket = 160;
constexpr int32_t CostReplaceWithError = 150;
constexpr int32_t CostUnclosedOpenerAtEnd = 150;

// Internal representation of an item after clean subrange collapsing.
struct Item {
  int32_t token_start_index;
  int32_t token_end_index;
  bool is_collapsed_block = false;
  bool contains_scope_brace = false;
  MismatchedBracketToken token;
  int32_t effective_header_indent = 0;
  bool is_continuation = false;
  bool is_first_on_line = false;
  bool follows_statement_header = false;
};

// Represents an unclosed opening bracket on the search stack.
struct OpenBracketInfo {
  TokenIndex token_index = TokenIndex::None;
  BracketTokenKind kind;
  int32_t effective_header_indent;
  int32_t expected_body_indent;
  bool is_synthetic;
  TokenIndex insertion_token_index = TokenIndex::None;
};

// Compactly pack item index and stack into a 64-bit integer for DenseMap.
static auto PackState(int32_t item_index, llvm::ArrayRef<OpenBracketInfo> stack)
    -> uint64_t {
  uint64_t key = static_cast<uint16_t>(item_index);
  uint32_t depth = std::min<uint32_t>(stack.size(), 6);
  key |= (static_cast<uint64_t>(depth) << 16);

  uint64_t stack_bits = 0;
  for (uint32_t i = 0; i < depth; ++i) {
    uint64_t k = static_cast<uint64_t>(stack[i].kind) & 0x7;
    uint64_t ind =
        (static_cast<uint64_t>(stack[i].effective_header_indent / 2) & 0x7);
    uint64_t synth = stack[i].is_synthetic ? 1 : 0;
    uint64_t entry = (k << 4) | (ind << 1) | synth;
    stack_bits |= (entry << (i * 7));
  }
  key |= (stack_bits << 20);
  return key;
}

// Node in the search tree.
struct SearchNode {
  int32_t item_index;
  llvm::SmallVector<OpenBracketInfo, 4> stack;
  int32_t cost;
  int32_t parent_node_index;
  BracketCorrection correction;
  bool has_correction = false;
};

// Solve a damaged region using the simple greedy fallback algorithm.
auto SolveNaive(llvm::ArrayRef<Item> items,
                llvm::SmallVectorImpl<BracketCorrection>& corrections) -> void {
  llvm::SmallVector<Item> open_stack;
  for (const auto& item : items) {
    if (item.is_collapsed_block) {
      continue;
    }
    auto kind = item.token.kind;
    if (IsOpeningBracket(kind)) {
      open_stack.push_back(item);
    } else if (IsClosingBracket(kind)) {
      // Look back at most 16 unclosed opening brackets on the stack.
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
        corrections.push_back({
            .diagnostic_kind = BracketDiagnosticKind::UnmatchedClosing,
            .diagnostic_token_index = item.token.token_index,
            .fix_action = BracketFixAction::ReplaceWithError,
            .fix_token_index = item.token.token_index,
            .fix_token_kind = ToTokenKind(kind),
        });
      } else {
        for (auto it = search_range.begin(); it != match_it; ++it) {
          corrections.push_back({
              .diagnostic_kind = BracketDiagnosticKind::UnmatchedOpening,
              .diagnostic_token_index = it->token.token_index,
              .fix_action = BracketFixAction::InsertBefore,
              .fix_token_index = item.token.token_index,
              .fix_token_kind =
                  ToTokenKind(MatchingClosingKind(it->token.kind)),
          });
        }
        open_stack.erase(match_it.base() - 1, open_stack.end());
      }
    }
  }
  for (const auto& open : open_stack) {
    corrections.push_back({
        .diagnostic_kind = BracketDiagnosticKind::UnmatchedOpening,
        .diagnostic_token_index = open.token.token_index,
        .fix_action = BracketFixAction::ReplaceWithError,
        .fix_token_index = open.token.token_index,
        .fix_token_kind = ToTokenKind(open.token.kind),
    });
  }
}

// Reconstructs corrections from the goal node back to root.
auto ReconstructPath(llvm::ArrayRef<SearchNode> nodes, int32_t goal_index,
                     llvm::SmallVectorImpl<BracketCorrection>& corrections)
    -> void {
  llvm::SmallVector<int32_t> path;
  for (int32_t curr = goal_index; curr != -1;
       curr = nodes[curr].parent_node_index) {
    path.push_back(curr);
  }
  std::reverse(path.begin(), path.end());

  for (int32_t idx : path) {
    const auto& node = nodes[idx];
    if (node.has_correction) {
      corrections.push_back(node.correction);
    }
  }
}

// Solve a damaged region using Dijkstra shortest-path search.
auto SolveRegionCostBased(llvm::ArrayRef<Item> items,
                          llvm::SmallVectorImpl<BracketCorrection>& corrections)
    -> void {
  if (items.size() > static_cast<size_t>(MaxRegionItemsForSearch)) {
    SolveNaive(items, corrections);
    return;
  }

  llvm::SmallVector<SearchNode> nodes;
  nodes.reserve(MaxSearchExpansions);

  // Min-heap ordered by cost.
  using QueueElem = std::pair<int32_t /*cost*/, int32_t /*node_index*/>;
  std::priority_queue<QueueElem, std::vector<QueueElem>, std::greater<>> queue;

  llvm::DenseMap<uint64_t, int32_t> visited_min_cost;

  // Root node at item 0 with empty stack.
  nodes.push_back(SearchNode{
      .item_index = 0,
      .stack = {},
      .cost = 0,
      .parent_node_index = -1,
  });
  queue.push({0, 0});
  visited_min_cost[PackState(0, {})] = 0;

  int32_t expansion_count = 0;
  int32_t best_goal_node = -1;

  while (!queue.empty()) {
    auto [cost, node_idx] = queue.top();
    queue.pop();

    if (++expansion_count > MaxSearchExpansions) {
      SolveNaive(items, corrections);
      return;
    }

    const SearchNode current = nodes[node_idx];
    uint64_t state_key = PackState(current.item_index, current.stack);
    auto visited_it = visited_min_cost.find(state_key);
    if (visited_it != visited_min_cost.end() && cost > visited_it->second) {
      continue;
    }

    // Check if we reached the end of items in this region.
    if (current.item_index == static_cast<int32_t>(items.size())) {
      if (current.stack.empty()) {
        best_goal_node = node_idx;
        break;
      }

      int32_t finish_cost = current.cost;
      int32_t parent = node_idx;
      for (const auto& entry : llvm::reverse(current.stack)) {
        if (!entry.is_synthetic) {
          finish_cost += CostUnclosedOpenerAtEnd;
          nodes.push_back(SearchNode{
              .item_index = static_cast<int32_t>(items.size()),
              .stack = {},
              .cost = finish_cost,
              .parent_node_index = parent,
              .correction =
                  BracketCorrection{
                      .diagnostic_kind =
                          BracketDiagnosticKind::UnmatchedOpening,
                      .diagnostic_token_index = entry.token_index,
                      .fix_action = BracketFixAction::ReplaceWithError,
                      .fix_token_index = entry.token_index,
                      .fix_token_kind = ToTokenKind(entry.kind)},
              .has_correction = true,
          });
          parent = static_cast<int32_t>(nodes.size() - 1);
        }
      }

      best_goal_node = parent;
      break;
    }

    const auto& item = items[current.item_index];

    auto try_enqueue = [&](int32_t next_item_idx,
                           llvm::SmallVector<OpenBracketInfo, 4> next_stack,
                           int32_t add_cost, BracketCorrection correction = {},
                           bool has_correction = false) {
      int32_t next_cost = current.cost + add_cost;
      uint64_t key = PackState(next_item_idx, next_stack);

      auto it = visited_min_cost.find(key);
      if (it != visited_min_cost.end() && it->second <= next_cost) {
        return;
      }
      visited_min_cost[key] = next_cost;

      auto new_idx = static_cast<int32_t>(nodes.size());
      nodes.push_back(SearchNode{
          .item_index = next_item_idx,
          .stack = std::move(next_stack),
          .cost = next_cost,
          .parent_node_index = node_idx,
          .correction = correction,
          .has_correction = has_correction,
      });
      queue.push({next_cost, new_idx});
    };

    if (item.is_collapsed_block) {
      int32_t penalty = CostBaselineMatch;
      if (!current.stack.empty() &&
          (current.stack.back().kind == BracketTokenKind::OpenParen ||
           current.stack.back().kind == BracketTokenKind::OpenSquareBracket) &&
          item.contains_scope_brace) {
        penalty += CostScopeInParen;
      }
      try_enqueue(current.item_index + 1, current.stack, penalty);
      continue;
    }

    auto kind = item.token.kind;

    if (kind == BracketTokenKind::Semi ||
        kind == BracketTokenKind::StatementIntroducer ||
        kind == BracketTokenKind::Other) {
      int32_t penalty = CostBaselineMatch;
      if (kind == BracketTokenKind::Semi && !current.stack.empty() &&
          (current.stack.back().kind == BracketTokenKind::OpenParen ||
           current.stack.back().kind == BracketTokenKind::OpenSquareBracket)) {
        penalty += CostSemiInParen;
      }

      if (!current.stack.empty() &&
          current.stack.back().kind == BracketTokenKind::OpenCurlyBrace &&
          item.token.line_indent < current.stack.back().expected_body_indent &&
          item.token.line_indent > 0) {
        penalty += CostDedentInsideScope;
      }

      // Transition 1: Normal advance past non-bracket token.
      try_enqueue(current.item_index + 1, current.stack, penalty);

      // Transition 2: Candidate synthetic insertion of `{` before this token.
      if ((item.is_first_on_line || item.follows_statement_header ||
           kind == BracketTokenKind::StatementIntroducer) &&
          current.stack.size() < MaxSearchStackDepth) {
        int32_t insert_cost = CostInsertScopeBraceTopLevel;
        if (item.follows_statement_header) {
          insert_cost = CostInsertScopeBraceAfterHeader;
        } else if (item.token.line_indent > item.effective_header_indent) {
          insert_cost = CostInsertScopeBraceIndented;
        }

        auto next_stack = current.stack;
        next_stack.push_back(OpenBracketInfo{
            .token_index = TokenIndex::None,
            .kind = BracketTokenKind::OpenCurlyBrace,
            .effective_header_indent = item.effective_header_indent,
            .expected_body_indent = item.effective_header_indent + 2,
            .is_synthetic = true,
            .insertion_token_index = item.token.token_index,
        });
        try_enqueue(current.item_index, std::move(next_stack), insert_cost);
      }

      // Transition 3: Candidate synthetic insertion of `}` before this token.
      if (!current.stack.empty() &&
          current.stack.back().kind == BracketTokenKind::OpenCurlyBrace &&
          item.is_first_on_line &&
          item.token.line_indent <=
              current.stack.back().effective_header_indent) {
        auto next_stack = current.stack;
        auto popped = next_stack.pop_back_val();
        BracketCorrection correction;
        bool has_corr = false;
        if (!popped.is_synthetic) {
          correction = BracketCorrection{
              .diagnostic_kind = BracketDiagnosticKind::UnmatchedOpening,
              .diagnostic_token_index = popped.token_index,
              .fix_action = BracketFixAction::InsertBefore,
              .fix_token_index = item.token.token_index,
              .fix_token_kind = TokenKind::CloseCurlyBrace,
          };
          has_corr = true;
        }
        try_enqueue(current.item_index, std::move(next_stack),
                    CostInsertCloseBrace, correction, has_corr);
      }
      continue;
    }

    if (IsOpeningBracket(kind)) {
      // Transition C1: Push opening bracket to stack.
      if (current.stack.size() < MaxSearchStackDepth) {
        auto next_stack = current.stack;
        int32_t header_indent = item.effective_header_indent;
        int32_t body_indent = item.token.is_struct_brace
                                  ? item.token.line_indent
                                  : header_indent + 2;
        int32_t penalty = CostBaselineMatch;
        if (kind == BracketTokenKind::OpenCurlyBrace &&
            !item.token.is_at_end_of_line && !item.token.is_struct_brace) {
          penalty += CostScopeNotAtEol;
        }
        next_stack.push_back(OpenBracketInfo{
            .token_index = item.token.token_index,
            .kind = kind,
            .effective_header_indent = header_indent,
            .expected_body_indent = body_indent,
            .is_synthetic = false,
        });
        try_enqueue(current.item_index + 1, std::move(next_stack), penalty);
      }

      // Transition C2: Replace unmatched opening bracket with Error.
      try_enqueue(
          current.item_index + 1, current.stack, CostReplaceWithError,
          BracketCorrection{
              .diagnostic_kind = BracketDiagnosticKind::UnmatchedOpening,
              .diagnostic_token_index = item.token.token_index,
              .fix_action = BracketFixAction::ReplaceWithError,
              .fix_token_index = item.token.token_index,
              .fix_token_kind = ToTokenKind(kind)},
          /*has_correction=*/true);
      continue;
    }

    if (IsClosingBracket(kind)) {
      // Transition D1: Match with stack top.
      if (!current.stack.empty() &&
          current.stack.back().kind == MatchingOpeningKind(kind)) {
        auto next_stack = current.stack;
        auto popped = next_stack.pop_back_val();
        int32_t penalty = CostBaselineMatch;
        if (kind == BracketTokenKind::CloseCurlyBrace) {
          penalty += std::abs(popped.effective_header_indent -
                              item.token.line_indent) *
                     CostIndentMismatchMultiplier;
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
          };
          has_corr = true;
        }
        try_enqueue(current.item_index + 1, std::move(next_stack), penalty,
                    correction, has_corr);
      }

      // Transition D2: Synthesize missing opener before this closing bracket.
      if (current.stack.empty() ||
          current.stack.back().kind != MatchingOpeningKind(kind)) {
        auto opener_kind = MatchingOpeningKind(kind);
        int32_t opener_cost = (kind == BracketTokenKind::CloseCurlyBrace)
                                  ? CostInsertScopeBraceTopLevel
                                  : CostInsertParenOrBracket;
        try_enqueue(
            current.item_index + 1, current.stack, opener_cost,
            BracketCorrection{
                .diagnostic_kind = BracketDiagnosticKind::UnmatchedClosing,
                .diagnostic_token_index = item.token.token_index,
                .fix_action = BracketFixAction::InsertBefore,
                .fix_token_index = item.token.token_index,
                .fix_token_kind = ToTokenKind(opener_kind)},
            /*has_correction=*/true);
      }

      // Transition D3: Synthesize closer before this token to close stack top.
      if (!current.stack.empty()) {
        auto next_stack = current.stack;
        auto popped = next_stack.pop_back_val();
        auto closer_kind = MatchingClosingKind(popped.kind);
        BracketCorrection correction;
        bool has_corr = false;
        if (!popped.is_synthetic) {
          correction = BracketCorrection{
              .diagnostic_kind = BracketDiagnosticKind::UnmatchedOpening,
              .diagnostic_token_index = popped.token_index,
              .fix_action = BracketFixAction::InsertBefore,
              .fix_token_index = item.token.token_index,
              .fix_token_kind = ToTokenKind(closer_kind),
          };
          has_corr = true;
        }
        try_enqueue(current.item_index, std::move(next_stack),
                    CostInsertCloseBrace, correction, has_corr);
      }

      // Transition D4: Replace unmatched closing bracket with Error.
      try_enqueue(
          current.item_index + 1, current.stack, CostReplaceWithError,
          BracketCorrection{
              .diagnostic_kind = BracketDiagnosticKind::UnmatchedClosing,
              .diagnostic_token_index = item.token.token_index,
              .fix_action = BracketFixAction::ReplaceWithError,
              .fix_token_index = item.token.token_index,
              .fix_token_kind = ToTokenKind(kind)},
          /*has_correction=*/true);
      continue;
    }
  }

  if (best_goal_node != -1) {
    ReconstructPath(nodes, best_goal_node, corrections);
  } else {
    SolveNaive(items, corrections);
  }
}

}  // namespace

auto FixMismatchedBrackets(llvm::ArrayRef<MismatchedBracketToken> tokens)
    -> llvm::SmallVector<BracketCorrection> {
  llvm::SmallVector<BracketCorrection> corrections;
  if (tokens.empty()) {
    return corrections;
  }

  auto num_tokens = static_cast<int32_t>(tokens.size());

  // 1. Pre-pass: Compute statement introducers and effective header
  // indentations.
  llvm::SmallVector<int32_t> effective_header_indent(num_tokens, 0);
  llvm::SmallVector<bool> is_continuation(num_tokens, false);
  llvm::SmallVector<bool> is_first_on_line(num_tokens, false);
  llvm::SmallVector<bool> follows_statement_header(num_tokens, false);

  int32_t stmt_start_line = tokens[0].line;
  int32_t stmt_start_indent = tokens[0].line_indent;
  bool prev_was_header_close = false;

  for (int32_t i = 0; i < num_tokens; ++i) {
    is_first_on_line[i] = (i == 0 || tokens[i].line != tokens[i - 1].line);
    follows_statement_header[i] = prev_was_header_close;

    if (i > 0) {
      auto prev_kind = tokens[i - 1].kind;
      if (prev_kind == BracketTokenKind::Semi ||
          prev_kind == BracketTokenKind::CloseCurlyBrace ||
          prev_kind == BracketTokenKind::OpenCurlyBrace ||
          (tokens[i].kind == BracketTokenKind::StatementIntroducer &&
           tokens[i].line != tokens[i - 1].line)) {
        stmt_start_line = tokens[i].line;
        stmt_start_indent = tokens[i].line_indent;
      }
    }
    effective_header_indent[i] = stmt_start_indent;
    is_continuation[i] = (tokens[i].line != stmt_start_line);

    prev_was_header_close =
        (tokens[i].kind == BracketTokenKind::CloseParen ||
         tokens[i].kind == BracketTokenKind::CloseSquareBracket);
  }

  // 2. Clean subrange collapsing.
  llvm::SmallVector<int32_t> open_stack;
  llvm::SmallVector<int32_t> match_partner(num_tokens, -1);
  llvm::SmallVector<bool> is_clean_range(num_tokens, false);

  for (int32_t i = 0; i < num_tokens; ++i) {
    auto kind = tokens[i].kind;
    if (IsOpeningBracket(kind)) {
      open_stack.push_back(i);
    } else if (IsClosingBracket(kind)) {
      if (!open_stack.empty() &&
          MatchingClosingKind(tokens[open_stack.back()].kind) == kind) {
        int32_t open_idx = open_stack.pop_back_val();
        match_partner[open_idx] = i;
        match_partner[i] = open_idx;

        bool clean = true;
        if (tokens[open_idx].kind == BracketTokenKind::OpenCurlyBrace) {
          if (!tokens[open_idx].is_struct_brace &&
              tokens[open_idx].line != tokens[i].line &&
              (!tokens[open_idx].is_at_end_of_line ||
               effective_header_indent[open_idx] != tokens[i].line_indent)) {
            clean = false;
          }
        } else {
          // For () and [], check if there are top-level semicolons or scopes.
          for (int32_t j = open_idx + 1; j < i; ++j) {
            if (tokens[j].kind == BracketTokenKind::Semi ||
                tokens[j].kind == BracketTokenKind::OpenCurlyBrace) {
              clean = false;
              break;
            }
          }
        }

        if (clean) {
          is_clean_range[open_idx] = true;
        }
      } else {
        open_stack.clear();
      }
    }
  }

  // 3. Build item sequence.
  llvm::SmallVector<Item> items;
  for (int32_t i = 0; i < num_tokens;) {
    if (is_clean_range[i] && match_partner[i] != -1) {
      int32_t close_idx = match_partner[i];
      bool has_scope = false;
      for (int32_t j = i; j <= close_idx; ++j) {
        if (tokens[j].kind == BracketTokenKind::OpenCurlyBrace) {
          has_scope = true;
          break;
        }
      }
      items.push_back(Item{
          .token_start_index = i,
          .token_end_index = close_idx,
          .is_collapsed_block = true,
          .contains_scope_brace = has_scope,
          .token = tokens[i],
          .effective_header_indent = effective_header_indent[i],
          .is_continuation = is_continuation[i],
          .is_first_on_line = is_first_on_line[i],
          .follows_statement_header = follows_statement_header[i],
      });
      i = close_idx + 1;
    } else {
      items.push_back(Item{
          .token_start_index = i,
          .token_end_index = i,
          .is_collapsed_block = false,
          .contains_scope_brace =
              (tokens[i].kind == BracketTokenKind::OpenCurlyBrace ||
               tokens[i].kind == BracketTokenKind::CloseCurlyBrace),
          .token = tokens[i],
          .effective_header_indent = effective_header_indent[i],
          .is_continuation = is_continuation[i],
          .is_first_on_line = is_first_on_line[i],
          .follows_statement_header = follows_statement_header[i],
      });
      ++i;
    }
  }

  // 4. Partition items into damaged regions and solve.
  llvm::SmallVector<int32_t> region_boundaries;
  region_boundaries.push_back(0);

  llvm::SmallVector<BracketTokenKind> region_stack;
  for (int32_t i = 0; i < static_cast<int32_t>(items.size()); ++i) {
    if (!items[i].is_collapsed_block) {
      auto k = items[i].token.kind;
      if (IsOpeningBracket(k)) {
        region_stack.push_back(k);
      } else if (IsClosingBracket(k) && !region_stack.empty() &&
                 MatchingClosingKind(region_stack.back()) == k) {
        region_stack.pop_back();
      }
    }

    // Partition before a top-level statement introducer when stack is empty,
    // or after a clean top-level block containing scope braces.
    if (region_stack.empty()) {
      if (items[i].is_collapsed_block && items[i].contains_scope_brace) {
        region_boundaries.push_back(i + 1);
      } else if (i > 0 &&
                 items[i].token.kind == BracketTokenKind::StatementIntroducer &&
                 (items[i].token.line_indent == 0 ||
                  items[i].is_first_on_line)) {
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

    bool has_error = false;
    for (int32_t i = start; i < end; ++i) {
      if (!items[i].is_collapsed_block) {
        auto k = items[i].token.kind;
        if (IsOpeningBracket(k) || IsClosingBracket(k)) {
          has_error = true;
          break;
        }
      }
    }

    if (has_error) {
      auto slice = llvm::ArrayRef<Item>(items).slice(start, end - start);
      SolveRegionCostBased(slice, corrections);
    }
  }

  return corrections;
}

}  // namespace Carbon::Lex

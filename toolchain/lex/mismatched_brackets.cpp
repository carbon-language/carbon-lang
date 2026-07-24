// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/mismatched_brackets.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <queue>

#include "common/check.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"

namespace Carbon::Lex {
namespace {

// Maximum number of collapsed items in a damaged region before falling back to
// naive greedy recovery.
constexpr int32_t MaxRegionItemsForSearch = 40;

// Layered beam search width limit.
constexpr size_t MaxBeamWidth = 64;

// Maximum stack depth allowed during search before capping.
constexpr size_t MaxSearchStackDepth = 8;

// Cost penalties for bracket recovery.
constexpr int32_t CostBaselineMatch = 0;
constexpr int32_t CostIndentMismatchMultiplier = 20;
constexpr int32_t CostIndentMismatchBase = 30;
constexpr int32_t CostScopeNotAtEol = 30;
constexpr int32_t CostDedentInsideScope = 40;
constexpr int32_t CostSemiInParen = 100;
constexpr int32_t CostScopeInParen = 100;

constexpr int32_t CostInsertScopeBraceBeforeIntroducer = 20;
constexpr int32_t CostInsertCloseBrace = 25;
constexpr int32_t CostReplaceUnmatchedClosing = 30;
constexpr int32_t CostInsertParenOrBracket = 40;
constexpr int32_t CostUnclosedOpenerAtEnd = 50;
constexpr int32_t CostUnclosedParenAtEnd = 200;
constexpr int32_t CostHighBaselinePenalty = 80;
constexpr int32_t CostReplaceUnmatchedOpening = 100;

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
  bool is_before_statement_introducer = false;
};

// Represents an unclosed opening bracket on the search stack.
struct OpenBracketInfo {
  TokenIndex token_index = TokenIndex::None;
  BracketTokenKind kind;
  int32_t line = -1;
  int32_t effective_header_indent;
  int32_t expected_body_indent;
  bool is_synthetic;
  bool is_at_end_of_line = false;
  bool is_struct_brace = false;
  TokenIndex insertion_token_index = TokenIndex::None;
  bool is_insert_after = false;
  int32_t byte_offset = 0;
  int32_t insertion_byte_offset = 0;

  friend auto operator==(const OpenBracketInfo& a, const OpenBracketInfo& b)
      -> bool {
    return a.token_index == b.token_index && a.kind == b.kind &&
           a.line == b.line &&
           a.effective_header_indent == b.effective_header_indent &&
           a.expected_body_indent == b.expected_body_indent &&
           a.is_synthetic == b.is_synthetic &&
           a.is_at_end_of_line == b.is_at_end_of_line &&
           a.is_struct_brace == b.is_struct_brace &&
           a.insertion_token_index == b.insertion_token_index &&
           a.is_insert_after == b.is_insert_after &&
           a.byte_offset == b.byte_offset &&
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

// Determines if a token follows a statement/declaration header.
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
  if (curr_kind == BracketTokenKind::Other &&
      tokens[token_index].line == tokens[token_index - 1].line) {
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
        if (tokens[token_index].line == tokens[j].line ||
            tokens[token_index].line_indent <= tokens[j].line_indent) {
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
  llvm::SmallVector<ParentEdge, 2> parent_edges;
};

// Solve a damaged region using the simple greedy fallback algorithm.
auto SolveNaive(llvm::ArrayRef<Item> items, TokenIndex /*region_end_token*/,
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
        corrections.push_back({
            .diagnostic_kind = BracketDiagnosticKind::UnmatchedOpening,
            .diagnostic_token_index = open.token.token_index,
            .fix_action = BracketFixAction::ReplaceWithError,
            .fix_token_index = open.token.token_index,
            .fix_token_kind = ToTokenKind(open.token.kind),
            .origin = "Naive_UnclosedParenBracket",
        });
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
        corrections.push_back({
            .diagnostic_kind = BracketDiagnosticKind::UnmatchedClosing,
            .diagnostic_token_index = item.token.token_index,
            .fix_action = BracketFixAction::ReplaceWithError,
            .fix_token_index = item.token.token_index,
            .fix_token_kind = ToTokenKind(kind),
            .origin = "Naive_UnmatchedClosing",
        });
      } else {
        for (auto it = search_range.begin(); it != match_it; ++it) {
          corrections.push_back({
              .diagnostic_kind = BracketDiagnosticKind::UnmatchedOpening,
              .diagnostic_token_index = it->token.token_index,
              .fix_action = BracketFixAction::ReplaceWithError,
              .fix_token_index = it->token.token_index,
              .fix_token_kind = ToTokenKind(it->token.kind),
              .origin = "Naive_PoppedOpener",
          });
        }
        open_stack.erase(match_it.base() - 1, open_stack.end());
      }
    }
  }

  for (const auto& open : llvm::reverse(open_stack)) {
    corrections.push_back({
        .diagnostic_kind = BracketDiagnosticKind::UnmatchedOpening,
        .diagnostic_token_index = open.token.token_index,
        .fix_action = BracketFixAction::ReplaceWithError,
        .fix_token_index = open.token.token_index,
        .fix_token_kind = ToTokenKind(open.token.kind),
        .origin = "Naive_UnclosedAtEnd",
    });
  }
}

auto HashStack(llvm::ArrayRef<OpenBracketInfo> stack) -> uint64_t {
  uint64_t h = stack.size();
  for (const auto& info : stack) {
    uint64_t k =
        (static_cast<uint64_t>(info.token_index.index) << 32) ^
        (static_cast<uint64_t>(info.insertion_token_index.index) << 8) ^
        static_cast<uint64_t>(info.kind) ^
        (static_cast<uint64_t>(info.expected_body_indent) << 16);
    h ^= k + 0x9e3779b9 + (h << 6) + (h >> 2);
  }
  return h;
}

// Solve a damaged region using Dijkstra shortest-path search with tie
// detection.
auto SolveRegionCostBased(llvm::ArrayRef<Item> items,
                          TokenIndex region_end_token,
                          llvm::SmallVectorImpl<BracketCorrection>& corrections)
    -> void {
  if (items.size() > static_cast<size_t>(MaxRegionItemsForSearch)) {
    SolveNaive(items, region_end_token, corrections);
    return;
  }

  llvm::SmallVector<BeamNode, 0> arena;
  arena.reserve(256);

  int32_t min_goal_cost = std::numeric_limits<int32_t>::max();

  auto try_add_to_layer =
      [&](llvm::SmallVectorImpl<int32_t>& layer_indices,
          llvm::DenseMap<uint64_t, int32_t>& layer_map, int32_t next_item_idx,
          llvm::SmallVector<OpenBracketInfo, 4> next_stack, int32_t next_cost,
          ParentEdge edge, llvm::SmallVectorImpl<int32_t>* worklist = nullptr) {
        if (next_cost > min_goal_cost) {
          return;
        }
        uint64_t stack_hash = HashStack(next_stack);
        auto map_it = layer_map.find(stack_hash);
        if (map_it != layer_map.end()) {
          int32_t idx = map_it->second;
          auto& exist_node = arena[idx];
          if (exist_node.stack == next_stack) {
            if (next_cost < exist_node.cost) {
              exist_node.cost = next_cost;
              exist_node.parent_edges.clear();
              exist_node.parent_edges.push_back(edge);
              if (worklist) {
                worklist->push_back(idx);
              }
            } else if (next_cost == exist_node.cost) {
              bool duplicate = false;
              for (const auto& exist_edge : exist_node.parent_edges) {
                if (exist_edge.parent_node_index == edge.parent_node_index &&
                    exist_edge.has_correction == edge.has_correction &&
                    (!edge.has_correction ||
                     (exist_edge.correction.diagnostic_token_index ==
                          edge.correction.diagnostic_token_index &&
                      exist_edge.correction.fix_action ==
                          edge.correction.fix_action &&
                      exist_edge.correction.fix_token_index ==
                          edge.correction.fix_token_index &&
                      exist_edge.correction.fix_token_kind ==
                          edge.correction.fix_token_kind))) {
                  duplicate = true;
                  break;
                }
              }
              if (!duplicate) {
                exist_node.parent_edges.push_back(edge);
              }
            }
            return;
          }
          // On hash collision, fall through to linear scan over layer_indices.
          for (int32_t idx : layer_indices) {
            auto& exist_node = arena[idx];
            if (exist_node.stack == next_stack) {
              if (next_cost < exist_node.cost) {
                exist_node.cost = next_cost;
                exist_node.parent_edges.clear();
                exist_node.parent_edges.push_back(edge);
                if (worklist) {
                  worklist->push_back(idx);
                }
              } else if (next_cost == exist_node.cost) {
                bool duplicate = false;
                for (const auto& exist_edge : exist_node.parent_edges) {
                  if (exist_edge.parent_node_index == edge.parent_node_index &&
                      exist_edge.has_correction == edge.has_correction &&
                      (!edge.has_correction ||
                       (exist_edge.correction.diagnostic_token_index ==
                            edge.correction.diagnostic_token_index &&
                        exist_edge.correction.fix_action ==
                            edge.correction.fix_action &&
                        exist_edge.correction.fix_token_index ==
                            edge.correction.fix_token_index &&
                        exist_edge.correction.fix_token_kind ==
                            edge.correction.fix_token_kind))) {
                    duplicate = true;
                    break;
                  }
                }
                if (!duplicate) {
                  exist_node.parent_edges.push_back(edge);
                }
              }
              return;
            }
          }
        }
        int32_t new_idx = static_cast<int32_t>(arena.size());
        arena.push_back(BeamNode{
            .item_index = next_item_idx,
            .stack = std::move(next_stack),
            .cost = next_cost,
            .parent_edges = {edge},
        });
        layer_indices.push_back(new_idx);
        layer_map[stack_hash] = new_idx;
        if (worklist) {
          worklist->push_back(new_idx);
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

    // Step 1: Epsilon moves within layer `i` (without advancing `i`).
    if (kind != BracketTokenKind::FileEnd) {
      for (int32_t idx : current_layer) {
        layer_map[HashStack(arena[idx].stack)] = idx;
      }
      llvm::SmallVector<int32_t> worklist = current_layer;
      size_t worklist_head = 0;

      while (worklist_head < worklist.size()) {
        int32_t node_idx = worklist[worklist_head++];
        const BeamNode current = arena[node_idx];
        if (current.cost > min_goal_cost) {
          continue;
        }

        auto try_enqueue_epsilon =
            [&](llvm::SmallVector<OpenBracketInfo, 4> next_stack,
                int32_t add_cost, BracketCorrection correction = {},
                bool has_correction = false) {
              int32_t next_cost = current.cost + add_cost;
              ParentEdge edge{
                  .parent_node_index = node_idx,
                  .correction = correction,
                  .has_correction = has_correction,
              };
              try_add_to_layer(current_layer, layer_map, i,
                               std::move(next_stack), next_cost, edge,
                               &worklist);
            };

        // 1a. Insert Closer (pop top of stack by synthesizing matching closer
        // before token i).
        if (!current.stack.empty()) {
          auto popped = current.stack.back();
          if (kind != MatchingClosingKind(popped.kind) &&
              !(popped.is_synthetic &&
                popped.insertion_token_index == item.token.token_index)) {
            bool is_discounted = false;
            const char* origin_name = "Epsilon_InsertCloser";
            if (popped.kind == BracketTokenKind::OpenCurlyBrace &&
                !popped.is_struct_brace && item.is_first_on_line &&
                item.token.line_indent < popped.expected_body_indent) {
              bool is_dedent_transition = true;
              if (i > 0) {
                const auto& prev_item = items[i - 1];
                if (prev_item.token.kind == BracketTokenKind::CloseCurlyBrace ||
                    prev_item.token.line_indent <= item.token.line_indent) {
                  is_dedent_transition = false;
                }
              }
              if (is_dedent_transition) {
                is_discounted = true;
                origin_name = "T3_DedentCloseBrace";
              }
            }
            if (IsClosingBracket(kind)) {
              auto req_opener = MatchingOpeningKind(kind);
              for (size_t s = 0; s + 1 < current.stack.size(); ++s) {
                if (current.stack[s].kind == req_opener) {
                  is_discounted = true;
                  origin_name = "TD3_CloserMismatchClosesStackTop";
                  break;
                }
              }
            }
            if (!is_discounted &&
                (popped.kind == BracketTokenKind::OpenParen ||
                 popped.kind == BracketTokenKind::OpenSquareBracket ||
                 (popped.kind == BracketTokenKind::OpenCurlyBrace &&
                  popped.is_struct_brace))) {
              if (popped.line == item.token.line ||
                  item.token.line_indent >= popped.effective_header_indent) {
                is_discounted = true;
                origin_name = "T_SynthesizeCloserBeforeItem";
              }
            }

            if (!is_discounted) {
              continue;
            }

            int32_t eps_cost = CostInsertCloseBrace +
                               (is_discounted ? 0 : CostHighBaselinePenalty);

            auto next_stack = current.stack;
            next_stack.pop_back();
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
                  .fix_byte_offset = item.token.byte_offset,
                  .origin = origin_name,
              };
              has_corr = true;
            }
            try_enqueue_epsilon(std::move(next_stack), eps_cost, correction,
                                has_corr);
          }
        }
      }

      // Phase 1b: Epsilon Openers / Pushes.
      // Iterate only over the states present after Phase 1a (original +
      // popped), without chaining synthetic openers onto newly pushed openers.
      size_t num_states_after_1a = current_layer.size();
      for (size_t idx = 0; idx < num_states_after_1a; ++idx) {
        int32_t node_idx = current_layer[idx];
        const BeamNode current = arena[node_idx];
        if (current.cost > min_goal_cost) {
          continue;
        }

        auto try_enqueue_epsilon_opener =
            [&](llvm::SmallVector<OpenBracketInfo, 4> next_stack,
                int32_t add_cost) {
              int32_t next_cost = current.cost + add_cost;
              ParentEdge edge{
                  .parent_node_index = node_idx,
                  .correction = {},
                  .has_correction = false,
              };
              try_add_to_layer(current_layer, layer_map, i,
                               std::move(next_stack), next_cost, edge, nullptr);
            };

        if (current.stack.size() < MaxSearchStackDepth) {
          // Push synthetic '{'
          {
            bool is_discounted = item.follows_statement_header &&
                                 !item.header_has_open_curly_brace &&
                                 !IsOpeningBracket(kind);
            int32_t base_cost = item.is_before_statement_introducer
                                    ? CostInsertScopeBraceBeforeIntroducer
                                    : CostInsertCloseBrace;
            int32_t eps_cost =
                base_cost + (is_discounted ? 0 : CostHighBaselinePenalty);
            auto next_stack = current.stack;
            next_stack.push_back(OpenBracketInfo{
                .token_index = TokenIndex::None,
                .kind = BracketTokenKind::OpenCurlyBrace,
                .line = item.token.line,
                .effective_header_indent = item.effective_header_indent,
                .expected_body_indent = item.effective_header_indent + 2,
                .is_synthetic = true,
                .insertion_token_index = item.token.token_index,
                .byte_offset = item.token.byte_offset,
                .insertion_byte_offset = item.token.byte_offset,
            });
            try_enqueue_epsilon_opener(std::move(next_stack), eps_cost);
          }
          // Push synthetic '('
          {
            int32_t eps_cost =
                CostInsertParenOrBracket + CostHighBaselinePenalty;
            auto next_stack = current.stack;
            next_stack.push_back(OpenBracketInfo{
                .token_index = TokenIndex::None,
                .kind = BracketTokenKind::OpenParen,
                .line = item.token.line,
                .effective_header_indent = item.effective_header_indent,
                .expected_body_indent = item.effective_header_indent + 2,
                .is_synthetic = true,
                .insertion_token_index = item.token.token_index,
                .byte_offset = item.token.byte_offset,
                .insertion_byte_offset = item.token.byte_offset,
            });
            try_enqueue_epsilon_opener(std::move(next_stack), eps_cost);
          }
          // Push synthetic '['
          {
            int32_t eps_cost =
                CostInsertParenOrBracket + CostHighBaselinePenalty;
            auto next_stack = current.stack;
            next_stack.push_back(OpenBracketInfo{
                .token_index = TokenIndex::None,
                .kind = BracketTokenKind::OpenSquareBracket,
                .line = item.token.line,
                .effective_header_indent = item.effective_header_indent,
                .expected_body_indent = item.effective_header_indent + 2,
                .is_synthetic = true,
                .insertion_token_index = item.token.token_index,
                .byte_offset = item.token.byte_offset,
                .insertion_byte_offset = item.token.byte_offset,
            });
            try_enqueue_epsilon_opener(std::move(next_stack), eps_cost);
          }
        }
      }

      layer_map.clear();
      if (current_layer.size() > MaxBeamWidth) {
        llvm::stable_sort(
            current_layer,
            [&](const int32_t& a_idx, const int32_t& b_idx) -> bool {
              return arena[a_idx].cost < arena[b_idx].cost;
            });
        current_layer.resize(MaxBeamWidth);
      }
    }

    // Step 2: Advance moves from layer `i` to layer `i + 1` (consuming token
    // `i`).
    llvm::SmallVector<int32_t> next_layer;

    for (int32_t node_idx : current_layer) {
      const BeamNode current = arena[node_idx];
      if (current.cost > min_goal_cost) {
        continue;
      }

      auto try_enqueue_advance =
          [&](llvm::SmallVector<OpenBracketInfo, 4> next_stack,
              int32_t add_cost, BracketCorrection correction = {},
              bool has_correction = false) {
            int32_t next_cost = current.cost + add_cost;
            ParentEdge edge{
                .parent_node_index = node_idx,
                .correction = correction,
                .has_correction = has_correction,
            };
            try_add_to_layer(next_layer, layer_map, i + 1,
                             std::move(next_stack), next_cost, edge, nullptr);
          };

      if (item.is_collapsed_block) {
        int32_t penalty = CostBaselineMatch;
        if (!current.stack.empty() &&
            (current.stack.back().kind == BracketTokenKind::OpenParen ||
             current.stack.back().kind == BracketTokenKind::OpenSquareBracket ||
             current.stack.back().is_struct_brace) &&
            item.contains_scope_brace) {
          penalty += CostScopeInParen;
        }
        if (!current.stack.empty() &&
            current.stack.back().kind == BracketTokenKind::OpenCurlyBrace &&
            !current.stack.back().is_struct_brace &&
            item.token.line_indent <=
                current.stack.back().effective_header_indent) {
          penalty += CostDedentInsideScope;
        }
        try_enqueue_advance(current.stack, penalty);
        continue;
      }

      if (IsOpeningBracket(kind)) {
        // Advance and push opener onto stack.
        if (current.stack.size() < MaxSearchStackDepth) {
          auto next_stack = current.stack;
          int32_t header_indent = item.effective_header_indent;
          int32_t body_indent = item.token.is_struct_brace
                                    ? item.token.line_indent
                                    : header_indent + 2;
          int32_t penalty = CostBaselineMatch;
          if (kind == BracketTokenKind::OpenCurlyBrace &&
              !item.token.is_at_end_of_line && !item.token.is_struct_brace &&
              item.follows_statement_header) {
            penalty += CostScopeNotAtEol;
          }
          next_stack.push_back(OpenBracketInfo{
              .token_index = item.token.token_index,
              .kind = kind,
              .line = item.token.line,
              .effective_header_indent = header_indent,
              .expected_body_indent = body_indent,
              .is_synthetic = false,
              .is_at_end_of_line = item.token.is_at_end_of_line,
              .is_struct_brace = item.token.is_struct_brace,
              .byte_offset = item.token.byte_offset,
              .insertion_byte_offset = item.token.byte_offset,
          });
          try_enqueue_advance(std::move(next_stack), penalty);
        }

        // Advance without pushing (replace unmatched opener with Error token).
        try_enqueue_advance(
            current.stack, CostReplaceUnmatchedOpening,
            BracketCorrection{
                .diagnostic_kind = BracketDiagnosticKind::UnmatchedOpening,
                .diagnostic_token_index = item.token.token_index,
                .fix_action = BracketFixAction::ReplaceWithError,
                .fix_token_index = item.token.token_index,
                .fix_token_kind = ToTokenKind(kind),
                .fix_byte_offset = item.token.byte_offset,
                .origin = "Advance_ReplaceOpeningError"},
            /*has_correction=*/true);
        continue;
      }

      if (IsClosingBracket(kind)) {
        // If matches top(stack): advance and pop stack.
        if (!current.stack.empty() &&
            current.stack.back().kind == MatchingOpeningKind(kind)) {
          bool allow_match = true;
          if (kind == BracketTokenKind::CloseCurlyBrace) {
            if (item.token.line_indent <
                current.stack.back().effective_header_indent) {
              allow_match = false;
            }
            if (current.stack.back().is_struct_brace && item.is_first_on_line &&
                current.stack.size() >= 2 &&
                item.token.line_indent <=
                    current.stack[current.stack.size() - 2]
                        .effective_header_indent) {
              allow_match = false;
            }
          }
          if (allow_match) {
            auto next_stack = current.stack;
            auto popped = next_stack.pop_back_val();
            int32_t penalty = CostBaselineMatch;
            if (kind == BracketTokenKind::CloseCurlyBrace) {
              if (popped.effective_header_indent != item.token.line_indent) {
                penalty += CostIndentMismatchBase +
                           std::abs(popped.effective_header_indent -
                                    item.token.line_indent) *
                               CostIndentMismatchMultiplier;
              }
            }
            BracketCorrection correction;
            bool has_corr = false;
            if (popped.is_synthetic) {
              correction = BracketCorrection{
                  .diagnostic_kind = BracketDiagnosticKind::UnmatchedClosing,
                  .diagnostic_token_index = item.token.token_index,
                  .fix_action = popped.is_insert_after
                                    ? BracketFixAction::InsertAfter
                                    : BracketFixAction::InsertBefore,
                  .fix_token_index = popped.insertion_token_index,
                  .fix_token_kind = ToTokenKind(popped.kind),
                  .fix_byte_offset = popped.insertion_byte_offset,
                  .origin = "TD1_SyntheticOpenerMatched",
              };
              has_corr = true;
            }
            try_enqueue_advance(std::move(next_stack), penalty, correction,
                                has_corr);
          }
        }

        // Advance without matching/popping (replace unmatched closer with Error
        // token).
        try_enqueue_advance(
            current.stack, CostReplaceUnmatchedClosing,
            BracketCorrection{
                .diagnostic_kind = BracketDiagnosticKind::UnmatchedClosing,
                .diagnostic_token_index = item.token.token_index,
                .fix_action = BracketFixAction::ReplaceWithError,
                .fix_token_index = item.token.token_index,
                .fix_token_kind = ToTokenKind(kind),
                .fix_byte_offset = item.token.byte_offset,
                .origin = "Advance_ReplaceClosingError"},
            /*has_correction=*/true);
        continue;
      }

      // Non-bracket token (Semi, StatementIntroducer, FileEnd, Other).
      int32_t penalty = CostBaselineMatch;
      if (kind == BracketTokenKind::Semi && !current.stack.empty() &&
          (current.stack.back().kind == BracketTokenKind::OpenParen ||
           current.stack.back().kind == BracketTokenKind::OpenSquareBracket ||
           current.stack.back().is_struct_brace)) {
        penalty += CostSemiInParen;
      }
      if (!current.stack.empty() &&
          current.stack.back().kind == BracketTokenKind::OpenCurlyBrace &&
          !current.stack.back().is_struct_brace && item.is_first_on_line &&
          item.token.line_indent <=
              current.stack.back().effective_header_indent) {
        penalty += CostDedentInsideScope;
      }
      try_enqueue_advance(current.stack, penalty);
    }

    // Step 3: Beam Pruning.
    layer_map.clear();
    if (next_layer.size() > MaxBeamWidth) {
      llvm::stable_sort(
          next_layer, [&](const int32_t& a_idx, const int32_t& b_idx) -> bool {
            return arena[a_idx].cost < arena[b_idx].cost;
          });
      next_layer.resize(MaxBeamWidth);
    }
    current_layer = std::move(next_layer);
  }

  llvm::SmallVector<int32_t> goal_node_indices;

  for (int32_t node_idx : current_layer) {
    const BeamNode current = arena[node_idx];
    if (current.cost > min_goal_cost) {
      continue;
    }
    if (current.stack.empty()) {
      if (current.cost < min_goal_cost) {
        min_goal_cost = current.cost;
        goal_node_indices.clear();
      }
      if (current.cost == min_goal_cost) {
        goal_node_indices.push_back(node_idx);
      }
      continue;
    }

    int32_t finish_cost = current.cost;
    int32_t parent = node_idx;
    for (const auto& entry : llvm::reverse(current.stack)) {
      if (!entry.is_synthetic) {
        finish_cost += (entry.kind == BracketTokenKind::OpenCurlyBrace
                            ? CostUnclosedOpenerAtEnd
                            : CostUnclosedParenAtEnd);
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
                            ToTokenKind(MatchingClosingKind(entry.kind))},
                .has_correction = true,
            }},
        });
        parent = new_idx;
      }
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
    SolveNaive(items, region_end_token, corrections);
    return;
  }

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
    SolveNaive(items, region_end_token, corrections);
    return;
  }

  llvm::SmallVector<BracketCorrection> baseline_path = all_paths.front();

  for (auto& corr : baseline_path) {
    bool is_tied = false;
    for (const auto& path : all_paths) {
      const auto* it =
          std::find_if(path.begin(), path.end(), [&](const auto& c) {
            return c.diagnostic_token_index == corr.diagnostic_token_index;
          });
      if (it == path.end() || it->diagnostic_kind != corr.diagnostic_kind ||
          it->fix_action != corr.fix_action ||
          it->fix_token_index != corr.fix_token_index ||
          it->fix_token_kind != corr.fix_token_kind) {
        is_tied = true;
        break;
      }
    }
    corr.is_tied = is_tied;
    corrections.push_back(corr);
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

  // 1. Initial pass to find clean matched bracket pairs.
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
        if (kind == BracketTokenKind::CloseCurlyBrace) {
          int32_t match_idx = -1;
          for (int32_t s = static_cast<int32_t>(open_stack.size()) - 1; s >= 0;
               --s) {
            int32_t cand = open_stack[s];
            if (tokens[cand].kind != BracketTokenKind::OpenCurlyBrace) {
              break;
            }
            if (tokens[cand].line == tokens[i].line ||
                tokens[cand].is_struct_brace ||
                tokens[cand].line_indent == tokens[i].line_indent) {
              match_idx = s;
              break;
            }
          }
          if (match_idx != -1) {
            int32_t open_idx = open_stack[match_idx];
            open_stack.erase(open_stack.begin() + match_idx);
            match_partner[open_idx] = i;
            match_partner[i] = open_idx;
          }
        } else {
          int32_t open_idx = open_stack.pop_back_val();
          match_partner[open_idx] = i;
          match_partner[i] = open_idx;
        }
      } else {
        open_stack.clear();
      }
    }
  }

  // Identify unmatched openers.
  llvm::SmallVector<bool> is_unmatched_opener(num_tokens, false);
  for (int32_t i = 0; i < num_tokens; ++i) {
    if (IsOpeningBracket(tokens[i].kind) && match_partner[i] == -1) {
      is_unmatched_opener[i] = true;
    }
  }

  // 2. Pre-pass: Compute associated indentation, first-on-line, and header
  // follows using backward scan.
  llvm::SmallVector<int32_t> effective_header_indent(num_tokens, 0);
  llvm::SmallVector<bool> is_first_on_line(num_tokens, false);
  llvm::SmallVector<bool> follows_statement_header(num_tokens, false);
  llvm::SmallVector<bool> header_has_open_curly_brace(num_tokens, false);
  llvm::SmallVector<bool> is_before_statement_introducer(num_tokens, false);

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
    if (tokens[i].kind == BracketTokenKind::StatementIntroducer && i > 0) {
      auto prev_kind = tokens[i - 1].kind;
      if (prev_kind != BracketTokenKind::OpenCurlyBrace) {
        if (prev_kind != BracketTokenKind::StatementIntroducer ||
            (tokens[i].line != tokens[i - 1].line &&
             tokens[i].line_indent > tokens[i - 1].line_indent)) {
          is_before_statement_introducer[i] = true;
        }
      }
    }
  }

  // 3. Mark clean subranges for safe collapsing (processed in reverse order
  // so inner ranges are evaluated before enclosing outer ranges).
  for (int32_t i = num_tokens - 1; i >= 0; --i) {
    auto kind = tokens[i].kind;
    if (match_partner[i] != -1 && match_partner[i] > i) {
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
      } else {
        // For parens/brackets, check if an unclosed opener of the same kind
        // precedes i in the active scope.
        for (int32_t u = 0; u < i; ++u) {
          if (is_unmatched_opener[u] && tokens[u].kind == kind) {
            clean = false;
            break;
          }
        }
      }

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
  }

  // 4. Build item sequence.
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
          .is_first_on_line = is_first_on_line[i],
          .follows_statement_header = follows_statement_header[i],
          .header_has_open_curly_brace = header_has_open_curly_brace[i],
          .is_before_statement_introducer = is_before_statement_introducer[i],
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
          .is_first_on_line = is_first_on_line[i],
          .follows_statement_header = follows_statement_header[i],
          .header_has_open_curly_brace = header_has_open_curly_brace[i],
          .is_before_statement_introducer = is_before_statement_introducer[i],
      });
      ++i;
    }
  }

  // 5. Partition items into damaged regions and solve.
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

    if (region_stack.empty()) {
      if (items[i].is_collapsed_block && items[i].contains_scope_brace) {
        region_boundaries.push_back(i + 1);
      } else if (i > 0 &&
                 items[i].token.kind == BracketTokenKind::StatementIntroducer &&
                 items[i].token.line_indent == 0) {
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

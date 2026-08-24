// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/language_server/sem_ir_index.h"

#include "common/check.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::LanguageServer {

auto GetNameToken(const Parse::TreeAndSubtrees& tree_and_subtrees,
                  Parse::NodeId node_id) -> Lex::TokenIndex {
  const auto& tree = tree_and_subtrees.tree();
  const auto& tokens = tree.tokens();
  for (auto child : tree_and_subtrees.children(node_id)) {
    switch (tree.node_kind(child)) {
      case Parse::NodeKind::IdentifierNameMaybeBeforeSignature:
      case Parse::NodeKind::IdentifierNameNotBeforeSignature: {
        auto token = tree.node_token(child);
        if (tokens.GetKind(token) == Lex::TokenKind::Identifier) {
          return token;
        }
        break;
      }
      default:
        break;
    }
  }
  return tree.node_token(node_id);
}

// Returns the token that `inst_id` was checked from, or `None` if it has no
// location in this file. Instructions imported from another file are located by
// an `ImportIRInstId`, and desugared instructions by the instruction they were
// desugared from, so only a `NodeId` location refers to this file's tokens.
static auto GetTokenForInst(const SemIR::File& sem_ir,
                            const Parse::TreeAndSubtrees& tree_and_subtrees,
                            SemIR::InstId inst_id) -> Lex::TokenIndex {
  auto loc_id = sem_ir.insts().GetCanonicalLocId(inst_id);
  if (loc_id.kind() != SemIR::LocId::Kind::NodeId) {
    return Lex::TokenIndex::None;
  }
  auto node_id = loc_id.node_id();
  if (!node_id.has_value()) {
    return Lex::TokenIndex::None;
  }
  return GetNameToken(tree_and_subtrees, node_id);
}

SemIRIndex::SemIRIndex(const SemIR::File& sem_ir,
                       const Parse::TreeAndSubtrees& tree_and_subtrees) {
  const auto& tokens = tree_and_subtrees.tree().tokens();
  // Populate `token_starts_` in three in-place passes. Note that we
  // need N+1 elements to hold the boundaries of N contiguous intervals,
  // plus an additional temporary element for reasons discussed below.
  //
  // First, we count the instructions per token. The array contents are
  // shifted by 2: `token_starts[i+2]` will hold the number of insts for
  // the token with `.index == i`.
  token_starts_.assign(tokens.size() + 2, 0);
  for (auto [inst_id, inst] : sem_ir.insts().enumerate()) {
    auto token = GetTokenForInst(sem_ir, tree_and_subtrees, inst_id);
    if (!token.has_value()) {
      continue;
    }
    ++token_starts_[token.index + 2];
  }

  // Perform a prefix sum, so that `token_starts_[i+2]` holds the number
  // of insts for tokens with `.index <= i`, i.e. the end of the interval for
  // token `i`, and hence `token_starts_[i+1]` is the start of the interval
  // for token `i`.
  for (size_t i = 1; i < token_starts_.size(); ++i) {
    token_starts_[i] += token_starts_[i - 1];
  }

  // Populate `insts_`, using `token_starts_[i+1]` as the index to write the next
  // inst for token `i`, which is incremented on each write. Thus, at the end of
  // the loop, `token_starts_[i+1]` is the past-the-end index for token `i`,
  // i.e. the start index for token `i+1`, which is the final state of `token_starts_`.
  insts_.resize(token_starts_.back(), SemIR::InstId::None);
  for (auto [inst_id, inst] : sem_ir.insts().enumerate()) {
    auto token = GetTokenForInst(sem_ir, tree_and_subtrees, inst_id);
    if (!token.has_value()) {
      continue;
    }
    insts_[token_starts_[token.index + 1]++] = inst_id;
  }

  // Drop the spare trailing slot now that every cursor has moved into place.
  token_starts_.pop_back();
  CARBON_CHECK(static_cast<size_t>(token_starts_.back()) == insts_.size());
}

auto SemIRIndex::InstsForToken(Lex::TokenIndex token) const
    -> llvm::ArrayRef<SemIR::InstId> {
  if (!token.has_value()) {
    return {};
  }
  CARBON_CHECK(static_cast<size_t>(token.index) + 1 < token_starts_.size(),
               "Token {0} is not from the indexed file", token.index);
  int32_t start = token_starts_[token.index];
  int32_t end = token_starts_[token.index + 1];
  return llvm::ArrayRef(insts_).slice(start, end - start);
}

}  // namespace Carbon::LanguageServer

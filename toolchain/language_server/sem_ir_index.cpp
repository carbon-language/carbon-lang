// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/language_server/sem_ir_index.h"

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
                       const Parse::TreeAndSubtrees& tree_and_subtrees)
    : insts_(tree_and_subtrees.tree().tokens().size(), [&](auto add) {
        for (auto [inst_id, inst] : sem_ir.insts().enumerate()) {
          add(GetTokenForInst(sem_ir, tree_and_subtrees, inst_id), inst_id);
        }
      }) {}

}  // namespace Carbon::LanguageServer

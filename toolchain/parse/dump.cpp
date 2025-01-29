// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef NDEBUG

#include "toolchain/parse/dump.h"

#include "common/ostream.h"
#include "toolchain/lex/dump.h"

namespace Carbon::Parse {

auto DumpNoNewline(const Tree& tree, NodeId node_id) -> void {
  if (!node_id.has_value()) {
    llvm::errs() << "NodeId(invalid)";
    return;
  }

  auto kind = tree.node_kind(node_id);
  auto token = tree.node_token(node_id);

  llvm::errs() << "NodeId(kind: " << kind << ", token: ";
  Lex::DumpNoNewline(tree.tokens(), token);
  llvm::errs() << ")";
}

LLVM_DUMP_METHOD auto Dump(const Tree& tree, Lex::TokenIndex token) -> void {
  Lex::Dump(tree.tokens(), token);
}

LLVM_DUMP_METHOD auto Dump(const Tree& tree, NodeId node_id) -> void {
  DumpNoNewline(tree, node_id);
  llvm::errs() << '\n';
}

}  // namespace Carbon::Parse

#endif  // NDEBUG

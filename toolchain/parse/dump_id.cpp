// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef NDEBUG

#include "toolchain/parse/dump_id.h"

#include "common/ostream.h"
#include "toolchain/lex/dump_id.h"

namespace Carbon::Parse {

auto DumpIdImpl(const Tree& tree, NodeId node_id) -> void {
  if (!node_id.is_valid()) {
    llvm::errs() << "NodeId(invalid)";
    return;
  }

  auto kind = tree.node_kind(node_id);
  auto token = tree.node_token(node_id);

  llvm::errs() << "NodeId(kind: ";
  kind.Print(llvm::errs());
  llvm::errs() << ", token: ";
  Lex::DumpIdImpl(tree.tokens(), token);
  llvm::errs() << ")";
}

// A set of DumpId() overloads that dump an object to stderr, useful for
// calling inside a debugger.
LLVM_DUMP_METHOD auto DumpId(const Parse::Tree& tree, Lex::TokenIndex token)
    -> void {
  Lex::DumpId(tree.tokens(), token);
}

LLVM_DUMP_METHOD auto DumpId(const Parse::Tree& tree, Parse::NodeId node_id)
    -> void {
  DumpIdImpl(tree, node_id);
  llvm::errs() << '\n';
}

}  // namespace Carbon::Parse

#endif  // NDEBUG

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/dump_id.h"

#include "common/ostream.h"
#include "toolchain/lex/dump_id.h"
#include "toolchain/parse/tree.h"

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

}  // namespace Carbon::Parse

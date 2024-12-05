// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/dump_id.h"

#include "llvm/Support/raw_ostream.h"
#include "toolchain/lex/dump_id.h"
#include "toolchain/parse/tree.h"

namespace Carbon::Parse::DumpIdOverloads {

auto DumpId(NodeId node_id, const Tree& tree) -> void {
  if (!node_id.is_valid()) {
    llvm::errs() << "NodeId(invalid)";
    return;
  }

  auto kind = tree.node_kind(node_id);
  auto token = tree.node_token(node_id);

  llvm::errs() << "NodeId(kind: ";
  kind.Print(llvm::errs());
  llvm::errs() << ", token: ";
  Lex::DumpIdOverloads::DumpId(token, tree.tokens());
  llvm::errs() << ")";
}

}  // namespace Carbon::Parse::DumpIdOverloads

namespace Carbon::Parse {
template <>
auto DumpIdMethods<Tree>::Newline() const -> void {
  llvm::errs() << "\n";
}
}  // namespace Carbon::Parse

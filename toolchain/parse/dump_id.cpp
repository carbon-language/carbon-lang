// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/dump_id.h"

#include "llvm/Support/raw_ostream.h"
#include "toolchain/parse/tree.h"

namespace Carbon::Parse::DumpIdOverloads {

auto DumpId(NodeId node_id, const Tree& tree) -> void {
  if (!node_id.is_valid()) {
    llvm::errs() << "NodeId(invalid)";
  }

  llvm::errs() << "NodeId(";
  auto token = tree.node_token(node_id);
  llvm::errs() << "token = ";
  tree.DumpId(token);
  llvm::errs() << ")";

  llvm::errs() << "A node id";
}

}  // namespace Carbon::Parse::DumpIdOverloads

namespace Carbon::Parse {
template <>
auto DumpIdMethods<Tree>::Newline() const -> void {
  llvm::errs() << "\n";
}
}  // namespace Carbon::Parse

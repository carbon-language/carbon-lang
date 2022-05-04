// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_SEMANTICS_NODES_DECLARED_NAME_H_
#define TOOLCHAIN_SEMANTICS_NODES_DECLARED_NAME_H_

#include "common/ostream.h"
#include "toolchain/parser/parse_tree.h"

namespace Carbon::Semantics {

// Semantic information for a name.
class DeclaredName {
 public:
  DeclaredName(ParseTree::Node node, llvm::StringRef str)
      : node_(node), str_(str) {}

  void Print(llvm::raw_ostream& out) const { out << str_; }

  auto node() const -> ParseTree::Node { return node_; }
  auto str() const -> llvm::StringRef { return str_; }

 private:
  ParseTree::Node node_;
  llvm::StringRef str_;
};

}  // namespace Carbon::Semantics

#endif  // TOOLCHAIN_SEMANTICS_NODES_DECLARED_NAME_H_

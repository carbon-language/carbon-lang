// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_SEMANTICS_NODES_LITERAL_H_
#define TOOLCHAIN_SEMANTICS_NODES_LITERAL_H_

#include "common/ostream.h"
#include "toolchain/parser/parse_tree.h"

namespace Carbon::Semantics {

// Semantic information for a literal.
class Literal {
 public:
  explicit Literal(ParseTree::Node node) : node_(node) {}

  void Print(llvm::raw_ostream& out) const { out << "literal"; }

  auto node() const -> ParseTree::Node { return node_; }

 private:
  ParseTree::Node node_;
};

}  // namespace Carbon::Semantics

#endif  // TOOLCHAIN_SEMANTICS_NODES_LITERAL_H_

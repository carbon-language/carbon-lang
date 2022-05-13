// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_FUNCTION_H_
#define CARBON_TOOLCHAIN_SEMANTICS_FUNCTION_H_

#include "toolchain/parser/parse_tree.h"

namespace Carbon::Semantics {

// Semantic information for a function.
class Function {
 public:
  Function(ParseTree::Node decl_node, ParseTree::Node name_node)
      : decl_node_(decl_node), name_node_(name_node) {}

  auto decl_node() const -> ParseTree::Node { return decl_node_; }
  auto name_node() const -> ParseTree::Node { return name_node_; }

 private:
  // The FunctionDeclaration node.
  ParseTree::Node decl_node_;

  // The function's DeclaredName node.
  ParseTree::Node name_node_;
};

}  // namespace Carbon::Semantics

#endif  // CARBON_TOOLCHAIN_SEMANTICS_FUNCTION_H_

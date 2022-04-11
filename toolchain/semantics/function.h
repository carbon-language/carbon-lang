// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_SEMANTICS_FUNCTION_H_
#define TOOLCHAIN_SEMANTICS_FUNCTION_H_

#include "common/ostream.h"
#include "toolchain/parser/parse_tree.h"

namespace Carbon::Semantics {

// Semantic information for a function.
class Function {
 public:
  Function(ParseTree::Node decl_node, llvm::StringRef name,
           ParseTree::Node name_node)
      : decl_node_(decl_node), name_(name), name_node_(name_node) {}

  void Print(llvm::raw_ostream& out) const { out << "fn " << name_ << "()"; }

  auto decl_node() const -> ParseTree::Node { return decl_node_; }
  auto name() const -> llvm::StringRef { return name_; }
  auto name_node() const -> ParseTree::Node { return name_node_; }
  auto body() const -> llvm::StringRef { return "todo"; }

  auto return_type_node() const -> ParseTree::Node { return return_type_node_; }

 private:
  // The FunctionDeclaration node.
  ParseTree::Node decl_node_;

  // The actual name.
  llvm::StringRef name_;

  // The function's DeclaredName node.
  ParseTree::Node name_node_;

  // The function's ReturnType node.
  ParseTree::Node return_type_node_;
};

}  // namespace Carbon::Semantics

#endif  // TOOLCHAIN_SEMANTICS_FUNCTION_H_

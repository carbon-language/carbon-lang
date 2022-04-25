// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_SEMANTICS_FUNCTION_H_
#define TOOLCHAIN_SEMANTICS_FUNCTION_H_

#include "common/ostream.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/declared_name.h"

namespace Carbon::Semantics {

// Semantic information for a function.
class Function {
 public:
  explicit Function(ParseTree::Node decl_node, DeclaredName name)
      : decl_node_(decl_node), name_(name) {}

  void Print(llvm::raw_ostream& out) const { out << "fn " << name_ << "()"; }

  auto decl_node() const -> ParseTree::Node { return decl_node_; }
  auto name() const -> const DeclaredName& { return name_; }
  auto body() const -> ParseTree::Node { return {}; }

  void set_name(DeclaredName name) { name_ = name; }

 private:
  // The FunctionDeclaration node.
  ParseTree::Node decl_node_;

  // The function's name.
  DeclaredName name_;
};

}  // namespace Carbon::Semantics

#endif  // TOOLCHAIN_SEMANTICS_FUNCTION_H_

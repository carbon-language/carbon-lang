// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_SEMANTICS_NODES_FUNCTION_H_
#define TOOLCHAIN_SEMANTICS_NODES_FUNCTION_H_

#include "common/ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/nodes/declared_name.h"
#include "toolchain/semantics/nodes/pattern_binding.h"

namespace Carbon::Semantics {

// Semantic information for a function.
class Function {
 public:
  explicit Function(ParseTree::Node node, DeclaredName name,
                    llvm::SmallVector<PatternBinding, 0> params)
      : node_(node), name_(name), params_(std::move(params)) {}

  void Print(llvm::raw_ostream& out) const { out << "fn " << name_ << "()"; }

  auto node() const -> ParseTree::Node { return node_; }
  auto name() const -> const DeclaredName& { return name_; }
  auto params() const -> llvm::ArrayRef<PatternBinding> { return params_; }

  // TODO:
  auto body() const -> ParseTree::Node { return {}; }

 private:
  // The FunctionDeclaration node.
  ParseTree::Node node_;

  // The function's name.
  DeclaredName name_;

  // Regular function parameters.
  llvm::SmallVector<PatternBinding, 0> params_;
};

}  // namespace Carbon::Semantics

#endif  // TOOLCHAIN_SEMANTICS_NODES_FUNCTION_H_

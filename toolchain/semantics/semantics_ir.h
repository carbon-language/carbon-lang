// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_

#include "llvm/ADT/SmallVector.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/nodes/function.h"
#include "toolchain/semantics/nodes/literal.h"
#include "toolchain/semantics/nodes/meta_node_block.h"
#include "toolchain/semantics/nodes/pattern_binding.h"
#include "toolchain/semantics/nodes/return.h"

namespace Carbon::Testing {
class SemanticsIRSingleton;
}  // namespace Carbon::Testing

namespace Carbon {

// Provides semantic analysis on a ParseTree.
class SemanticsIR {
 public:
  void Print(llvm::raw_ostream& out, Semantics::Declaration decl) const;

  // File-level declarations.
  auto root_block() const -> const Semantics::DeclarationBlock& {
    return root_block_;
  }

 private:
  friend class SemanticsIRFactory;
  friend class Testing::SemanticsIRSingleton;

  explicit SemanticsIR(const ParseTree& parse_tree)
      : parse_tree_(&parse_tree) {}

  // Saves the Expression and returns a reference Statement. Non-statement
  // expressions don't need to be tracked this way.
  auto StoreExpressionStatement(Semantics::Expression expr)
      -> Semantics::Statement;
  // Saves the Function and returns a reference Declaration.
  auto StoreFunction(Semantics::Function function) -> Semantics::Declaration;
  // Saves the Literal and returns a reference Expression.
  auto StoreLiteral(Semantics::Literal lit) -> Semantics::Expression;
  // Saves the Return and returns a reference Statement.
  auto StoreReturn(Semantics::Return ret) -> Semantics::Statement;

  // Helpers for debug printing.
  void Print(llvm::raw_ostream& out, ParseTree::Node node) const;
  void Print(llvm::raw_ostream& out, Semantics::DeclaredName name) const;
  void Print(llvm::raw_ostream& out, Semantics::Expression expr) const;
  void Print(llvm::raw_ostream& out, Semantics::Function function) const;
  void Print(llvm::raw_ostream& out, Semantics::Literal literal) const;
  void Print(llvm::raw_ostream& out, Semantics::PatternBinding binding) const;

  // Lists for Semantics::DeclarationKind.
  llvm::SmallVector<Semantics::Function, 0> functions_;

  // Lists for Semantics::ExpressionKind.
  llvm::SmallVector<Semantics::Literal, 0> literals_;

  // Lists for Semantics::StatementKind.
  llvm::SmallVector<Semantics::Expression, 0> expression_statements_;
  llvm::SmallVector<Semantics::Return, 0> returns_;

  // The file-level block.
  Semantics::DeclarationBlock root_block_;

  const ParseTree* parse_tree_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_

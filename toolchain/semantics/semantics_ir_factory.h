// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FACTORY_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FACTORY_H_

#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/semantics_ir.h"

namespace Carbon {

// The main semantic analysis entry.
class SemanticsIRFactory {
 public:
  // Builds the SemanticsIR without doing any substantial semantic analysis.
  static auto Build(const ParseTree& parse_tree) -> SemanticsIR;

 private:
  explicit SemanticsIRFactory(const ParseTree& parse_tree)
      : semantics_(parse_tree) {}

  void Build();

  // Requires that a node have no children, to emphasize why the subtree isn't
  // otherwise checked.
  void RequireNodeEmpty(ParseTree::Node node);

  // Each of these takes a parse tree node and does a transformation based on
  // its type. These functions are per ParseNodeKind.
  auto TransformCodeBlock(ParseTree::Node node) -> Semantics::StatementBlock;
  auto TransformDeclaredName(ParseTree::Node node) -> Semantics::DeclaredName;
  auto TransformExpression(ParseTree::Node node) -> Semantics::Expression;
  auto TransformExpressionStatement(ParseTree::Node node)
      -> Semantics::Statement;
  auto TransformFunctionDeclaration(ParseTree::Node node)
      -> std::tuple<llvm::StringRef, Semantics::Declaration>;
  auto TransformParameterList(ParseTree::Node node)
      -> llvm::SmallVector<Semantics::PatternBinding, 0>;
  auto TransformPatternBinding(ParseTree::Node node)
      -> Semantics::PatternBinding;
  auto TransformReturnType(ParseTree::Node node) -> Semantics::Expression;
  auto TransformReturnStatement(ParseTree::Node node) -> Semantics::Statement;

  // Convenience accessor.
  auto parse_tree() -> const ParseTree& { return *semantics_.parse_tree_; }

  SemanticsIR semantics_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FACTORY_H_

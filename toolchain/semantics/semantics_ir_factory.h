// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FACTORY_H_
#define TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FACTORY_H_

#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/nodes/declared_name.h"
#include "toolchain/semantics/nodes/expression.h"
#include "toolchain/semantics/nodes/function.h"
#include "toolchain/semantics/nodes/literal.h"
#include "toolchain/semantics/nodes/pattern_binding.h"
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

  auto TransformCodeBlock(ParseTree::Node node) -> void;
  auto TransformDeclaredName(ParseTree::Node node) -> Semantics::DeclaredName;
  void TransformFunctionDeclaration(ParseTree::Node node,
                                    SemanticsIR::Block& block);
  auto TransformParameterList(ParseTree::Node node)
      -> llvm::SmallVector<Semantics::PatternBinding, 0>;
  auto TransformExpression(ParseTree::Node node) -> Semantics::Expression;
  auto TransformPatternBinding(ParseTree::Node node)
      -> Semantics::PatternBinding;
  auto TransformReturnType(ParseTree::Node node) -> Semantics::Expression;

  // Convenience accessor.
  auto parse_tree() -> const ParseTree& { return *semantics_.parse_tree_; }

  SemanticsIR semantics_;
};

}  // namespace Carbon

#endif  // TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FACTORY_H_

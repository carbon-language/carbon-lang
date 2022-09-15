// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FACTORY_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FACTORY_H_

#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/parse_subtree_consumer.h"
#include "toolchain/semantics/semantics_ir.h"

namespace Carbon {

// The main semantic analysis entry.
class SemanticsIRFactory {
 public:
  // Builds the SemanticsIR without doing any substantial semantic analysis.
  static auto Build(const TokenizedBuffer& tokens, const ParseTree& parse_tree)
      -> SemanticsIR;

 private:
  explicit SemanticsIRFactory(const TokenizedBuffer& tokens,
                              const ParseTree& parse_tree)
      : tokens_(&tokens), semantics_(parse_tree) {}

  void Build();

  // Requires that a node have no children, to emphasize why the subtree isn't
  // otherwise checked.
  void RequireNodeEmpty(ParseTree::Node node);

  // Transforms a block subtree, such as a file or CodeBlock, into its semantic
  // nodes.
  auto TransformBlockSubtree(ParseSubtreeConsumer& subtree,
                             ParseNodeKind end_kind)
      -> llvm::SmallVector<Semantics::NodeRef, 0>;

  // Each of these takes a parse tree node and does a transformation based on
  // its type. These functions are per ParseNodeKind.
  auto TransformCodeBlock(ParseTree::Node node)
      -> llvm::SmallVector<Semantics::NodeRef, 0>;
  void TransformDeclaredName(llvm::SmallVector<Semantics::NodeRef, 0>& nodes,
                             ParseTree::Node node, Semantics::NodeId target_id);
  void TransformExpression(llvm::SmallVector<Semantics::NodeRef, 0>& nodes,
                           ParseTree::Node node, Semantics::NodeId target_id);
  // auto TransformExpressionStatement(ParseTree::Node node)
  //   -> Semantics::Statement;
  void TransformFunctionDeclaration(
      llvm::SmallVector<Semantics::NodeRef, 0>& nodes, ParseTree::Node node);
  void TransformInfixOperator(llvm::SmallVector<Semantics::NodeRef, 0>& nodes,
                              ParseTree::Node node,
                              Semantics::NodeId target_id);
  // auto TransformParameterList(ParseTree::Node node)
  //   -> llvm::SmallVector<Semantics::PatternBinding, 0>
  // auto TransformPatternBinding(ParseTree::Node node)
  //   -> Semantics::PatternBinding;
  // auto TransformReturnType(ParseTree::Node node) -> Semantics::Statement;
  void TransformReturnStatement(llvm::SmallVector<Semantics::NodeRef, 0>& nodes,
                                ParseTree::Node node);

  // Returns a unique ID for the SemanticsIR.
  auto next_id() -> Semantics::NodeId {
    return Semantics::NodeId(id_counter_++);
  }

  // Convenience accessor.
  auto parse_tree() -> const ParseTree& { return *semantics_.parse_tree_; }

  // Tokens for getting data on literals.
  const TokenizedBuffer* tokens_;

  // The SemanticsIR being constructed.
  SemanticsIR semantics_;

  // A counter for unique IDs.
  int32_t id_counter_ = 0;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FACTORY_H_

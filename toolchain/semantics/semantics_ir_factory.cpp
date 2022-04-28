// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir_factory.h"

#include <stack>

#include "common/check.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/semantics/parse_subtree_consumer.h"

namespace Carbon {

auto SemanticsIRFactory::Build(const ParseTree& parse_tree) -> SemanticsIR {
  SemanticsIRFactory builder(parse_tree);
  builder.Build();
  return builder.semantics_;
}

void SemanticsIRFactory::Build() {
  auto subtree = ParseSubtreeConsumer::ForTree(parse_tree());
  // FileEnd is a placeholder node which can be discarded.
  RequireNodeEmpty(subtree.RequireConsume(ParseNodeKind::FileEnd()));
  while (llvm::Optional<ParseTree::Node> node = subtree.TryConsume()) {
    const auto node_kind = parse_tree().node_kind(*node);
    switch (node_kind) {
      case ParseNodeKind::FunctionDeclaration():
        TransformFunctionDeclaration(*node, semantics_.root_block_);
        break;
      default:
        FATAL() << "At index " << node->index() << ", unexpected "
                << node_kind.name();
    }
  }
}

void SemanticsIRFactory::RequireNodeEmpty(ParseTree::Node node) {
  auto subtree_size = parse_tree().node_subtree_size(node);
  CHECK(subtree_size == 1) << "At index " << node.index() << ", expected "
                           << parse_tree().node_kind(node)
                           << "would have subtree_size of 1, but was "
                           << subtree_size;
}

auto SemanticsIRFactory::TransformDeclaredName(ParseTree::Node node)
    -> Semantics::DeclaredName {
  CHECK(parse_tree().node_kind(node) == ParseNodeKind::DeclaredName());
  RequireNodeEmpty(node);

  return Semantics::DeclaredName(parse_tree().GetNodeText(node), node);
}

void SemanticsIRFactory::TransformFunctionDeclaration(
    ParseTree::Node node, SemanticsIR::Block& block) {
  CHECK(parse_tree().node_kind(node) == ParseNodeKind::FunctionDeclaration());

  auto subtree = ParseSubtreeConsumer::ForParent(parse_tree(), node);
  // TODO: Parse code.
  (void)subtree.TryConsume(ParseNodeKind::CodeBlock());
  // TODO: Parse return type.
  (void)subtree.TryConsume(ParseNodeKind::ReturnType());
  TransformParameterList(
      subtree.RequireConsume(ParseNodeKind::ParameterList()));
  auto name = TransformDeclaredName(
      subtree.RequireConsume(ParseNodeKind::DeclaredName()));
  semantics_.AddFunction(block, Semantics::Function(node, name));
}

void SemanticsIRFactory::TransformParameterList(ParseTree::Node node) {
  CHECK(parse_tree().node_kind(node) == ParseNodeKind::ParameterList());

  auto subtree = ParseSubtreeConsumer::ForParent(parse_tree(), node);

  RequireNodeEmpty(subtree.RequireConsume(ParseNodeKind::ParameterListEnd()));
  if (auto first_param_node =
          subtree.TryConsume(ParseNodeKind::PatternBinding())) {
    TransformPatternBinding(*first_param_node);

    while (auto comma_node =
               subtree.TryConsume(ParseNodeKind::ParameterListComma())) {
      RequireNodeEmpty(*comma_node);
      TransformPatternBinding(
          subtree.RequireConsume(ParseNodeKind::PatternBinding()));
    }
  }
}

void SemanticsIRFactory::TransformPattern(ParseTree::Node node) {
  RequireNodeEmpty(node);

  // TODO: Turn this into an expression.
}

void SemanticsIRFactory::TransformPatternBinding(ParseTree::Node node) {
  CHECK(parse_tree().node_kind(node) == ParseNodeKind::PatternBinding());

  auto subtree = ParseSubtreeConsumer::ForParent(parse_tree(), node);
  // TODO: Need to rewrite to handle expressions here.
  TransformPattern(subtree.RequireConsume(ParseNodeKind::Literal()));
  TransformDeclaredName(subtree.RequireConsume(ParseNodeKind::DeclaredName()));
}

}  // namespace Carbon

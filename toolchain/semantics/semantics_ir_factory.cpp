// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir_factory.h"

#include <stack>

#include "common/check.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"

namespace Carbon {

class SubtreeConsumer {
 public:
  using ParseTreeIterator = std::reverse_iterator<ParseTree::PostorderIterator>;

  static auto ForParent(const ParseTree& parse_tree,
                        ParseTree::Node parent_node) -> SubtreeConsumer {
    auto range = llvm::reverse(parse_tree.postorder(parent_node));
    // The cursor should be one after the parent.
    return SubtreeConsumer(parse_tree, ++range.begin(), range.end());
  }

  static auto ForTree(const ParseTree& parse_tree) -> SubtreeConsumer {
    auto range = llvm::reverse(parse_tree.postorder());
    return SubtreeConsumer(parse_tree, range.begin(), range.end());
  }

  // Prevent copies because we require completion of parsing in the destructor.
  SubtreeConsumer(const SubtreeConsumer&) = delete;
  auto operator=(const SubtreeConsumer&) -> SubtreeConsumer& = delete;

  ~SubtreeConsumer() {
    CHECK(is_done()) << "At index " << (*cursor_).index() << ", unhandled "
                     << parse_tree_->node_kind(*cursor_);
  }

  [[nodiscard]] auto RequireConsume(ParseNodeKind node_kind)
      -> ParseTree::Node {
    CHECK(!is_done());
    llvm::Optional<ParseTree::Node> node = TryConsume(node_kind);
    CHECK(node != llvm::None)
        << "At index " << (*cursor_).index() << ", expected " << node_kind
        << ", found " << parse_tree_->node_kind(*cursor_);
    return *node;
  }

  [[nodiscard]] auto TryConsume() -> llvm::Optional<ParseTree::Node> {
    if (is_done()) {
      return llvm::None;
    }
    return GetNodeAndAdvance();
  }

  [[nodiscard]] auto TryConsume(ParseNodeKind node_kind)
      -> llvm::Optional<ParseTree::Node> {
    if (is_done() || parse_tree_->node_kind(*cursor_) != node_kind) {
      return llvm::None;
    }
    return GetNodeAndAdvance();
  }

  // Returns true if there are no more nodes to consume.
  auto is_done() -> bool { return cursor_ == subtree_end_; }

 private:
  // Constructs for a subtree.
  SubtreeConsumer(const ParseTree& parse_tree, ParseTreeIterator cursor,
                  ParseTreeIterator subtree_end)
      : parse_tree_(&parse_tree), cursor_(cursor), subtree_end_(subtree_end) {}

  auto GetNodeAndAdvance() -> ParseTree::Node {
    auto node = *cursor_;
    cursor_ += parse_tree_->node_subtree_size(node);
    return node;
  }

  const ParseTree* parse_tree_;
  ParseTreeIterator cursor_;
  ParseTreeIterator subtree_end_;
};

auto SemanticsIRFactory::Build(const ParseTree& parse_tree) -> SemanticsIR {
  SemanticsIRFactory builder(parse_tree);
  builder.Build();
  return builder.semantics_;
}

void SemanticsIRFactory::Build() {
  SubtreeConsumer subtree = SubtreeConsumer::ForTree(parse_tree());
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

  SubtreeConsumer subtree = SubtreeConsumer::ForParent(parse_tree(), node);
  // TODO
  (void)subtree.TryConsume(ParseNodeKind::CodeBlock());
  // TODO
  (void)subtree.TryConsume(ParseNodeKind::ReturnType());
  TransformParameterList(
      subtree.RequireConsume(ParseNodeKind::ParameterList()));
  auto name = TransformDeclaredName(
      subtree.RequireConsume(ParseNodeKind::DeclaredName()));
  semantics_.AddFunction(block, Semantics::Function(node, name));
}

void SemanticsIRFactory::TransformParameterList(ParseTree::Node node) {
  CHECK(parse_tree().node_kind(node) == ParseNodeKind::ParameterList());

  SubtreeConsumer subtree = SubtreeConsumer::ForParent(parse_tree(), node);

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

  // TODO
}

void SemanticsIRFactory::TransformPatternBinding(ParseTree::Node node) {
  CHECK(parse_tree().node_kind(node) == ParseNodeKind::PatternBinding());

  SubtreeConsumer subtree = SubtreeConsumer::ForParent(parse_tree(), node);
  // TODO: Need to rewrite to handle expressions here.
  TransformPattern(subtree.RequireConsume(ParseNodeKind::Literal()));
  TransformDeclaredName(subtree.RequireConsume(ParseNodeKind::DeclaredName()));
}

}  // namespace Carbon

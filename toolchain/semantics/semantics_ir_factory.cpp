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

auto SemanticsIRFactory::Build(const ParseTree& parse_tree) -> SemanticsIR {
  SemanticsIRFactory builder(parse_tree);
  builder.Build();
  return builder.semantics_;
}

void SemanticsIRFactory::Build() {
  std::stack<int> depth_ends;
  for (; cursor_ != range_.end();) {
    ParseTree::Node n = *cursor_;
    const auto node_kind = parse_tree().node_kind(n);
    switch (node_kind) {
      case ParseNodeKind::FileEnd():
        // Discard.
        CHECK(parse_tree().node_subtree_size(n) == 1);
        ++cursor_;
        break;
      case ParseNodeKind::FunctionDeclaration():
        ParseFunctionDeclaration(semantics_.root_block_);
        break;
      default:
        FATAL() << "Unhandled node kind at index " << n.index() << ": "
                << node_kind.name();
    }
  }
}

void SemanticsIRFactory::ParseFunctionDeclaration(SemanticsIR::Block& block) {
  ParseTree::Node decl_node = *cursor_;
  CHECK(parse_tree().node_kind(decl_node) ==
        ParseNodeKind::FunctionDeclaration());
  auto subtree_end = GetSubtreeEnd();
  ++cursor_;

  while (cursor_ != subtree_end) {
    ParseTree::Node n = *cursor_;
    const auto node_kind = parse_tree().node_kind(n);
    llvm::Optional<Semantics::Function> fn;
    /*
    llvm::Optional<ParseTree::Node> code_block;
    llvm::Optional<ParseTree::Node> return_type;
    llvm::Optional<ParseTree::Node> parameter_list;
    for (ParseTree::Node node : semantics_.parse_tree_->children(decl_node)) {
      llvm::errs() << semantics_.parse_tree_->node_kind(node).name() << "\n";
      */
    switch (node_kind) {
      case ParseNodeKind::DeclaredName():
        CHECK(parse_tree().node_subtree_size(n) == 1);
        ++cursor_;
        fn = semantics_.AddFunction(block, decl_node, n);
        break;
      case ParseNodeKind::ParameterList():
        cursor_ = GetSubtreeEnd();
        break;
      case ParseNodeKind::CodeBlock():
        cursor_ = GetSubtreeEnd();
        break;
      case ParseNodeKind::ReturnType():
        cursor_ = GetSubtreeEnd();
        break;
      default:
        FATAL() << "Unhandled node kind at index " << n.index() << ": "
                << node_kind.name();
    }
  }
}

}  // namespace Carbon

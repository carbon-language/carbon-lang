// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir_factory.h"

#include "common/check.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"

namespace Carbon {

auto SemanticsIRFactory::Build(const ParseTree& parse_tree) -> SemanticsIR {
  SemanticsIRFactory builder(parse_tree);
  builder.ProcessRoots();
  return builder.semantics_;
}

void SemanticsIRFactory::ProcessRoots() {
  for (ParseTree::Node node : semantics_.parse_tree_->roots()) {
    switch (semantics_.parse_tree_->node_kind(node)) {
      case ParseNodeKind::FunctionDeclaration():
        ProcessFunctionNode(semantics_.root_block_, node);
        break;
      case ParseNodeKind::FileEnd():
        // No action needed.
        break;
      default:
        CARBON_FATAL() << "Unhandled node kind: "
                       << semantics_.parse_tree_->node_kind(node).name();
    }
  }
}

void SemanticsIRFactory::ProcessFunctionNode(SemanticsIR::Block& block,
                                             ParseTree::Node decl_node) {
  llvm::Optional<Semantics::Function> fn;
  for (ParseTree::Node node : semantics_.parse_tree_->children(decl_node)) {
    switch (semantics_.parse_tree_->node_kind(node)) {
      case ParseNodeKind::DeclaredName():
        fn = semantics_.AddFunction(block, decl_node, node);
        break;
      case ParseNodeKind::ParameterList():
        // TODO: Maybe something like Semantics::AddVariable passed to
        // Function::AddParameter.
        break;
      case ParseNodeKind::CodeBlock():
        // TODO: Should accumulate the definition into the code block.
        break;
      default:
        CARBON_FATAL() << "Unhandled node kind: "
                       << semantics_.parse_tree_->node_kind(node).name();
    }
  }
}

}  // namespace Carbon

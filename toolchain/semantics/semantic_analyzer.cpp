// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantic_analyzer.h"

#include <optional>

#include "common/check.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"

namespace Carbon {

auto SemanticAnalyzer::Analyze(const ParseTree& parse_tree,
                               DiagnosticConsumer& consumer) -> Semantics {
  SemanticAnalyzer analyzer(parse_tree, consumer);
  return analyzer.semantics_;
}

void SemanticAnalyzer::ProcessRoots() {
  for (ParseTree::Node node : semantics_.parse_tree_->Roots()) {
    switch (semantics_.parse_tree_->GetNodeKind(node)) {
      case ParseNodeKind::FunctionDeclaration():
        ProcessFunctionNode(semantics_.root_name_scope_, node);
        break;
      case ParseNodeKind::FileEnd():
        // No action needed.
        break;
      default:
        FATAL() << "Unhandled node kind: "
                << semantics_.parse_tree_->GetNodeKind(node).GetName();
    }
  }
}

void SemanticAnalyzer::ProcessFunctionNode(
    llvm::StringMap<Semantics::NamedEntity>& name_scope,
    ParseTree::Node fn_node) {
  llvm::Expected<Semantics::Function> fn;
  for (ParseTree::Node node : semantics_.parse_tree_->Children(fn_node)) {
    switch (semantics_.parse_tree_->GetNodeKind(node)) {
      case ParseNodeKind::DeclaredName():
        fn = semantics_.AddFunction(name_scope,
                                    semantics_.parse_tree_->GetNodeText(node));
        break;
      case ParseNodeKind::CodeBlock():
      case ParseNodeKind::ParameterList():
        // TODO: Should add information to the function object.
        // Something like Function::AddParameter, etc.
        break;
      default:
        FATAL() << "Unhandled node kind: "
                << semantics_.parse_tree_->GetNodeKind(node).GetName();
    }
  }
  (void)fn;
}

}  // namespace Carbon

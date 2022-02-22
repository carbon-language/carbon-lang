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
  analyzer.ProcessRoots();
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
    ParseTree::Node decl_node) {
  llvm::Optional<Semantics::Function> fn = llvm::None;
  for (ParseTree::Node node : semantics_.parse_tree_->Children(decl_node)) {
    switch (semantics_.parse_tree_->GetNodeKind(node)) {
      case ParseNodeKind::DeclaredName():
        fn = semantics_.AddFunction(emitter_, name_scope, decl_node, node);
        break;
      case ParseNodeKind::ParameterList():
        // TODO: Maybe something like Semantics::AddVariable passed to
        // Function::AddParameter.
        break;
      case ParseNodeKind::CodeBlock():
        // TODO: Should accumulate the definition into the code block.
        break;
      default:
        FATAL() << "Unhandled node kind: "
                << semantics_.parse_tree_->GetNodeKind(node).GetName();
    }
  }
  (void)fn;
}

}  // namespace Carbon

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics.h"

#include <optional>

#include "common/check.h"
#include "toolchain/parser/parse_node_kind.h"

namespace Carbon {

class Semantics::Analyzer {
 public:
  Analyzer(const ParseTree& parse_tree, DiagnosticConsumer& consumer)
      : parse_tree_(parse_tree), consumer_(consumer) {}

  auto Analyze() -> Semantics {
    semantics_ = Semantics();
    for (ParseTree::Node node : parse_tree_.Roots()) {
      switch (parse_tree_.GetNodeKind(node)) {
        case ParseNodeKind::FunctionDeclaration():
          AnalyzeFunction(semantics_->root_name_scope_, node);
          break;
        case ParseNodeKind::FileEnd():
          // No action needed.
          break;
        default:
          FATAL() << "Unhandled node kind: "
                  << parse_tree_.GetNodeKind(node).GetName();
      }
    }
    consumer_.HandleDiagnostic({});
    return *semantics_;
  }

 private:
  void AnalyzeFunction(llvm::StringMap<Entity>& name_scope,
                       ParseTree::Node fn_node) {
    for (ParseTree::Node node : parse_tree_.Children(fn_node)) {
      Function fn;
      switch (parse_tree_.GetNodeKind(node)) {
        case ParseNodeKind::DeclaredName():
          fn.name_node = node;
          break;
        case ParseNodeKind::CodeBlock():
        case ParseNodeKind::ParameterList():
          // TODO: Analyze.
          break;
        default:
          FATAL() << "Unhandled node kind: "
                  << parse_tree_.GetNodeKind(node).GetName();
      }
      name_scope.insert(
          {parse_tree_.GetNodeText(fn.name_node),
           {Entity::Category::Function,
            static_cast<int32_t>(semantics_->functions_.size())}});
      semantics_->functions_.push_back(fn);
    }
  }

  const ParseTree& parse_tree_;
  DiagnosticConsumer& consumer_;
  std::optional<Semantics> semantics_;
};

auto Semantics::Analyze(const ParseTree& parse_tree,
                        DiagnosticConsumer& consumer) -> Semantics {
  Analyzer analyzer(parse_tree, consumer);
  return analyzer.Analyze();
}

}  // namespace Carbon

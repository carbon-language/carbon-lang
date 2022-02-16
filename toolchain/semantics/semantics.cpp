// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics.h"

#include <optional>

#include "common/check.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"

namespace Carbon {

struct NameConflict : DiagnosticBase<NameConflict> {
  static constexpr llvm::StringLiteral ShortName = "semantics-name-conflict";

  auto Format() -> std::string {
    return llvm::formatv(
        "Name conflict for `{0}`; previously declared at {1}:{2}:{3}.", name,
        conflict.file_name, conflict.line_number, conflict.column_number);
  }

  llvm::StringRef name;
  Diagnostic::Location conflict;
};

// Provides diagnostic locations for a ParseTree Node.
class ParseTreeNodeLocationTranslator
    : public DiagnosticLocationTranslator<ParseTree::Node> {
 public:
  explicit ParseTreeNodeLocationTranslator(const ParseTree& parse_tree)
      : parse_tree_(&parse_tree) {}

  // Translate a particular node to a location.
  auto GetLocation(ParseTree::Node loc) -> Diagnostic::Location override {
    auto token = parse_tree_->GetNodeToken(loc);
    TokenizedBuffer::TokenLocationTranslator translator(
        parse_tree_->tokens(), /*last_line_lexed_to_column=*/nullptr);
    return translator.GetLocation(token);
  }

 private:
  const ParseTree* parse_tree_;
};

// Runs the actual analysis for Semantics. This is separate from Semantics in
// order to track analysis-specific context, such as the emitter, which should
// not be part of the resulting Semantics object.
class Semantics::Analyzer {
 public:
  Analyzer(const ParseTree& parse_tree, DiagnosticConsumer& consumer)
      : parse_tree_(&parse_tree),
        translator_(parse_tree),
        emitter_(translator_, consumer) {}

  // Produces a Semantics object from the ParseTree.
  auto Analyze() -> Semantics;

 private:
  // Analyzes a function's node, adding it to the provided name scope.
  void AnalyzeFunction(llvm::StringMap<NamedEntity>& name_scope,
                       ParseTree::Node fn_node);

  // Returns the location of an entity. This assists diagnostic output where
  // supplemental locations are provided in formatting.
  auto GetEntityLocation(NamedEntity entity) -> Diagnostic::Location;

  const ParseTree* parse_tree_;
  ParseTreeNodeLocationTranslator translator_;
  DiagnosticEmitter<ParseTree::Node> emitter_;
  std::optional<Semantics> semantics_;
};

auto Semantics::Analyze(const ParseTree& parse_tree,
                        DiagnosticConsumer& consumer) -> Semantics {
  Analyzer analyzer(parse_tree, consumer);
  return analyzer.Analyze();
}

auto Semantics::Analyzer::Analyze() -> Semantics {
  semantics_ = Semantics();
  for (ParseTree::Node node : parse_tree_->Roots()) {
    switch (parse_tree_->GetNodeKind(node)) {
      case ParseNodeKind::FunctionDeclaration():
        AnalyzeFunction(semantics_->root_name_scope_, node);
        break;
      case ParseNodeKind::FileEnd():
        // No action needed.
        break;
      default:
        FATAL() << "Unhandled node kind: "
                << parse_tree_->GetNodeKind(node).GetName();
    }
  }
  return *semantics_;
}

void Semantics::Analyzer::AnalyzeFunction(
    llvm::StringMap<NamedEntity>& name_scope, ParseTree::Node fn_node) {
  Function fn;
  for (ParseTree::Node node : parse_tree_->Children(fn_node)) {
    switch (parse_tree_->GetNodeKind(node)) {
      case ParseNodeKind::DeclaredName():
        fn.name_node = node;
        break;
      case ParseNodeKind::CodeBlock():
      case ParseNodeKind::ParameterList():
        // TODO: Analyze.
        break;
      default:
        FATAL() << "Unhandled node kind: "
                << parse_tree_->GetNodeKind(node).GetName();
    }
  }
  llvm::StringRef fn_name = parse_tree_->GetNodeText(fn.name_node);
  auto [it, success] = name_scope.insert(
      {fn_name,
       {NamedEntity::Kind::Function,
        static_cast<int32_t>(semantics_->functions_.size())}});
  if (!success) {
    emitter_.EmitError<NameConflict>(
        fn.name_node,
        {.name = fn_name, .conflict = GetEntityLocation(name_scope[fn_name])});
    return;
  }
  semantics_->functions_.push_back(fn);
}

auto Semantics::Analyzer::GetEntityLocation(NamedEntity entity)
    -> Diagnostic::Location {
  switch (entity.kind_) {
    case NamedEntity::Kind::Function: {
      Function fn = semantics_->functions_[entity.index_];
      return translator_.GetLocation(fn.name_node);
    }
  }
}

}  // namespace Carbon

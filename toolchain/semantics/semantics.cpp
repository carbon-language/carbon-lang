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

class ParseTreeNodeTranslator
    : public DiagnosticLocationTranslator<ParseTree::Node> {
 public:
  explicit ParseTreeNodeTranslator(const ParseTree& parse_tree)
      : parse_tree_(&parse_tree) {}

  auto GetLocation(ParseTree::Node loc) -> Diagnostic::Location override {
    auto token = parse_tree_->GetNodeToken(loc);
    TokenizedBuffer::TokenLocationTranslator translator(
        parse_tree_->tokens(), /*last_line_lexed_to_column=*/nullptr);
    return translator.GetLocation(token);
  }

 private:
  const ParseTree* parse_tree_;
};

class Semantics::Analyzer {
 public:
  Analyzer(const ParseTree& parse_tree, DiagnosticConsumer& consumer)
      : parse_tree_(&parse_tree),
        translator_(parse_tree),
        emitter_(translator_, consumer) {}

  auto Analyze() -> Semantics;

 private:
  void AnalyzeFunction(llvm::StringMap<Entity>& name_scope,
                       ParseTree::Node fn_node);

  auto GetEntityLocation(Entity entity) -> Diagnostic::Location;

  const ParseTree* parse_tree_;
  ParseTreeNodeTranslator translator_;
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

void Semantics::Analyzer::AnalyzeFunction(llvm::StringMap<Entity>& name_scope,
                                          ParseTree::Node fn_node) {
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
       {Entity::Kind::Function,
        static_cast<int32_t>(semantics_->functions_.size())}});
  if (!success) {
    emitter_.EmitError<NameConflict>(
        fn.name_node,
        {.name = fn_name, .conflict = GetEntityLocation(name_scope[fn_name])});
    return;
  }
  semantics_->functions_.push_back(fn);
}

auto Semantics::Analyzer::GetEntityLocation(Entity entity)
    -> Diagnostic::Location {
  switch (entity.kind_) {
    case Entity::Kind::Function: {
      Function fn = semantics_->functions_[entity.index_];
      return translator_.GetLocation(fn.name_node);
    }
  }
}

}  // namespace Carbon

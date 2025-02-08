// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <optional>

#include "common/check.h"
#include "toolchain/language_server/handle.h"
#include "toolchain/lex/token_index.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/parse/tree_and_subtrees.h"

namespace Carbon::LanguageServer {

// Returns the text of first child of kind IdentifierNameBeforeParams or
// IdentifierNameNotBeforeParams.
static auto GetIdentifierName(const Parse::TreeAndSubtrees& tree_and_subtrees,
                              Parse::NodeId node)
    -> std::optional<llvm::StringRef> {
  const auto& tokens = tree_and_subtrees.tree().tokens();
  for (auto child : tree_and_subtrees.children(node)) {
    switch (tree_and_subtrees.tree().node_kind(child)) {
      case Parse::NodeKind::IdentifierNameBeforeParams:
      case Parse::NodeKind::IdentifierNameNotBeforeParams: {
        auto token = tree_and_subtrees.tree().node_token(child);
        if (tokens.GetKind(token) == Lex::TokenKind::Identifier) {
          return tokens.GetTokenText(token);
        }
        break;
      }
      default:
        break;
    }
  }
  return std::nullopt;
}

class SymbolStore {
 public:
  // Adds a symbol with no children.
  void AddSymbol(clang::clangd::DocumentSymbol symbol) {
    if (open_symbols_.empty()) {
      top_level_symbols_.push_back(std::move(symbol));
    } else {
      open_symbols_.back().children.push_back(symbol);
    }
  }

  // Starts a symbol potentially with children.
  void StartSymbol(clang::clangd::DocumentSymbol symbol) {
    open_symbols_.push_back(std::move(symbol));
  }

  auto HasOpenSymbol() const -> bool { return !open_symbols_.empty(); }

  // Completes a symbol, appending to parent list.
  void EndSymbol() {
    CARBON_CHECK(HasOpenSymbol());
    AddSymbol(open_symbols_.pop_back_val());
  }

  auto Collect() -> std::vector<clang::clangd::DocumentSymbol> {
    // Shouldn't happen in a valid tree but may as well handle gracefully.
    while (!open_symbols_.empty()) {
      EndSymbol();
    }

    return std::move(top_level_symbols_);
  }

 private:
  std::vector<clang::clangd::DocumentSymbol> top_level_symbols_;
  llvm::SmallVector<clang::clangd::DocumentSymbol> open_symbols_;
};

auto HandleDocumentSymbol(
    Context& context, const clang::clangd::DocumentSymbolParams& params,
    llvm::function_ref<
        void(llvm::Expected<std::vector<clang::clangd::DocumentSymbol>>)>
        on_done) -> void {
  auto* file = context.LookupFile(params.textDocument.uri.file());
  if (!file) {
    return;
  }

  const auto& tree_and_subtrees = file->tree_and_subtrees();
  const auto& tree = tree_and_subtrees.tree();
  const auto& tokens = tree.tokens();

  SymbolStore symbols;
  for (const auto& node_id : tree.postorder()) {
    auto node_kind = tree.node_kind(node_id);
    clang::clangd::SymbolKind symbol_kind;
    bool is_leaf = false;
    switch (node_kind) {
      case Parse::NodeKind::FunctionDecl:
        is_leaf = true;
        symbol_kind = clang::clangd::SymbolKind::Function;
        break;
      case Parse::NodeKind::FunctionDefinitionStart:
        symbol_kind = clang::clangd::SymbolKind::Function;
        break;
      case Parse::NodeKind::Namespace:
        symbol_kind = clang::clangd::SymbolKind::Namespace;
        break;
      case Parse::NodeKind::InterfaceDefinitionStart:
      case Parse::NodeKind::NamedConstraintDefinitionStart:
        symbol_kind = clang::clangd::SymbolKind::Interface;
        break;
      case Parse::NodeKind::ClassDecl:
        is_leaf = true;
        symbol_kind = clang::clangd::SymbolKind::Class;
        break;
      case Parse::NodeKind::ClassDefinitionStart:
        symbol_kind = clang::clangd::SymbolKind::Class;
        break;

      case Parse::NodeKind::FunctionDefinition:
      case Parse::NodeKind::NamedConstraintDefinition:
      case Parse::NodeKind::InterfaceDefinition:
      case Parse::NodeKind::ClassDefinition: {
        if (symbols.HasOpenSymbol()) {
          // Symbols definition has completed, pop it from stack and add to
          // parent/root.
          symbols.EndSymbol();
        }
        continue;
      }

      default:
        continue;
    }

    if (auto name = GetIdentifierName(tree_and_subtrees, node_id)) {
      auto token = tree.node_token(node_id);
      clang::clangd::Position pos{tokens.GetLineNumber(token) - 1,
                                  tokens.GetColumnNumber(token) - 1};

      clang::clangd::DocumentSymbol symbol{
          .name = std::string(*name),
          .kind = symbol_kind,
          .range = {.start = pos, .end = pos},
          .selectionRange = {.start = pos, .end = pos},
      };

      if (is_leaf) {
        symbols.AddSymbol(std::move(symbol));
      } else {
        symbols.StartSymbol(std::move(symbol));
      }
    }
  }

  on_done(symbols.Collect());
}

}  // namespace Carbon::LanguageServer

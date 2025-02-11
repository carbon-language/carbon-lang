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

// Returns the token of first child of kind IdentifierNameBeforeParams or
// IdentifierNameNotBeforeParams.
static auto GetSymbolIdentifier(const Parse::TreeAndSubtrees& tree_and_subtrees,
                                Parse::NodeId node)
    -> std::optional<Lex::TokenIndex> {
  const auto& tokens = tree_and_subtrees.tree().tokens();
  for (auto child : tree_and_subtrees.children(node)) {
    switch (tree_and_subtrees.tree().node_kind(child)) {
      case Parse::NodeKind::IdentifierNameBeforeParams:
      case Parse::NodeKind::IdentifierNameNotBeforeParams: {
        auto token = tree_and_subtrees.tree().node_token(child);
        if (tokens.GetKind(token) == Lex::TokenKind::Identifier) {
          return token;
        }
        break;
      }
      default:
        break;
    }
  }
  return std::nullopt;
}

// Helper class to collect all symbols during traversal of AST.
// Symbols are "open" when their signature has been declared but their body is
// still ongoing. New symbols are added to last open symbol, or top level list
// if no symbols are open.
class SymbolStore {
 public:
  // Adds a symbol with no children.
  auto AddSymbol(clang::clangd::DocumentSymbol symbol) -> void {
    if (open_symbols_.empty()) {
      top_level_symbols_.push_back(std::move(symbol));
    } else {
      open_symbols_.back().children.push_back(symbol);
    }
  }

  // Starts a symbol potentially with children.
  auto StartSymbol(clang::clangd::DocumentSymbol symbol) -> void {
    open_symbols_.push_back(std::move(symbol));
  }

  auto HasOpenSymbol() const -> bool { return !open_symbols_.empty(); }

  // Completes a symbol, appending to parent list.
  auto EndSymbol() -> void {
    CARBON_CHECK(HasOpenSymbol());
    AddSymbol(open_symbols_.pop_back_val());
  }

  // Returns final top level symbols.
  auto Collect() -> std::vector<clang::clangd::DocumentSymbol> {
    CARBON_CHECK(!HasOpenSymbol());
    return std::move(top_level_symbols_);
  }

 private:
  std::vector<clang::clangd::DocumentSymbol> top_level_symbols_;
  llvm::SmallVector<clang::clangd::DocumentSymbol> open_symbols_;
};

// Constructs a Range from a closed interval of tokens [start, end].
static auto GetTokenRange(const Lex::TokenizedBuffer& tokens,
                          Lex::TokenIndex start, Lex::TokenIndex end)
    -> clang::clangd::Range {
  auto start_line = tokens.GetLine(start);
  auto start_col = tokens.GetColumnNumber(start);
  auto [end_line, end_col] = tokens.GetEndLoc(end);

  return clang::clangd::Range{
      .start = {.line = start_line.index, .character = start_col - 1},
      .end = {.line = end_line.index, .character = end_col - 1},
  };
}

// Finds a spanning range for the provided definition / declaration ast node.
// In the case of a definition start, will include the body as well.
static auto GetSymbolRange(const Parse::TreeAndSubtrees& tree_and_subtrees,
                           const Parse::NodeId& ast_node)
    -> clang::clangd::Range {
  const auto& tokens = tree_and_subtrees.tree().tokens();

  // The left-most node will always be the first node in postorder traversal.
  auto start_node = *tree_and_subtrees.postorder(ast_node).begin();

  auto start_token = tree_and_subtrees.tree().node_token(start_node);
  auto end_token = tree_and_subtrees.tree().node_token(ast_node);
  if (tokens.GetKind(end_token).is_opening_symbol()) {
    // DefinitionStart nodes use an opening token, so find its closing token to
    // span the entire class/function body.
    return GetTokenRange(tokens, start_token,
                         tokens.GetMatchedClosingToken(end_token));
  } else {
    return GetTokenRange(tokens, start_token, end_token);
  }
}

auto HandleDocumentSymbol(
    Context& context, const clang::clangd::DocumentSymbolParams& params,
    llvm::function_ref<
        auto(llvm::Expected<std::vector<clang::clangd::DocumentSymbol>>)->void>
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

    if (auto identifier = GetSymbolIdentifier(tree_and_subtrees, node_id)) {
      clang::clangd::DocumentSymbol symbol{
          .name = std::string(tokens.GetTokenText(*identifier)),
          .kind = symbol_kind,
          .range = GetSymbolRange(tree_and_subtrees, node_id),
          .selectionRange = GetTokenRange(tokens, *identifier, *identifier),
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

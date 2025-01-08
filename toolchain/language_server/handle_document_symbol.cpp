// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/shared_value_stores.h"
#include "toolchain/diagnostics/null_diagnostics.h"
#include "toolchain/language_server/handle.h"
#include "toolchain/lex/lex.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/parse/parse.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon::LanguageServer {

// Returns the text of first child of kind Parse::NodeKind::IdentifierName.
static auto GetIdentifierName(const SharedValueStores& value_stores,
                              const Lex::TokenizedBuffer& tokens,
                              const Parse::TreeAndSubtrees& p,
                              Parse::NodeId node)
    -> std::optional<llvm::StringRef> {
  for (auto ch : p.children(node)) {
    if (p.tree().node_kind(ch) == Parse::NodeKind::IdentifierName) {
      auto token = p.tree().node_token(ch);
      if (tokens.GetKind(token) == Lex::TokenKind::Identifier) {
        return value_stores.identifiers().Get(tokens.GetIdentifier(token));
      }
    }
  }
  return std::nullopt;
}

auto HandleDocumentSymbol(
    Context& context, const clang::clangd::DocumentSymbolParams& params,
    llvm::function_ref<
        void(llvm::Expected<std::vector<clang::clangd::DocumentSymbol>>)>
        on_done) -> void {
  SharedValueStores value_stores;
  llvm::vfs::InMemoryFileSystem vfs;
  auto lookup = context.files().Lookup(params.textDocument.uri.file());
  CARBON_CHECK(lookup);
  vfs.addFile(lookup.key(), /*mtime=*/0,
              llvm::MemoryBuffer::getMemBufferCopy(lookup.value()));

  auto buf =
      SourceBuffer::MakeFromFile(vfs, lookup.key(), NullDiagnosticConsumer());
  auto lexed = Lex::Lex(value_stores, *buf, NullDiagnosticConsumer());
  auto parsed = Parse::Parse(lexed, NullDiagnosticConsumer(), nullptr);
  Parse::TreeAndSubtrees tree_and_subtrees(lexed, parsed);
  std::vector<clang::clangd::DocumentSymbol> result;
  for (const auto& node : parsed.postorder()) {
    clang::clangd::SymbolKind symbol_kind;
    switch (parsed.node_kind(node)) {
      case Parse::NodeKind::FunctionDecl:
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
      case Parse::NodeKind::ClassDefinitionStart:
        symbol_kind = clang::clangd::SymbolKind::Class;
        break;
      default:
        continue;
    }

    if (auto name =
            GetIdentifierName(value_stores, lexed, tree_and_subtrees, node)) {
      auto tok = parsed.node_token(node);
      clang::clangd::Position pos{lexed.GetLineNumber(tok) - 1,
                                  lexed.GetColumnNumber(tok) - 1};

      clang::clangd::DocumentSymbol symbol{
          .name = std::string(*name),
          .kind = symbol_kind,
          .range = {.start = pos, .end = pos},
          .selectionRange = {.start = pos, .end = pos},
      };

      result.push_back(symbol);
    }
  }
  on_done(result);
}

}  // namespace Carbon::LanguageServer

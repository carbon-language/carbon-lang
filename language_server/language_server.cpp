// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "language_server.h"

#include "toolchain/diagnostics/null_diagnostics.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon::LS {

namespace {
template <typename T>
// TODO: handle parsing error
auto ParseJSON(llvm::json::Value& v) -> T {
  llvm::json::Path::Root root_path;
  T parsed;
  clang::clangd::fromJSON(v, parsed, root_path);
  return parsed;
}
}  // namespace

auto LanguageServer::onNotify(llvm::StringRef method, llvm::json::Value value)
    -> bool {
  if (method == "textDocument/didOpen") {
    auto params = ParseJSON<clang::clangd::DidOpenTextDocumentParams>(value);
    vfs_.addFile(
        params.textDocument.uri.file(), /*mtime=*/0,
        llvm::MemoryBuffer::getMemBufferCopy(params.textDocument.text));
  }
  if (method == "textDocument/didChange") {
    // TODO: sync files
  }
  return true;
}

auto LanguageServer::onCall(llvm::StringRef method, llvm::json::Value params,
                            llvm::json::Value id) -> bool {
  if (method == "initialize") {
    llvm::json::Object capabilities{{"documentSymbolProvider", true}};
    transport_->reply(
        id, llvm::json::Object{{"capabilities", std::move(capabilities)}});
  }
  if (method == "textDocument/documentSymbol") {
    auto symbols_params =
        ParseJSON<clang::clangd::DocumentSymbolParams>(params);
    auto symbols = Symbols(symbols_params);
    transport_->reply(id, symbols);
  }

  return true;
}

auto LanguageServer::onReply(llvm::json::Value /*id*/,
                             llvm::Expected<llvm::json::Value> /*result*/)
    -> bool {
  return true;
}

static auto getName(ParseTree& p, ParseTree::Node node)
    -> std::optional<llvm::StringRef> {
  for (auto ch : p.children(node)) {
    if (p.node_kind(ch) == ParseNodeKind::Name) {
      return p.GetNodeText(ch);
    }
  }
  return std::nullopt;
}

auto LanguageServer::Symbols(clang::clangd::DocumentSymbolParams& params)
    -> std::vector<clang::clangd::DocumentSymbol> {
  auto buf = SourceBuffer::CreateFromFile(vfs_, params.textDocument.uri.file());
  auto lexed = TokenizedBuffer::Lex(*buf, NullDiagnosticConsumer());
  auto parsed = ParseTree::Parse(lexed, NullDiagnosticConsumer(), nullptr);
  std::vector<clang::clangd::DocumentSymbol> result;
  // TODO: use preorder
  for (const auto& node : parsed.postorder()) {
    switch (parsed.node_kind(node)) {
      case ParseNodeKind::FunctionDeclaration:
      case ParseNodeKind::FunctionDefinitionStart:
      case ParseNodeKind::Namespace:
      case ParseNodeKind::InterfaceDefinitionStart:
        break;
      default:
        continue;
    }

    if (auto name = getName(parsed, node)) {
      auto tok = parsed.node_token(node);
      clang::clangd::Position pos{lexed.GetLineNumber(tok) - 1,
                                  lexed.GetColumnNumber(tok) - 1};

      clang::clangd::DocumentSymbol symbol{
          .name = std::string(*name),
          .kind = clang::clangd::SymbolKind::Function,
          .range = {.start = pos, .end = pos},
          .selectionRange = {.start = pos, .end = pos},
      };

      result.push_back(symbol);
    }
  }
  return result;
}

void LanguageServer::Start() {
  auto transport =
      clang::clangd::newJSONTransport(stdin, llvm::outs(), nullptr, true);
  LanguageServer ls(std::move(transport));
  auto error = ls.transport_->loop(ls);
  llvm::errs() << "Error: " << error << "\n";
}
}  // namespace Carbon::LS

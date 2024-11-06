// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/language_server/server.h"

#include "toolchain/base/shared_value_stores.h"
#include "toolchain/diagnostics/null_diagnostics.h"
#include "toolchain/lex/lex.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/parse/parse.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon::LanguageServer {

Server::Server(std::FILE* input_stream, llvm::raw_ostream& output_stream)
    : transport_(clang::clangd::newJSONTransport(input_stream, output_stream,
                                                 /*InMirror=*/nullptr,
                                                 /*Pretty=*/true)),
      binder_(handlers_, *this) {
  binder_.notification("textDocument/didOpen", this,
                       &Server::OnDidOpenTextDocument);
  binder_.notification("textDocument/didChange", this,
                       &Server::OnDidChangeTextDocument);
  binder_.method("initialize", this, &Server::OnInitialize);
  binder_.method("textDocument/documentSymbol", this,
                 &Server::OnDocumentSymbol);
}

auto Server::Run() -> ErrorOr<Success> {
  llvm::Error err = transport_->loop(*this);
  if (err.success()) {
    return Success();
  } else {
    std::string str;
    llvm::raw_string_ostream out(str);
    out << err;
    return Error(str);
  }
}

void Server::OnDidOpenTextDocument(
    clang::clangd::DidOpenTextDocumentParams const& params) {
  files_.emplace(params.textDocument.uri.file(), params.textDocument.text);
}

void Server::OnDidChangeTextDocument(
    clang::clangd::DidChangeTextDocumentParams const& params) {
  // Full text is sent if full sync is specified in capabilities.
  CARBON_CHECK(params.contentChanges.size() == 1);
  std::string file = params.textDocument.uri.file().str();
  files_[file] = params.contentChanges[0].text;
}

void Server::OnInitialize(
    clang::clangd::NoParams const& /*client_capabilities*/,
    clang::clangd::Callback<llvm::json::Object> cb) {
  llvm::json::Object capabilities{{"documentSymbolProvider", true},
                                  {"textDocumentSync", /*Full=*/1}};

  llvm::json::Object reply{{"capabilities", std::move(capabilities)}};
  cb(reply);
}

auto Server::onNotify(llvm::StringRef method, llvm::json::Value value) -> bool {
  if (method == "exit") {
    return false;
  }
  if (auto handler = handlers_.NotificationHandlers.find(method);
      handler != handlers_.NotificationHandlers.end()) {
    handler->second(std::move(value));
  } else {
    clang::clangd::log("unhandled notification {0}", method);
  }

  return true;
}

auto Server::onCall(llvm::StringRef method, llvm::json::Value params,
                    llvm::json::Value id) -> bool {
  if (auto handler = handlers_.MethodHandlers.find(method);
      handler != handlers_.MethodHandlers.end()) {
    // TODO: Improve this if add threads.
    handler->second(std::move(params),
                    [&](llvm::Expected<llvm::json::Value> reply) {
                      transport_->reply(id, std::move(reply));
                    });
  } else {
    transport_->reply(
        id, llvm::make_error<clang::clangd::LSPError>(
                "method not found", clang::clangd::ErrorCode::MethodNotFound));
  }

  return true;
}

auto Server::onReply(llvm::json::Value /*id*/,
                     llvm::Expected<llvm::json::Value> /*result*/) -> bool {
  return true;
}

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

void Server::OnDocumentSymbol(
    clang::clangd::DocumentSymbolParams const& params,
    clang::clangd::Callback<std::vector<clang::clangd::DocumentSymbol>> cb) {
  SharedValueStores value_stores;
  llvm::vfs::InMemoryFileSystem vfs;
  auto file = params.textDocument.uri.file().str();
  vfs.addFile(file, /*mtime=*/0,
              llvm::MemoryBuffer::getMemBufferCopy(files_.at(file)));

  auto buf = SourceBuffer::MakeFromFile(vfs, file, NullDiagnosticConsumer());
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
  cb(result);
}

}  // namespace Carbon::LanguageServer

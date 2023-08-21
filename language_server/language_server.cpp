// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "language_server/language_server.h"

#include "clang-tools-extra/clangd/Protocol.h"
#include "toolchain/diagnostics/null_diagnostics.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon::LS {

namespace {
template <typename T>
// Parse untyped JSON into a typed form.
auto ParseJSON(const llvm::json::Value& untyped, T& parsed) -> bool {
  llvm::json::Path::Root root_path;
  return clang::clangd::fromJSON(untyped, parsed, root_path);
}
}  // namespace

void LanguageServer::OnDidOpenTextDocument(
    clang::clangd::DidOpenTextDocumentParams const& params) {
  files_.emplace(params.textDocument.uri.file(), params.textDocument.text);
}

void LanguageServer::OnDidChangeTextDocument(
    clang::clangd::DidChangeTextDocumentParams const& params) {
  // full text is sent if full sync is specified in capabilities.
  assert(params.contentChanges.size() == 1);
  std::string file = params.textDocument.uri.file().str();
  files_[file] = params.contentChanges[0].text;
}

void LanguageServer::OnInitialize(
    clang::clangd::NoParams const& client_capabilities,
    clang::clangd::Callback<llvm::json::Object> cb) {
  llvm::json::Object capabilities{{"documentSymbolProvider", true},
                                  {"textDocumentSync", /*Full=*/1}};

  llvm::json::Object reply{{"capabilities", std::move(capabilities)}};
  cb(reply);
};

auto LanguageServer::onNotify(llvm::StringRef method, llvm::json::Value value)
    -> bool {
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

auto LanguageServer::onCall(llvm::StringRef method, llvm::json::Value params,
                            llvm::json::Value id) -> bool {
  if (auto handler = handlers_.MethodHandlers.find(method);
      handler != handlers_.MethodHandlers.end()) {
    // TODO: improve this if add threads
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

auto LanguageServer::onReply(llvm::json::Value /*id*/,
                             llvm::Expected<llvm::json::Value> /*result*/)
    -> bool {
  return true;
}

// Returns the text of first child of kind ParseNodeKind::Name.
static auto getName(ParseTree& p, ParseTree::Node node)
    -> std::optional<llvm::StringRef> {
  for (auto ch : p.children(node)) {
    if (p.node_kind(ch) == ParseNodeKind::Name) {
      return p.GetNodeText(ch);
    }
  }
  return std::nullopt;
}

void LanguageServer::OnDocumentSymbol(
    clang::clangd::DocumentSymbolParams const& params,
    clang::clangd::Callback<std::vector<clang::clangd::DocumentSymbol>> cb) {
  llvm::vfs::InMemoryFileSystem vfs;
  auto file = params.textDocument.uri.file().str();
  vfs.addFile(file, /*mtime=*/0,
              llvm::MemoryBuffer::getMemBufferCopy(files_.at(file)));

  auto buf = SourceBuffer::CreateFromFile(vfs, file);
  auto lexed = TokenizedBuffer::Lex(*buf, NullDiagnosticConsumer());
  auto parsed = ParseTree::Parse(lexed, NullDiagnosticConsumer(), nullptr);
  std::vector<clang::clangd::DocumentSymbol> result;
  for (const auto& node : parsed.postorder()) {
    clang::clangd::SymbolKind symbol_kind;
    switch (parsed.node_kind(node)) {
      case ParseNodeKind::FunctionDeclaration:
      case ParseNodeKind::FunctionDefinitionStart:
        symbol_kind = clang::clangd::SymbolKind::Function;
        break;
      case ParseNodeKind::Namespace:
        symbol_kind = clang::clangd::SymbolKind::Namespace;
        break;
      case ParseNodeKind::InterfaceDefinitionStart:
      case ParseNodeKind::NamedConstraintDefinitionStart:
        symbol_kind = clang::clangd::SymbolKind::Interface;
        break;
      case ParseNodeKind::ClassDefinitionStart:
        symbol_kind = clang::clangd::SymbolKind::Class;
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
          .kind = symbol_kind,
          .range = {.start = pos, .end = pos},
          .selectionRange = {.start = pos, .end = pos},
      };

      result.push_back(symbol);
    }
  }
  cb(result);
}

void LanguageServer::Start() {
  auto transport =
      clang::clangd::newJSONTransport(stdin, llvm::outs(), nullptr, true);
  LanguageServer ls(std::move(transport));
  clang::clangd::LSPBinder binder(ls.handlers_, ls);
  binder.notification("textDocument/didOpen", &ls,
                      &LanguageServer::OnDidOpenTextDocument);
  binder.notification("textDocument/didChange", &ls,
                      &LanguageServer::OnDidChangeTextDocument);
  binder.method("initialize", &ls, &LanguageServer::OnInitialize);
  binder.method("textDocument/documentSymbol", &ls,
                &LanguageServer::OnDocumentSymbol);
  auto error = ls.transport_->loop(ls);
  llvm::errs() << "Error: " << error << "\n";
}
}  // namespace Carbon::LS

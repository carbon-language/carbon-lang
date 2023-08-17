// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "language_server.h"

#include "third_party/clangd/Protocol.h"
#include "toolchain/diagnostics/null_diagnostics.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon::LS {

namespace {
template <typename T>
// TODO: handle parsing error
auto ParseJSON(llvm::json::Value& v, T& parsed) -> bool {
  llvm::json::Path::Root root_path;
  return clang::clangd::fromJSON(v, parsed, root_path);
}
}  // namespace

auto LanguageServer::onNotify(llvm::StringRef method, llvm::json::Value value)
    -> bool {
  if (method == "textDocument/didOpen") {
    clang::clangd::DidOpenTextDocumentParams params;
    if (!ParseJSON(value, params)) {
      return false;
    }
    files_.emplace(params.textDocument.uri.file(), params.textDocument.text);
  }
  if (method == "textDocument/didChange") {
    clang::clangd::DidChangeTextDocumentParams params;
    if (!ParseJSON(value, params)) {
      return false;
    }
    // full text is sent if full sync is specified in capabilities.
    assert(params.contentChanges.size() == 1);
    std::string file = params.textDocument.uri.file().str();
    files_[file] = params.contentChanges[0].text;
  }
  return true;
}

auto LanguageServer::onCall(llvm::StringRef method, llvm::json::Value params,
                            llvm::json::Value id) -> bool {
  if (method == "initialize") {
    llvm::json::Object capabilities{{"documentSymbolProvider", true},
                                    {"textDocumentSync", /*Full=*/1}};
    transport_->reply(
        id, llvm::json::Object{{"capabilities", std::move(capabilities)}});
  }
  if (method == "textDocument/documentSymbol") {
    clang::clangd::DocumentSymbolParams symbol_params;
    if (!ParseJSON(params, symbol_params)) {
      return false;
    }
    auto symbols = Symbols(symbol_params);
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
  llvm::vfs::InMemoryFileSystem vfs;
  auto file = params.textDocument.uri.file().str();
  vfs.addFile(file, /*mtime=*/0,
              llvm::MemoryBuffer::getMemBufferCopy(files_.at(file)));

  auto buf = SourceBuffer::CreateFromFile(vfs, file);
  auto lexed = TokenizedBuffer::Lex(*buf, NullDiagnosticConsumer());
  auto parsed = ParseTree::Parse(lexed, NullDiagnosticConsumer(), nullptr);
  std::vector<clang::clangd::DocumentSymbol> result;
  for (const auto& node : parsed.postorder()) {
    switch (parsed.node_kind(node)) {
      case ParseNodeKind::FunctionDeclaration:
      case ParseNodeKind::FunctionDefinitionStart:
      case ParseNodeKind::Namespace:
      case ParseNodeKind::InterfaceDefinitionStart:
      case ParseNodeKind::NamedConstraintDefinitionStart:
      case ParseNodeKind::ClassDefinitionStart:
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

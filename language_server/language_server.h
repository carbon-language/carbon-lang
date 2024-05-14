// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_LANGUAGE_SERVER_LANGUAGE_SERVER_H_
#define CARBON_LANGUAGE_SERVER_LANGUAGE_SERVER_H_
#include <unordered_map>
#include <vector>

#include "clang-tools-extra/clangd/LSPBinder.h"
#include "clang-tools-extra/clangd/Protocol.h"
#include "clang-tools-extra/clangd/Transport.h"
#include "clang-tools-extra/clangd/support/Function.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon::LS {
class LanguageServer : public clang::clangd::Transport::MessageHandler,
                       public clang::clangd::LSPBinder::RawOutgoing {
 public:
  // Start the language server.
  static void Start();

  // Transport::MessageHandler
  // Handlers returns true to keep processing messages, or false to shut down.

  // Handler called on notification by client.
  auto onNotify(llvm::StringRef method, llvm::json::Value value)
      -> bool override;
  // Handler called on method call by client.
  auto onCall(llvm::StringRef method, llvm::json::Value params,
              llvm::json::Value id) -> bool override;
  // Handler called on response of Transport::call.
  auto onReply(llvm::json::Value id, llvm::Expected<llvm::json::Value> result)
      -> bool override;

  // LSPBinder::RawOutgoing

  // Send method call to client
  void callMethod(llvm::StringRef method, llvm::json::Value params,
                  clang::clangd::Callback<llvm::json::Value> reply) override {
    // TODO: implement when needed
  }

  // Send notification to client
  void notify(llvm::StringRef method, llvm::json::Value params) override {
    transport_->notify(method, params);
  }

 private:
  const std::unique_ptr<clang::clangd::Transport> transport_;
  // content of files managed by the language client.
  std::unordered_map<std::string, std::string> files_;
  // handlers for client methods and notifications
  clang::clangd::LSPBinder::RawHandlers handlers_;

  explicit LanguageServer(std::unique_ptr<clang::clangd::Transport> transport)
      : transport_(std::move(transport)) {}

  // Typed handlers for notifications and method calls by client.

  // Client opened a document.
  void OnDidOpenTextDocument(
      clang::clangd::DidOpenTextDocumentParams const& params);

  // Client updated content of a document.
  void OnDidChangeTextDocument(
      clang::clangd::DidChangeTextDocumentParams const& params);

  // Capabilities negotiation
  void OnInitialize(clang::clangd::NoParams const& client_capabilities,
                    clang::clangd::Callback<llvm::json::Object> cb);

  // Code outline
  void OnDocumentSymbol(
      clang::clangd::DocumentSymbolParams const& params,
      clang::clangd::Callback<std::vector<clang::clangd::DocumentSymbol>> cb);
};

}  // namespace Carbon::LS

#endif  // CARBON_LANGUAGE_SERVER_LANGUAGE_SERVER_H_

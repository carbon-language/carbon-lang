// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LANGUAGE_SERVER_SERVER_H_
#define CARBON_TOOLCHAIN_LANGUAGE_SERVER_SERVER_H_

#include <memory>
#include <unordered_map>
#include <vector>

#include "clang-tools-extra/clangd/LSPBinder.h"
#include "clang-tools-extra/clangd/Protocol.h"
#include "clang-tools-extra/clangd/Transport.h"
#include "common/error.h"

namespace Carbon::LanguageServer {

// Provides a LSP implementation for Carbon.
class Server : public clang::clangd::Transport::MessageHandler,
               public clang::clangd::LSPBinder::RawOutgoing {
 public:
  // Prepares the server to run on the provided streams.
  explicit Server(std::FILE* input_stream, llvm::raw_ostream& output_stream);

  // Runs the server in a loop, returning the result. Currently this always
  // returns an error when the input stream is closed.
  auto Run() -> ErrorOr<Success>;

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
  auto callMethod(llvm::StringRef method, llvm::json::Value params,
                  clang::clangd::Callback<llvm::json::Value> reply)
      -> void override {
    // TODO: Implement when needed.
  }

  // Send notification to client
  auto notify(llvm::StringRef method, llvm::json::Value params)
      -> void override {
    transport_->notify(method, params);
  }

 private:
  // Typed handlers for notifications and method calls by client.

  // Client opened a document.
  auto OnDidOpenTextDocument(
      clang::clangd::DidOpenTextDocumentParams const& params) -> void;

  // Client updated content of a document.
  auto OnDidChangeTextDocument(
      clang::clangd::DidChangeTextDocumentParams const& params) -> void;

  // Capabilities negotiation
  auto OnInitialize(clang::clangd::NoParams const& client_capabilities,
                    clang::clangd::Callback<llvm::json::Object> cb) -> void;

  // Code outline
  auto OnDocumentSymbol(
      clang::clangd::DocumentSymbolParams const& params,
      clang::clangd::Callback<std::vector<clang::clangd::DocumentSymbol>> cb)
      -> void;

  const std::unique_ptr<clang::clangd::Transport> transport_;
  // content of files managed by the language client.
  std::unordered_map<std::string, std::string> files_;
  // handlers for client methods and notifications
  clang::clangd::LSPBinder::RawHandlers handlers_;
  // Binds client calls to member methods.
  clang::clangd::LSPBinder binder_;
};

}  // namespace Carbon::LanguageServer

#endif  // CARBON_TOOLCHAIN_LANGUAGE_SERVER_SERVER_H_

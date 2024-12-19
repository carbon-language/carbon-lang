// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LANGUAGE_SERVER_INCOMING_MESSAGES_H_
#define CARBON_TOOLCHAIN_LANGUAGE_SERVER_INCOMING_MESSAGES_H_

#include "clang-tools-extra/clangd/LSPBinder.h"
#include "clang-tools-extra/clangd/Transport.h"

namespace Carbon::LanguageServer {

// Handles LSP messages from the client (IDE extension) by forwarding them to
// `handlers_`.
//
// Handlers can return false to indicate server shutdown. Currently we only
// return true.
class IncomingMessages : public clang::clangd::Transport::MessageHandler {
 public:
  explicit IncomingMessages(clang::clangd::Transport* transport)
      : transport_(transport) {}

  // Forwards notifications.
  auto onNotify(llvm::StringRef method, llvm::json::Value value)
      -> bool override;

  // Calls the requested method.
  auto onCall(llvm::StringRef method, llvm::json::Value params,
              llvm::json::Value id) -> bool override;

  // Handles replies.
  // TODO: Implement when needed.
  auto onReply(llvm::json::Value /*id*/,
               llvm::Expected<llvm::json::Value> /*result*/) -> bool override {
    return true;
  }

  auto handlers() -> clang::clangd::LSPBinder::RawHandlers& {
    return handlers_;
  }

 private:
  // The connection to the client.
  clang::clangd::Transport* transport_;

  // Used with `LSPBinder` to attach message handlers.
  clang::clangd::LSPBinder::RawHandlers handlers_;
};

}  // namespace Carbon::LanguageServer

#endif  // CARBON_TOOLCHAIN_LANGUAGE_SERVER_INCOMING_MESSAGES_H_

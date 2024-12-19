// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LANGUAGE_SERVER_INCOMING_MESSAGES_H_
#define CARBON_TOOLCHAIN_LANGUAGE_SERVER_INCOMING_MESSAGES_H_

#include "clang-tools-extra/clangd/LSPBinder.h"
#include "clang-tools-extra/clangd/Transport.h"
#include "common/check.h"
#include "common/map.h"
#include "toolchain/language_server/context.h"
#include "toolchain/language_server/handler_registry.h"

namespace Carbon::LanguageServer {

// Handles LSP messages from the client (IDE extension) by forwarding them to
// `handlers_`.
//
// Handlers can return false to indicate server shutdown. Currently we only
// return true.
//
// TODO: Consider adding multithreading support for calls.
class IncomingMessages : public clang::clangd::Transport::MessageHandler {
 public:
  explicit IncomingMessages(clang::clangd::Transport* transport,
                            Context* context);

  // Calls the requested method.
  auto onCall(llvm::StringRef method, llvm::json::Value params,
              llvm::json::Value id) -> bool override;

  // Forwards notifications.
  auto onNotify(llvm::StringRef method, llvm::json::Value value)
      -> bool override;

  // Handles replies.
  // TODO: Implement when needed.
  auto onReply(llvm::json::Value /*id*/,
               llvm::Expected<llvm::json::Value> /*result*/) -> bool override {
    return true;
  }

 private:
  // The connection to the client.
  clang::clangd::Transport* transport_;
  // The context for handlers.
  Context* context_;

  // Handlers for LSP calls.
  Map<std::string, CallHandler> call_handlers_;

  // Handlers for LSP notifications.
  Map<std::string, NotificationHandler> notification_handlers_;
};

}  // namespace Carbon::LanguageServer

#endif  // CARBON_TOOLCHAIN_LANGUAGE_SERVER_INCOMING_MESSAGES_H_

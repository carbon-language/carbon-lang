// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/language_server/incoming_messages.h"

namespace Carbon::LanguageServer {

auto IncomingMessages::onNotify(llvm::StringRef method, llvm::json::Value value)
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

auto IncomingMessages::onCall(llvm::StringRef method, llvm::json::Value params,
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

}  // namespace Carbon::LanguageServer

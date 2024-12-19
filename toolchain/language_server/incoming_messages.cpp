// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/language_server/incoming_messages.h"

#include "toolchain/language_server/handler_registry.h"

namespace Carbon::LanguageServer {

IncomingMessages::IncomingMessages(clang::clangd::Transport* transport,
                                   Context* context)
    : transport_(transport), context_(context) {
  CARBON_CHECK(!CallHandlerRegistry::entries().empty());
  for (const auto& call_handler : CallHandlerRegistry::entries()) {
    auto name = call_handler.getName();
    auto result = call_handlers_.Insert(
        name, call_handler.instantiate()->GetHandler(name));
    CARBON_CHECK(result.is_inserted());
  }

  CARBON_CHECK(!NotificationHandlerRegistry::entries().empty());
  for (const auto& notification_handler :
       NotificationHandlerRegistry::entries()) {
    auto name = notification_handler.getName();
    auto result = notification_handlers_.Insert(
        name, notification_handler.instantiate()->GetHandler(name));
    CARBON_CHECK(result.is_inserted());
  }
}

auto IncomingMessages::onCall(llvm::StringRef method, llvm::json::Value params,
                              llvm::json::Value id) -> bool {
  if (auto result = call_handlers_.Lookup(method)) {
    (result.value())(*context_, std::move(params),
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

auto IncomingMessages::onNotify(llvm::StringRef method, llvm::json::Value value)
    -> bool {
  if (method == "exit") {
    return false;
  }
  if (auto result = notification_handlers_.Lookup(method)) {
    (result.value())(*context_, std::move(value));
  } else {
    clang::clangd::log("unhandled notification {0}", method);
  }

  return true;
}

}  // namespace Carbon::LanguageServer

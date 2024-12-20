// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/language_server/incoming_messages.h"

#include "toolchain/language_server/handler_registry.h"

namespace Carbon::LanguageServer {

// Copies a registry's entries to the corresponding handler map.
template <typename RegistryT, typename HandlerT>
static auto BuildHandlerMap(Map<std::string, HandlerT>& handlers) -> void {
  CARBON_CHECK(!RegistryT::entries().empty(),
               "An empty registry may mean a linking error");
  for (const auto& handler_entry : RegistryT::entries()) {
    auto name = handler_entry.getName();
    auto result =
        handlers.Insert(name, handler_entry.instantiate()->GetHandler(name));
    CARBON_CHECK(result.is_inserted(), "Duplicate handler: {0}", name);
  }
}

IncomingMessages::IncomingMessages(clang::clangd::Transport* transport,
                                   Context* context)
    : transport_(transport), context_(context) {
  BuildHandlerMap<CallHandlerRegistry>(call_handlers_);
  BuildHandlerMap<NotificationHandlerRegistry>(notification_handlers_);
}

auto IncomingMessages::onCall(llvm::StringRef name, llvm::json::Value params,
                              llvm::json::Value id) -> bool {
  if (auto result = call_handlers_.Lookup(name)) {
    (result.value())(*context_, std::move(params),
                     [&](llvm::Expected<llvm::json::Value> reply) {
                       transport_->reply(id, std::move(reply));
                     });
  } else {
    transport_->reply(id, llvm::make_error<clang::clangd::LSPError>(
                              llvm::formatv("call `{0}` not found", name),
                              clang::clangd::ErrorCode::MethodNotFound));
  }

  return true;
}

auto IncomingMessages::onNotify(llvm::StringRef name, llvm::json::Value value)
    -> bool {
  if (name == "exit") {
    return false;
  }
  if (auto result = notification_handlers_.Lookup(name)) {
    (result.value())(*context_, std::move(value));
  } else {
    clang::clangd::log("notification `{0}` not found", name);
  }

  return true;
}

}  // namespace Carbon::LanguageServer

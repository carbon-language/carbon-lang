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

namespace Carbon::LanguageServer {

// Handles LSP messages from the client (IDE extension) by forwarding them to
// `handlers_`.
//
// Handlers can return false to indicate server shutdown, although that's only
// used for the `exit` notification.
//
// TODO: Consider adding multithreading support for calls.
class IncomingMessages : public clang::clangd::Transport::MessageHandler {
 public:
  explicit IncomingMessages(clang::clangd::Transport* transport,
                            Context* context);

  // Dispatches calls to the appropriate entry in `call_handlers_`. Always
  // returns true.
  auto onCall(llvm::StringRef name, llvm::json::Value params,
              llvm::json::Value id) -> bool override;

  // Dispatches notifications to the appropriate entry in
  // `notification_handlers_`, except for `exit` which directly returns false.
  auto onNotify(llvm::StringRef name, llvm::json::Value value) -> bool override;

  // Handles replies. Always returns true.
  auto onReply(llvm::json::Value /*id*/,
               llvm::Expected<llvm::json::Value> /*result*/) -> bool override;

 private:
  // These are the signatures expected for handlers.
  using CallHandler = std::function<void(
      Context& context, llvm::json::Value raw_params,
      llvm::function_ref<void(llvm::Expected<llvm::json::Value>)> on_done)>;
  using NotificationHandler =
      std::function<void(Context& context, llvm::json::Value raw_params)>;

  template <typename ParamsT, typename ResultT>
  auto AddCallHandler(
      llvm::StringRef name,
      void (*handler)(Context&, const ParamsT&,
                      llvm::function_ref<void(llvm::Expected<ResultT>)>))
      -> void;
  template <typename ParamsT>
  auto AddNotificationHandler(llvm::StringRef name,
                              void (*handler)(Context&, const ParamsT&))
      -> void;

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

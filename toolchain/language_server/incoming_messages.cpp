// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/language_server/incoming_messages.h"

#include "common/ostream.h"
#include "common/raw_string_ostream.h"
#include "toolchain/language_server/handle.h"

namespace Carbon::LanguageServer {

// Parses a JSON value into a specific parameter type. The name of the method is
// used when producing errors.
template <typename ParamsT>
inline auto Parse(llvm::StringRef name, const llvm::json::Value& raw_params)
    -> llvm::Expected<ParamsT> {
  ParamsT params;
  llvm::json::Path::Root root;
  if (!clang::clangd::fromJSON(raw_params, params, root)) {
    return llvm::make_error<clang::clangd::LSPError>(
        llvm::formatv("in call to `{0}`, JSON parse failed: {1}", name,
                      llvm::fmt_consume(root.getError())),
        clang::clangd::ErrorCode::InvalidParams);
  }
  return std::move(params);
}

template <typename ParamsT, typename ResultT>
auto IncomingMessages::AddCallHandler(
    llvm::StringRef name,
    void (*handler)(Context&, const ParamsT&,
                    llvm::function_ref<void(llvm::Expected<ResultT>)>))
    -> void {
  CallHandler parsing_handler =
      [name, handler](
          Context& context, llvm::json::Value raw_params,
          llvm::function_ref<void(llvm::Expected<llvm::json::Value>)> on_done)
      -> void {
    auto params = Parse<ParamsT>(name, raw_params);
    if (!params) {
      on_done(params.takeError());
      return;
    }
    handler(context, *params, on_done);
  };
  auto result = call_handlers_.Insert(name, parsing_handler);
  CARBON_CHECK(result.is_inserted(), "Duplicate handler: {0}", name);
}

template <typename ParamsT>
auto IncomingMessages::AddNotificationHandler(llvm::StringRef name,
                                              void (*handler)(Context&,
                                                              const ParamsT&))
    -> void {
  NotificationHandler parsing_handler =
      [name, handler](Context& context, llvm::json::Value raw_params) -> void {
    auto params = Parse<ParamsT>(name, raw_params);
    if (!params) {
      // TODO: Maybe we should do something more with this error?
      llvm::consumeError(params.takeError());
    }
    handler(context, *params);
  };
  auto result = notification_handlers_.Insert(name, parsing_handler);
  CARBON_CHECK(result.is_inserted(), "Duplicate handler: {0}", name);
}

IncomingMessages::IncomingMessages(clang::clangd::Transport* transport,
                                   Context* context)
    : transport_(transport), context_(context) {
  AddCallHandler("textDocument/documentSymbol", &HandleDocumentSymbol);
  AddCallHandler("initialize", &HandleInitialize);
  AddNotificationHandler("textDocument/didChange",
                         &HandleDidChangeTextDocument);
  AddNotificationHandler("textDocument/didOpen", &HandleDidOpenTextDocument);
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
                              llvm::formatv("unsupported call `{0}`", name),
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
    CARBON_DIAGNOSTIC(LanguageServerUnsupportedNotification, Warning,
                      "unsupported notification `{0}`", std::string);
    context_->no_loc_emitter().Emit(LanguageServerUnsupportedNotification,
                                    name.str());
  }

  return true;
}

auto IncomingMessages::onReply(llvm::json::Value id,
                               llvm::Expected<llvm::json::Value> result)
    -> bool {
  RawStringOstream id_str;
  id_str << id;
  RawStringOstream result_str;
  if (result) {
    result_str << result.get();
  } else {
    result_str << result.takeError();
  }
  CARBON_DIAGNOSTIC(LanguageServerUnexpectedReply, Warning,
                    "unexpected reply to request ID {0}: {1}", std::string,
                    std::string);
  context_->no_loc_emitter().Emit(LanguageServerUnexpectedReply,
                                  id_str.TakeStr(), result_str.TakeStr());
  return true;
}

}  // namespace Carbon::LanguageServer

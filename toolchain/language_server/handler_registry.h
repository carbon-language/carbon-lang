// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LANGUAGE_SERVER_HANDLER_REGISTRY_H_
#define CARBON_TOOLCHAIN_LANGUAGE_SERVER_HANDLER_REGISTRY_H_

#include "clang-tools-extra/clangd/Protocol.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Registry.h"

// This provides handler registration for `IncomingMessages`, allowing us to
// have functions and their endpoints to be grouped together in `handle_* files.
// Each `Handle*` function should have a corresponding registration object. It
// will look like:
//
//   static RegisterCallHandler<HandleDocumentSymbol> register_call(
//       "textDocument/didChange", "");
//
// In this example:
// - "RegisterCallHandler" indicates that this is a call (versus a
//   notification).
// - "HandleDocumentSymbol" is the function being registered.
// - "textDocument/didChange" is the name of the LSP endpoint.
// - The second string parameter is empty and unused at present.
//
// The registered handlers are automatically added to `IncomingMessages` on
// construction.

namespace Carbon::LanguageServer {

class Context;

// These are the signatures expected by `IncomingMessages` for handlers.
using CallHandler = std::function<void(
    Context& context, llvm::json::Value raw_param,
    llvm::function_ref<void(llvm::Expected<llvm::json::Value>)> on_done)>;
using NotificationHandler =
    std::function<void(Context& context, llvm::json::Value raw_param)>;

// We have a base registry entry which can produce the appropriate handler.
template <typename HandlerT>
class HandlerRegistryEntry {
 public:
  virtual ~HandlerRegistryEntry() = default;
  virtual auto GetHandler(llvm::StringRef name) -> HandlerT = 0;
};

// A separate registry is provided for each signature.
using CallHandlerRegistry = llvm::Registry<HandlerRegistryEntry<CallHandler>>;
using NotificationHandlerRegistry =
    llvm::Registry<HandlerRegistryEntry<NotificationHandler>>;

namespace Internal {

// Defined below.
template <auto Fn>
class CallHandlerWrapper;
template <auto Fn>
class NotificationHandlerWrapper;

}  // namespace Internal

// These provide a simplified way to register a handler. See the top of the file
// for documentation.
template <auto Fn>
using RegisterCallHandler =
    CallHandlerRegistry::Add<Internal::CallHandlerWrapper<Fn>>;
template <auto Fn>
using RegisterNotificationHandler =
    NotificationHandlerRegistry::Add<Internal::NotificationHandlerWrapper<Fn>>;

// Only internal implementation details are below.

namespace Internal {

// Parses a JSON value into a specific parameter type.
template <typename ParamT>
inline auto Parse(const llvm::json::Value& raw_param,
                  llvm::StringRef payload_name, llvm::StringRef payload_kind)
    -> llvm::Expected<ParamT> {
  ParamT result;
  llvm::json::Path::Root root;
  if (!clang::clangd::fromJSON(raw_param, result, root)) {
    return llvm::make_error<clang::clangd::LSPError>(
        llvm::formatv("failed to decode {0} {1}: {2}", payload_name,
                      payload_kind, llvm::fmt_consume(root.getError())),
        clang::clangd::ErrorCode::InvalidParams);
  }
  return std::move(result);
}

// Adapts a typed handler to `CallHandler` for `CallHandlerWrapper`.
template <typename ParamT, typename ResultT>
inline auto ParseForCallHandler(
    llvm::StringRef name,
    void (*handler)(Context&, const ParamT&,
                    llvm::function_ref<void(llvm::Expected<ResultT>)>))
    -> CallHandler {
  return
      [name, handler](
          Context& context, llvm::json::Value raw_param,
          llvm::function_ref<void(llvm::Expected<llvm::json::Value>)> on_done)
          -> void {
        auto param = Parse<ParamT>(raw_param, name, "request");
        if (!param) {
          on_done(param.takeError());
          return;
        }
        handler(context, *param, on_done);
      };
}

// Adapts a typed handler to `NotificationHandler` for
// `NotificationHandlerWrapper`.
template <typename ParamT>
inline auto ParseForNotificationHandler(llvm::StringRef name,
                                        void (*handler)(Context&,
                                                        const ParamT&))
    -> NotificationHandler {
  return
      [name, handler](Context& context, llvm::json::Value raw_param) -> void {
        auto param = Parse<ParamT>(raw_param, name, "request");
        if (!param) {
          // TODO: Maybe we should do something more with this error?
          llvm::consumeError(param.takeError());
        }
        handler(context, *param);
      };
}

// For a given handler's function, we produce a wrapper that adapts the actual
// signature and does casting with `llvm::json::Value`. These get instantiated
// when accessing `instantiate()` through `CallHandlerRegistry::entries()`.
template <auto Fn>
class CallHandlerWrapper : public HandlerRegistryEntry<CallHandler> {
 public:
  auto GetHandler(llvm::StringRef name) -> CallHandler override {
    return ParseForCallHandler(name, Fn);
  }
};
template <auto Fn>
class NotificationHandlerWrapper
    : public HandlerRegistryEntry<NotificationHandler> {
 public:
  auto GetHandler(llvm::StringRef name) -> NotificationHandler override {
    return ParseForNotificationHandler(name, Fn);
  }
};

}  // namespace Internal

}  // namespace Carbon::LanguageServer

#endif  // CARBON_TOOLCHAIN_LANGUAGE_SERVER_HANDLER_REGISTRY_H_

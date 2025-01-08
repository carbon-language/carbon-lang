// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LANGUAGE_SERVER_OUTGOING_MESSAGES_H_
#define CARBON_TOOLCHAIN_LANGUAGE_SERVER_OUTGOING_MESSAGES_H_

#include "clang-tools-extra/clangd/LSPBinder.h"
#include "clang-tools-extra/clangd/Transport.h"

namespace Carbon::LanguageServer {

// Handles sending LSP messages to the client (IDE extension).
class OutgoingMessages : public clang::clangd::LSPBinder::RawOutgoing {
 public:
  explicit OutgoingMessages(clang::clangd::Transport* transport)
      : transport_(transport) {}

  // Calls a method on the client.
  // TODO: Implement when needed.
  auto callMethod(llvm::StringRef /*method*/, llvm::json::Value /*params*/,
                  clang::clangd::Callback<llvm::json::Value> /*reply*/)
      -> void override {}

  // Sets a notification to the client.
  auto notify(llvm::StringRef method, llvm::json::Value params)
      -> void override {
    transport_->notify(method, params);
  }

 private:
  // The connection to the client.
  clang::clangd::Transport* transport_;
};

}  // namespace Carbon::LanguageServer

#endif  // CARBON_TOOLCHAIN_LANGUAGE_SERVER_OUTGOING_MESSAGES_H_

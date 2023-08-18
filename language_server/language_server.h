// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_LANGUAGE_SERVER_LANGUAGE_SERVER_H_
#define CARBON_LANGUAGE_SERVER_LANGUAGE_SERVER_H_
#include <unordered_map>
#include <vector>

#include "third_party/clangd/Protocol.h"
#include "third_party/clangd/Transport.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon::LS {
class LanguageServer : public clang::clangd::Transport::MessageHandler {
 public:
  static void Start();
  explicit LanguageServer(std::unique_ptr<clang::clangd::Transport> transport)
      : transport_(std::move(transport)) {}

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

 private:
  const std::unique_ptr<clang::clangd::Transport> transport_;
  // content of files managed by the language client.
  std::unordered_map<std::string, std::string> files_;

  auto Symbols(clang::clangd::DocumentSymbolParams const& params)
      -> std::vector<clang::clangd::DocumentSymbol>;
};

}  // namespace Carbon::LS

#endif  // CARBON_LANGUAGE_SERVER_LANGUAGE_SERVER_H_

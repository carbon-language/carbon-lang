// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_LANGUAGE_SERVER_LANGUAGE_SERVER_H
#define CARBON_LANGUAGE_SERVER_LANGUAGE_SERVER_H
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

  auto onNotify(llvm::StringRef method, llvm::json::Value value)
      -> bool override;
  auto onCall(llvm::StringRef method, llvm::json::Value params,
              llvm::json::Value id) -> bool override;
  auto onReply(llvm::json::Value id, llvm::Expected<llvm::json::Value> result)
      -> bool override;

 private:
  std::unique_ptr<clang::clangd::Transport> transport_;
  std::unordered_map<std::string, std::string> files_;

  auto Symbols(clang::clangd::DocumentSymbolParams& params)
      -> std::vector<clang::clangd::DocumentSymbol>;
};

}  // namespace Carbon::LS

#endif  // CARBON_LANGUAGE_SERVER_LANGUAGE_SERVER_H

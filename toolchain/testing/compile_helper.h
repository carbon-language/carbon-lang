// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_TESTING_COMPILE_HELPER_H_
#define CARBON_TOOLCHAIN_TESTING_COMPILE_HELPER_H_

#include <forward_list>

#include "llvm/Support/VirtualFileSystem.h"
#include "toolchain/diagnostics/diagnostic_consumer.h"
#include "toolchain/lex/lex.h"
#include "toolchain/parse/parse.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon::Testing {

// A test helper for compile-related functionality.
class CompileHelper {
 public:
  // Returns the result of lex.
  auto GetTokenizedBuffer(llvm::StringRef text,
                          DiagnosticConsumer* consumer = nullptr)
      -> Lex::TokenizedBuffer& {
    auto& source = GetSourceBuffer(text);

    value_store_storage_.emplace_front();
    token_storage_.push_front(Lex::Lex(value_store_storage_.front(), source,
                                       consumer ? *consumer : consumer_));
    return token_storage_.front();
  }

  auto GetTokenizedBufferWithSharedValueStore(llvm::StringRef text)
      -> std::pair<Lex::TokenizedBuffer&, SharedValueStores&> {
    auto& tokens = GetTokenizedBuffer(text);
    return {tokens, value_store_storage_.front()};
  }

  // Returns the result of parse.
  auto GetTree(llvm::StringRef text) -> Parse::Tree& {
    auto& tokens = GetTokenizedBuffer(text);
    tree_storage_.push_front(Parse::Parse(tokens, consumer_,
                                          /*vlog_stream=*/nullptr));
    return tree_storage_.front();
  }

  // Returns the result of parse (with extra subtree information).
  auto GetTreeAndSubtrees(llvm::StringRef text) -> Parse::TreeAndSubtrees& {
    auto& tree = GetTree(text);
    tree_and_subtrees_storage_.push_front(
        Parse::TreeAndSubtrees(token_storage_.front(), tree));
    return tree_and_subtrees_storage_.front();
  }

  // Returns the results of both lex and parse (with extra subtree information).
  auto GetTokenizedBufferWithTreeAndSubtrees(llvm::StringRef text)
      -> std::pair<Lex::TokenizedBuffer&, Parse::TreeAndSubtrees&> {
    auto& tree_and_subtrees = GetTreeAndSubtrees(text);
    return {token_storage_.front(), tree_and_subtrees};
  }

 private:
  // Produces a source buffer for the input text.s
  auto GetSourceBuffer(llvm::StringRef text) -> SourceBuffer& {
    std::string filename = llvm::formatv("test{0}.carbon", ++file_index_);
    CARBON_CHECK(fs_.addFile(filename, /*ModificationTime=*/0,
                             llvm::MemoryBuffer::getMemBuffer(text)));
    source_storage_.push_front(
        std::move(*SourceBuffer::MakeFromFile(fs_, filename, consumer_)));
    return source_storage_.front();
  }

  // Diagnostics will be printed to console.
  DiagnosticConsumer& consumer_ = ConsoleDiagnosticConsumer();

  // An index to generate unique filenames.
  int file_index_ = 0;

  // A filesystem for storing test files.
  llvm::vfs::InMemoryFileSystem fs_;

  // Storage for various compile artifacts.
  std::forward_list<SourceBuffer> source_storage_;
  std::forward_list<SharedValueStores> value_store_storage_;
  std::forward_list<Lex::TokenizedBuffer> token_storage_;
  std::forward_list<Parse::Tree> tree_storage_;
  std::forward_list<Parse::TreeAndSubtrees> tree_and_subtrees_storage_;
};

}  // namespace Carbon::Testing

#endif  // CARBON_TOOLCHAIN_TESTING_COMPILE_HELPER_H_

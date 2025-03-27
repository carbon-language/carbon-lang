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
                          Diagnostics::Consumer* consumer = nullptr)
      -> Lex::TokenizedBuffer&;

  // Returns the result of lex along with shared values.
  auto GetTokenizedBufferWithSharedValueStore(
      llvm::StringRef text, Diagnostics::Consumer* consumer = nullptr)
      -> std::pair<Lex::TokenizedBuffer&, SharedValueStores&>;

  // Returns the result of parse.
  auto GetTree(llvm::StringRef text) -> Parse::Tree&;

  // Returns the result of parse (with extra subtree information).
  auto GetTreeAndSubtrees(llvm::StringRef text) -> Parse::TreeAndSubtrees&;

  // Returns the results of both lex and parse (with extra subtree information).
  auto GetTokenizedBufferWithTreeAndSubtrees(llvm::StringRef text)
      -> std::pair<Lex::TokenizedBuffer&, Parse::TreeAndSubtrees&>;

 private:
  // Produces a source buffer for the input text.
  auto GetSourceBuffer(llvm::StringRef text) -> SourceBuffer&;

  // Diagnostics will be printed to console.
  Diagnostics::Consumer& consumer_ = Diagnostics::ConsoleConsumer();

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

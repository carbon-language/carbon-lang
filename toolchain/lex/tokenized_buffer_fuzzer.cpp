// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "common/check.h"
#include "llvm/ADT/StringRef.h"
#include "testing/fuzzing/libfuzzer.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/diagnostics/null_diagnostics.h"
#include "toolchain/lex/lex.h"

namespace Carbon::Testing {

// NOLINTNEXTLINE: Match the documented fuzzer entry point declaration style.
extern "C" int LLVMFuzzerTestOneInput(const unsigned char* data, size_t size) {
  // Ignore large inputs.
  // TODO: Investigate replacement with an error limit. Content with errors on
  // escaped quotes (`\"` repeated) have O(M * N) behavior for M errors in a
  // file length N, so either that will need to also be fixed or M will need to
  // shrink for large (1MB+) inputs.
  // This also affects parse/parse_fuzzer.cpp.
  if (size > 100000) {
    return 0;
  }
  static constexpr llvm::StringLiteral TestFileName = "test.carbon";
  llvm::vfs::InMemoryFileSystem fs;
  llvm::StringRef data_ref(reinterpret_cast<const char*>(data), size);
  CARBON_CHECK(fs.addFile(
      TestFileName, /*ModificationTime=*/0,
      llvm::MemoryBuffer::getMemBuffer(data_ref, /*BufferName=*/TestFileName,
                                       /*RequiresNullTerminator=*/false)));
  auto source =
      SourceBuffer::MakeFromFile(fs, TestFileName, Diagnostics::NullConsumer());

  SharedValueStores value_stores;
  auto buffer = Lex::Lex(value_stores, *source, Diagnostics::NullConsumer());
  if (buffer.has_errors()) {
    return 0;
  }

  // Walk the lexed and tokenized buffer to ensure it isn't corrupt in some way.
  //
  // TODO: We should enhance this to do more sanity checks on the resulting
  // token stream.
  for (Lex::TokenIndex token : buffer.tokens()) {
    int line_number = buffer.GetLineNumber(token);
    CARBON_CHECK(line_number > 0, "Invalid line number!");
    CARBON_CHECK(line_number < INT_MAX, "Invalid line number!");
    int column_number = buffer.GetColumnNumber(token);
    CARBON_CHECK(column_number > 0, "Invalid line number!");
    CARBON_CHECK(column_number < INT_MAX, "Invalid line number!");
  }

  return 0;
}

}  // namespace Carbon::Testing

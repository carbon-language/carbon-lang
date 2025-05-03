// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstring>

#include "llvm/ADT/StringRef.h"
#include "testing/fuzzing/libfuzzer.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/diagnostics/null_diagnostics.h"
#include "toolchain/lex/lex.h"
#include "toolchain/parse/parse.h"

namespace Carbon::Testing {

// NOLINTNEXTLINE: Match the documented fuzzer entry point declaration style.
extern "C" int LLVMFuzzerTestOneInput(const unsigned char* data, size_t size) {
  // Ignore large inputs.
  // TODO: See tokenized_buffer_fuzzer.cpp.
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

  // Lex the input.
  SharedValueStores value_stores;
  auto tokens = Lex::Lex(value_stores, *source, Diagnostics::NullConsumer());
  if (tokens.has_errors()) {
    return 0;
  }

  // Now parse it into a tree. Note that parsing will (when asserts are enabled)
  // walk the entire tree to verify it so we don't have to do that here.
  Parse::Parse(tokens, Diagnostics::NullConsumer(), /*vlog_stream=*/nullptr);
  return 0;
}

}  // namespace Carbon::Testing

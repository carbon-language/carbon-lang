// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>

#include "common/check.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/diagnostics/null_diagnostics.h"
#include "toolchain/lexer/tokenized_buffer.h"

namespace Carbon {

// NOLINTNEXTLINE: Match the documented fuzzer entry point declaration style.
extern "C" int LLVMFuzzerTestOneInput(const unsigned char* data,
                                      std::size_t size) {
  // We need two bytes of data to compute a file name length.
  if (size < 2) {
    return 0;
  }
  uint16_t raw_filename_length;
  std::memcpy(&raw_filename_length, data, 2);
  data += 2;
  size -= 2;
  size_t filename_length = raw_filename_length;

  // We need enough data to populate this filename length.
  if (size < filename_length) {
    return 0;
  }
  llvm::StringRef filename(reinterpret_cast<const char*>(data),
                           filename_length);
  data += filename_length;
  size -= filename_length;

  // The rest of the data is the source text.
  auto source = SourceBuffer::CreateFromText(
      llvm::StringRef(reinterpret_cast<const char*>(data), size), filename);

  auto buffer = TokenizedBuffer::Lex(source, NullDiagnosticConsumer());
  if (buffer.HasErrors()) {
    return 0;
  }

  // Walk the lexed and tokenized buffer to ensure it isn't corrupt in some way.
  //
  // TODO: We should enhance this to do more sanity checks on the resulting
  // token stream.
  for (TokenizedBuffer::Token token : buffer.Tokens()) {
    int line_number = buffer.GetLineNumber(token);
    (void)line_number;
    CHECK(line_number > 0) << "Invalid line number!";
    CHECK(line_number < INT_MAX) << "Invalid line number!";
    int column_number = buffer.GetColumnNumber(token);
    (void)column_number;
    CHECK(column_number > 0) << "Invalid line number!";
    CHECK(column_number < INT_MAX) << "Invalid line number!";
  }

  return 0;
}

}  // namespace Carbon

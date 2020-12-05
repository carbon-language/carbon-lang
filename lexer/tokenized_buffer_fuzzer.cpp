// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "diagnostics/diagnostic_emitter.h"
#include "lexer/tokenized_buffer.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon {

extern "C" int LLVMFuzzerTestOneInput(const unsigned char* data, size_t size) {
  // We need two bytes of data to compute a file name length.
  if (size < 2)
    return 0;
  unsigned short raw_filename_length;
  std::memcpy(&raw_filename_length, data, 2);
  data += 2;
  size -= 2;
  size_t filename_length = raw_filename_length;

  // We need enough data to populate this filename length.
  if (size < filename_length)
    return 0;
  llvm::StringRef filename(reinterpret_cast<const char*>(data),
                           filename_length);
  data += filename_length;
  size -= filename_length;

  // The rest of the data is the source text.
  auto source = SourceBuffer::CreateFromText(
      llvm::StringRef(reinterpret_cast<const char*>(data), size), filename);

  // Use a real diagnostic emitter to get lazy codepaths to execute.
  DiagnosticEmitter emitter = NullDiagnosticEmitter();

  auto buffer = TokenizedBuffer::Lex(source, emitter);
  if (buffer.HasErrors())
    return 0;

  // Walk the lexed and tokenized buffer to ensure it isn't corrupt in some way.
  //
  // TODO: We should enhance this to do more sanity checks on the resulting
  // token stream.
  for (TokenizedBuffer::Token token : buffer.Tokens()) {
    int line_number = buffer.GetLineNumber(token);
    (void)line_number;
    assert(line_number > 0 && "Invalid line number!");
    assert(line_number < INT_MAX && "Invalid line number!");
    int column_number = buffer.GetColumnNumber(token);
    (void)column_number;
    assert(column_number > 0 && "Invalid line number!");
    assert(column_number < INT_MAX && "Invalid line number!");
  }

  return 0;
}

}  // namespace Carbon

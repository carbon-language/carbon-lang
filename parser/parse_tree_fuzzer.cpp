// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstring>

#include "diagnostics/diagnostic_emitter.h"
#include "lexer/tokenized_buffer.h"
#include "llvm/ADT/StringRef.h"
#include "parser/parse_tree.h"

namespace Carbon {

// NOLINTNEXTLINE: Match the documented fuzzer entry point declaration style.
extern "C" int LLVMFuzzerTestOneInput(const unsigned char* data,
                                      std::size_t size) {
  // We need two bytes of data to compute a file name length.
  if (size < 2) {
    return 0;
  }
  unsigned short raw_filename_length;
  std::memcpy(&raw_filename_length, data, 2);
  data += 2;
  size -= 2;
  std::size_t filename_length = raw_filename_length;

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

  // Use a real diagnostic emitter to get lazy codepaths to execute.
  DiagnosticEmitter emitter = NullDiagnosticEmitter();

  // Lex the input.
  auto tokens = TokenizedBuffer::Lex(source, emitter);
  if (tokens.HasErrors()) {
    return 0;
  }

  // Now parse it into a tree. Note that parsing will (when asserts are enabled)
  // walk the entire tree to verify it so we don't have to do that here.
  ParseTree tree = ParseTree::Parse(tokens, emitter);
  if (tree.HasErrors()) {
    return 0;
  }

  // In the absence of parse errors, we should have exactly as many nodes as
  // tokens.
  assert(tree.Size() == tokens.Size() && "Unexpected number of tree nodes!");

  return 0;
}

}  // namespace Carbon

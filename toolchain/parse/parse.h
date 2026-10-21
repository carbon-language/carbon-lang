// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_PARSE_H_
#define CARBON_TOOLCHAIN_PARSE_PARSE_H_

#include "common/ostream.h"
#include "toolchain/diagnostics/emitter.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree.h"

namespace Carbon::Parse {

struct ParseOptions {
  // Options must be set individually, not through initialization.
  explicit ParseOptions() = default;

  // If set, a consumer for diagnostics. Otherwise, diagnostics go to stderr.
  Diagnostics::Consumer* consumer = nullptr;

  // If set, enables verbose output.
  llvm::raw_ostream* vlog_stream = nullptr;

  // If set, the parse tree will be dumped to this.
  llvm::raw_ostream* dump_stream = nullptr;

  // The format to dump the parse tree in if `dump_stream` is set.
  enum class DumpFormat {
    PrettyPostorder,
    YamlPostorder,
    YamlPreorder,
  };
  DumpFormat dump_format;
};

// Parses the token buffer into a `Tree`.
//
// This is the factory function which is used to build parse trees.
auto Parse(Lex::TokenizedBuffer& tokens, ParseOptions options) -> Tree;

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_PARSE_H_

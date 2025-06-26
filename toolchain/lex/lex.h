// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEX_LEX_H_
#define CARBON_TOOLCHAIN_LEX_LEX_H_

#include "toolchain/base/shared_value_stores.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon::Lex {

struct LexParams {
  // Must be non-null.
  SharedValueStores* value_stores;

  // Must be non-null.
  SourceBuffer* source;

  // Must be non-null.
  Diagnostics::Consumer* consumer;
};

// Lexes a buffer of source code into a tokenized buffer.
//
// The provided source buffer must outlive any returned `TokenizedBuffer`
// which will refer into the source.
auto Lex(LexParams params) -> TokenizedBuffer;

}  // namespace Carbon::Lex

#endif  // CARBON_TOOLCHAIN_LEX_LEX_H_

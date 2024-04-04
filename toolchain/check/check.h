// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CHECK_H_
#define CARBON_TOOLCHAIN_CHECK_CHECK_H_

#include "common/ostream.h"
#include "toolchain/base/value_store.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::Check {

// Constructs builtins. A single instance should be reused with CheckParseTree
// calls associated with a given compilation.
inline auto MakeBuiltins(SharedValueStores& value_stores) -> SemIR::File {
  return SemIR::File(value_stores);
}

// Checking information that's tracked per file.
struct Unit {
  SharedValueStores* value_stores;
  const Lex::TokenizedBuffer* tokens;
  const Parse::Tree* parse_tree;
  DiagnosticConsumer* consumer;
  // The generated IR. Unset on input, set on output.
  std::optional<SemIR::File>* sem_ir;
};

// Checks a group of parse trees. This will use imports to decide the order of
// checking.
auto CheckParseTrees(const SemIR::File& builtin_ir,
                     llvm::MutableArrayRef<Unit> units, bool prelude_import,
                     llvm::raw_ostream* vlog_stream) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CHECK_H_

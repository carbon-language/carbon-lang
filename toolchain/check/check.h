// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CHECK_H_
#define CARBON_TOOLCHAIN_CHECK_CHECK_H_

#include "common/ostream.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/base/timings.h"
#include "toolchain/check/sem_ir_diagnostic_converter.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::Check {

// Checking information that's tracked per file.
struct Unit {
  DiagnosticConsumer* consumer;
  SharedValueStores* value_stores;
  // The `timings` may be null if nothing is to be recorded.
  Timings* timings;
  const Lex::TokenizedBuffer* tokens;
  const Parse::Tree* parse_tree;

  // Returns a lazily constructed TreeAndSubtrees.
  llvm::function_ref<const Parse::TreeAndSubtrees&()>
      get_parse_tree_and_subtrees;

  // The unit's SemIR, provided as empty and filled in by CheckParseTrees.
  SemIR::File* sem_ir;

  // Diagnostic converters.
  Parse::NodeLocConverter* node_converter;
  SemIRDiagnosticConverter* sem_ir_converter;
};

// Checks a group of parse trees. This will use imports to decide the order of
// checking.
auto CheckParseTrees(llvm::MutableArrayRef<Unit> units, bool prelude_import,
                     llvm::raw_ostream* vlog_stream) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CHECK_H_

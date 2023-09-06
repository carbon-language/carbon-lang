// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CHECK_H_
#define CARBON_TOOLCHAIN_CHECK_CHECK_H_

#include "common/ostream.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::Check {

// Constructs builtins. A single instance should be reused with CheckParseTree
// calls associated with a given compilation.
inline auto MakeBuiltins() -> SemIR::File { return SemIR::File(); }

// Produces and checks the IR for the provided Parse::Tree.
extern auto CheckParseTree(const SemIR::File& builtin_ir,
                           const Lex::TokenizedBuffer& tokens,
                           const Parse::Tree& parse_tree,
                           DiagnosticConsumer& consumer,
                           llvm::raw_ostream* vlog_stream) -> SemIR::File;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CHECK_H_

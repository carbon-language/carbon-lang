// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEX_HELPERS_H_
#define CARBON_TOOLCHAIN_LEX_HELPERS_H_

#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon::Lex {

// Should guard calls to getAsInteger due to performance issues with large
// integers. Emits an error if the text cannot be lexed.
auto CanLexInt(Diagnostics::Emitter<const char*>& emitter, llvm::StringRef text)
    -> bool;

}  // namespace Carbon::Lex

#endif  // CARBON_TOOLCHAIN_LEX_HELPERS_H_

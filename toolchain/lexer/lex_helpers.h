// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_LEXER_LEX_HELPERS_H_
#define TOOLCHAIN_LEXER_LEX_HELPERS_H_

#include <optional>

#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon {

// Should guard calls to getAsInteger due to performance issues with large
// integers. Emits an error if the text cannot be lexed.
auto CanLexInteger(DiagnosticEmitter<const char*>& emitter,
                   llvm::StringRef text) -> bool;

}  // namespace Carbon

#endif  // TOOLCHAIN_LEXER_LEX_HELPERS_H_

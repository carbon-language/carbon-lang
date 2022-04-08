// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lexer/lex_helpers.h"

#include "llvm/Support/FormatVariadic.h"

namespace Carbon {

auto CanLexInteger(DiagnosticEmitter<const char*>& emitter,
                   llvm::StringRef text) -> bool {
  // llvm::getAsInteger is used for parsing, but it's quadratic and visibly slow
  // on large integer values. This limit exists to avoid hitting those limits.
  // Per https://github.com/carbon-language/carbon-lang/issues/980, it may be
  // feasible to optimize integer parsing in order to address performance if
  // this limit becomes an issue.
  //
  // 2^128 would be 39 decimal digits or 128 binary. In either case, this limit
  // is far above the threshold for normal integers.
  constexpr size_t DigitLimit = 1000;
  if (text.size() > DigitLimit) {
    CARBON_DIAGNOSTIC(
        TooManyDigits, Error,
        "Found a sequence of {0} digits, which is greater than the "
        "limit of {1}.",
        size_t, size_t);
    emitter.Emit(text.begin(), TooManyDigits, text.size(), DigitLimit);
    return false;
  }
  return true;
}

}  // namespace Carbon

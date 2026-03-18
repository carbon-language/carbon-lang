// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/helpers.h"

namespace Carbon::Lex {

auto CanLexInt(Diagnostics::Emitter<const char*>& emitter, llvm::StringRef text)
    -> bool {
  // Integer parsing has poor scaling characteristics for extremely large digit
  // amounts. We've done some performance work on this, but this limit exists to
  // avoid really extreme cases.
  //
  // 2^128 would be 39 decimal digits or 128 binary. In either case, this limit
  // is far above the threshold for normal ints.
  constexpr size_t DigitLimit = 10000;
  if (text.size() > DigitLimit) {
    CARBON_DIAGNOSTIC(
        TooManyDigits, Error,
        "found a sequence of {0} digits, which is greater than the "
        "limit of {1}",
        size_t, size_t);
    emitter.Emit(text.begin(), TooManyDigits, text.size(), DigitLimit);
    return false;
  }
  return true;
}

}  // namespace Carbon::Lex

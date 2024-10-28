// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/format.h"

#include "toolchain/format/formatter.h"

namespace Carbon::Format {

auto Format(const Lex::TokenizedBuffer& tokens, llvm::raw_ostream& out)
    -> bool {
  return Formatter(&tokens, &out).Run();
}

}  // namespace Carbon::Format

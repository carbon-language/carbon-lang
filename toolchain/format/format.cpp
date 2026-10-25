// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/format.h"

#include "toolchain/format/formatter.h"

namespace Carbon::Format {

auto Format(const Parse::Tree& tree, llvm::raw_ostream& out, const Style& style)
    -> bool {
  Formatter formatter(&tree, style);
  bool formatted_cleanly = formatter.Run();
  out << formatter.TakeOutput();
  return formatted_cleanly;
}

}  // namespace Carbon::Format

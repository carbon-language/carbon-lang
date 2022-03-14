// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics.h"

namespace Carbon {

Semantics Semantics::Analyze(const ParseTree& parse_tree,
                             DiagnosticConsumer& consumer) {
  Semantics semantics;
  return semantics;
}

}  // namespace Carbon

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_SEMANTICS_SEMANTICS_H_
#define TOOLCHAIN_SEMANTICS_SEMANTICS_H_

#include "toolchain/parser/parse_tree.h"

namespace Carbon {

// Provides semantic analysis on a ParseTree.
class Semantics {
 public:
  // Analyzes a parse tree and returns the constructed semantic information.
  static auto Analyze(const ParseTree& parse_tree, DiagnosticConsumer& consumer)
      -> Semantics;

 private:
  Semantics() = default;
};

}  // namespace Carbon

#endif  // TOOLCHAIN_SEMANTICS_SEMANTICS_H_

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FACTORY_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FACTORY_H_

#include <optional>

#include "llvm/ADT/StringMap.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/semantics_ir.h"

namespace Carbon {

// The main semantic analysis entry.
class SemanticsIRFactory {
 public:
  // Builds the SemanticsIR without doing any substantial semantic analysis.
  static auto Build(const ParseTree& parse_tree) -> SemanticsIR;

 private:
  explicit SemanticsIRFactory(const ParseTree& parse_tree)
      : semantics_(parse_tree) {}

  // Processes the roots of the ParseTree into semantics_, transitively
  // handling children.
  void ProcessRoots();

  // Turns a function node from the parse tree into a semantic function node,
  // adding it to the containing scope.
  void ProcessFunctionNode(SemanticsIR::Block& block,
                           ParseTree::Node decl_node);

  SemanticsIR semantics_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FACTORY_H_

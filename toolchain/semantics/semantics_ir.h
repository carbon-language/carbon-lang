// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/node_store.h"

namespace Carbon::Testing {
class SemanticsIRForTest;
}  // namespace Carbon::Testing

namespace Carbon {

// Provides semantic analysis on a ParseTree.
class SemanticsIR {
 public:
  // File-level declarations.
  auto root_block() const -> llvm::ArrayRef<Semantics::NodeRef> {
    return root_block_;
  }

  // Prints the node information.
  auto Print(llvm::raw_ostream& out, Semantics::NodeRef node_ref) const -> void;

 private:
  friend class SemanticsIRFactory;
  friend class Testing::SemanticsIRForTest;

  explicit SemanticsIR(const ParseTree& parse_tree)
      : parse_tree_(&parse_tree) {}

  Semantics::NodeStore nodes_;
  llvm::SmallVector<Semantics::NodeRef, 0> root_block_;

  const ParseTree* parse_tree_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_

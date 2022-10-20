// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon::Testing {
class SemanticsIRForTest;
}  // namespace Carbon::Testing

namespace Carbon {

// Provides semantic analysis on a ParseTree.
class SemanticsIR {
 public:
  // Prints the full IR.
  auto Print(llvm::raw_ostream& out) const -> void;

 private:
  friend class SemanticsIRFactory;

  explicit SemanticsIR(const ParseTree& parse_tree)
      : parse_tree_(&parse_tree) {}

  auto AddIdentifier(llvm::StringRef identifier) -> SemanticsIdentifierId {
    SemanticsIdentifierId id(identifiers_.size());
    identifiers_.push_back(identifier);
    return id;
  }

  auto AddIntegerLiteral(llvm::APInt integer_literal)
      -> SemanticsIntegerLiteralId {
    SemanticsIntegerLiteralId id(integer_literals_.size());
    integer_literals_.push_back(integer_literal);
    return id;
  }

  auto AddNode(SemanticsNode node) -> SemanticsNodeId {
    SemanticsNodeId id(nodes_.size());
    nodes_.push_back(node);
    return id;
  }

  llvm::SmallVector<llvm::StringRef> identifiers_;
  llvm::SmallVector<llvm::APInt> integer_literals_;
  llvm::SmallVector<SemanticsNode> nodes_;

  const ParseTree* parse_tree_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_

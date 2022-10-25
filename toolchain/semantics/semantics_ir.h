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
  SemanticsIR() { BuildBuiltins(); }

  // Adds the IR for the provided ParseTree.
  auto Build(const TokenizedBuffer& tokens, const ParseTree& parse_tree)
      -> void;

  // Prints the full IR.
  auto Print(llvm::raw_ostream& out) const -> void;

 private:
  friend class SemanticsParseTreeHandler;

  auto BuildBuiltins() -> void;

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

  // Starts a new node block.
  auto AddNodeBlock() -> SemanticsNodeBlockId {
    SemanticsNodeBlockId id(node_blocks_.size());
    node_blocks_.resize(node_blocks_.size() + 1);
    return id;
  }

  auto AddNode(SemanticsNodeBlockId block_id, SemanticsNode node) {
    auto& block = node_blocks_[block_id.id];
    SemanticsNodeId node_id(block.size());
    block.push_back(node);
    return node_id;
  }

  llvm::SmallVector<llvm::StringRef> identifiers_;
  llvm::SmallVector<llvm::APInt> integer_literals_;
  llvm::SmallVector<llvm::SmallVector<SemanticsNode>> node_blocks_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/lexer/numeric_literal.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon::Testing {
class SemanticsIRForTest;
}  // namespace Carbon::Testing

namespace Carbon {

// Provides semantic analysis on a ParseTree.
class SemanticsIR {
 public:
  // Produces the builtins.
  static auto MakeBuiltinIR() -> SemanticsIR;

  // Adds the IR for the provided ParseTree.
  static auto MakeFromParseTree(const SemanticsIR& builtin_ir,
                                const TokenizedBuffer& tokens,
                                const ParseTree& parse_tree,
                                DiagnosticConsumer& consumer,
                                llvm::raw_ostream* vlog_stream) -> SemanticsIR;

  // Prints the full IR.
  auto Print(llvm::raw_ostream& out) const -> void;

 private:
  friend class SemanticsParseTreeHandler;

  // As noted under cross_reference_irs_, the current IR must always be at
  // index 1. This is a constant for that.
  static constexpr auto ThisIR = SemanticsCrossReferenceIRId(1);

  // Passed to the constructor to disambiguiate from a copy constructor.
  enum FromBuiltins { Constructor };

  // For the builtin IR only.
  SemanticsIR() : SemanticsIR(*this) {}

  // For most IRs. Has an unused argument to separate from a copy constructor,
  // since it does not provide copy semantics (pass FromBuiltins).
  explicit SemanticsIR(const SemanticsIR& builtins, FromBuiltins /*unused*/);

  // Returns the type of the requested node.
  auto GetType(SemanticsNodeId node_id) -> SemanticsNodeId {
    return nodes_[node_id.index].type();
  }

  // Adds an identifier, returning an ID to reference it.
  // TODO: Deduplicate strings.
  // TODO: Probably make generic for all strings, including literals.
  auto AddIdentifier(llvm::StringRef identifier) -> SemanticsIdentifierId {
    SemanticsIdentifierId id(identifiers_.size());
    identifiers_.push_back(identifier);
    return id;
  }

  // Adds an integer literal, returning an ID to reference it.
  auto AddIntegerLiteral(llvm::APInt integer_literal)
      -> SemanticsIntegerLiteralId {
    SemanticsIntegerLiteralId id(integer_literals_.size());
    integer_literals_.push_back(integer_literal);
    return id;
  }

  // Adds an empty new node block, returning an ID to reference it and add
  // items.
  auto AddNodeBlock() -> SemanticsNodeBlockId {
    SemanticsNodeBlockId id(node_blocks_.size());
    node_blocks_.resize(node_blocks_.size() + 1);
    return id;
  }

  // Adds a node to a specified block, returning an ID to reference the node.
  auto AddNode(SemanticsNodeBlockId block_id, SemanticsNode node)
      -> SemanticsNodeId {
    SemanticsNodeId node_id(nodes_.size());
    nodes_.push_back(node);
    node_blocks_[block_id.index].push_back(node_id);
    return node_id;
  }

  // Related IRs. There will always be at least 2 entries, the builtin IR (used
  // for references of builtins) followed by the current IR (used for references
  // crossing node blocks).
  llvm::SmallVector<const SemanticsIR*> cross_reference_irs_;

  // Storage for identifiers.
  llvm::SmallVector<llvm::StringRef> identifiers_;

  // Storage for integer literals.
  llvm::SmallVector<llvm::APInt> integer_literals_;

  // All nodes. The first entries will always be cross-references to builtins,
  // at indices matching SemanticsBuiltinKind ordering.
  llvm::SmallVector<SemanticsNode> nodes_;

  // Storage for blocks within the IR. These reference entries in nodes_.
  llvm::SmallVector<llvm::SmallVector<SemanticsNodeId>> node_blocks_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_

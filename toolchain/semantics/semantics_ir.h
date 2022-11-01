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

// The ID of a cross-referenced IR (within cross_reference_irs_).
struct SemanticsCrossReferenceIRId {
  SemanticsCrossReferenceIRId() : id(-1) {}
  constexpr explicit SemanticsCrossReferenceIRId(int32_t id) : id(id) {}

  auto Print(llvm::raw_ostream& out) const -> void { out << "ir" << id; }

  int32_t id;
};

// A cross-reference between node blocks or IRs; essentially, anything that's
// not in the same SemanticsNodeBlock as the referencing node.
struct SemanticsCrossReference {
  SemanticsCrossReference() = default;
  SemanticsCrossReference(SemanticsCrossReferenceIRId ir,
                          SemanticsNodeBlockId node_block, SemanticsNodeId node)
      : ir(ir), node_block(node_block), node(node) {}

  auto Print(llvm::raw_ostream& out) const -> void {
    out << "xref(" << ir << ", " << node_block << ", " << node << ")";
  }

  SemanticsCrossReferenceIRId ir;
  SemanticsNodeBlockId node_block;
  SemanticsNodeId node;
};

// Provides semantic analysis on a ParseTree.
class SemanticsIR {
 public:
  // Produces the builtins.
  static auto MakeBuiltinIR() -> SemanticsIR;

  // Adds the IR for the provided ParseTree.
  static auto MakeFromParseTree(const SemanticsIR& builtin_ir,
                                const TokenizedBuffer& tokens,
                                const ParseTree& parse_tree) -> SemanticsIR;

  // Prints the full IR.
  auto Print(llvm::raw_ostream& out) const -> void;

 private:
  friend class SemanticsParseTreeHandler;

  // For the builtin IR only.
  SemanticsIR() : SemanticsIR(*this) {}
  // For most IRs.
  SemanticsIR(const SemanticsIR& builtins)
      : cross_reference_irs_({&builtins, this}),
        cross_references_(builtins.cross_references_) {}

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
    auto& block = node_blocks_[block_id.id];
    SemanticsNodeId node_id(block.size());
    block.push_back(node);
    return node_id;
  }

  // Related IRs. There will always be at least 2 entries, the builtin IR (used
  // for references of builtins) followed by the current IR (used for references
  // crossing node blocks).
  llvm::SmallVector<const SemanticsIR*> cross_reference_irs_;

  // Cross-references within the current IR across node blocks, and to other
  // IRs. The first entries will always be builtins, at indices matching
  // SemanticsBuiltinKind ordering.
  // TODO: Deduplicate cross-references after they can be added outside
  // builtins.
  llvm::SmallVector<SemanticsCrossReference> cross_references_;

  // Storage for identifiers.
  llvm::SmallVector<llvm::StringRef> identifiers_;

  // Storage for integer literals.
  llvm::SmallVector<llvm::APInt> integer_literals_;

  // Storage for blocks within the IR.
  llvm::SmallVector<llvm::SmallVector<SemanticsNode>> node_blocks_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_

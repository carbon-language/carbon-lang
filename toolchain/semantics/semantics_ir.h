// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
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

  // Returns true if there were errors creating the semantics IR.
  auto has_errors() const -> bool { return has_errors_; }

 private:
  friend class SemanticsParseTreeHandler;

  explicit SemanticsIR(const SemanticsIR* builtin_ir)
      : cross_reference_irs_({builtin_ir == nullptr ? this : builtin_ir}) {}

  // Returns the requested node.
  auto GetNode(SemanticsNodeId node_id) const -> const SemanticsNode& {
    return nodes_[node_id.index];
  }

  // Returns the type of the requested node.
  auto GetType(SemanticsNodeId node_id) -> SemanticsNodeId {
    return GetNode(node_id).type();
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

  // Adds an string, returning an ID to reference it.
  auto AddString(llvm::StringRef str) -> SemanticsStringId {
    // If the string has already been stored, return the corresponding ID.
    if (auto existing_id = GetString(str)) {
      return *existing_id;
    }

    // Allocate the string and store it in the map.
    SemanticsStringId id(strings_.size());
    strings_.push_back(str);
    CARBON_CHECK(string_to_id_.insert({str, id}).second);
    return id;
  }

  // Returns an ID for the string if it's previously been stored.
  auto GetString(llvm::StringRef str) -> std::optional<SemanticsStringId> {
    auto str_find = string_to_id_.find(str);
    if (str_find != string_to_id_.end()) {
      return str_find->second;
    }
    return std::nullopt;
  }

  bool has_errors_ = false;

  // Related IRs. There will always be at least 2 entries, the builtin IR (used
  // for references of builtins) followed by the current IR (used for references
  // crossing node blocks).
  llvm::SmallVector<const SemanticsIR*> cross_reference_irs_;

  // Storage for integer literals.
  llvm::SmallVector<llvm::APInt> integer_literals_;

  // Storage for strings. strings_ provides a list of allocated strings, while
  // string_to_id_ provides a mapping to identify strings.
  llvm::StringMap<SemanticsStringId> string_to_id_;
  llvm::SmallVector<llvm::StringRef> strings_;

  // All nodes. The first entries will always be cross-references to builtins,
  // at indices matching SemanticsBuiltinKind ordering.
  llvm::SmallVector<SemanticsNode> nodes_;

  // Storage for blocks within the IR. These reference entries in nodes_.
  llvm::SmallVector<llvm::SmallVector<SemanticsNodeId>> node_blocks_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_

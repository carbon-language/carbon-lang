// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_PARSE_TREE_HANDLER_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_PARSE_TREE_HANDLER_H_

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/semantics_ir.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

// Handles processing of a ParseTree for semantics.
class SemanticsParseTreeHandler {
 public:
  // Stores references for work.
  explicit SemanticsParseTreeHandler(const TokenizedBuffer& tokens,
                                     TokenDiagnosticEmitter& emitter,
                                     const ParseTree& parse_tree,
                                     SemanticsIR& semantics,
                                     llvm::raw_ostream* vlog_stream)
      : tokens_(&tokens),
        emitter_(&emitter),
        parse_tree_(&parse_tree),
        semantics_(&semantics),
        vlog_stream_(vlog_stream) {}

  // Outputs the ParseTree information into SemanticsIR.
  auto Build() -> void;

 private:
  // Prints the node_stack_ on stack dumps.
  class PrettyStackTraceNodeStack;
  // Prints the node_block_stack_ on stack dumps.
  class PrettyStackTraceNodeBlockStack;

  struct TraversalStackEntry {
    ParseTree::Node parse_node;
    // The result_id may be invalid if there's no result.
    SemanticsNodeId result_id;
  };
  static_assert(sizeof(TraversalStackEntry) == 8,
                "Unexpected TraversalStackEntry size");

  // Provides DenseMapInfo for SemanticsStringId.
  struct SemanticsStringIdMapInfo {
    static inline auto getEmptyKey() -> SemanticsStringId {
      return SemanticsStringId(llvm::DenseMapInfo<int32_t>::getEmptyKey());
    }
    static inline auto getTombstoneKey() -> SemanticsStringId {
      return SemanticsStringId(llvm::DenseMapInfo<int32_t>::getTombstoneKey());
    }

    static auto getHashValue(const SemanticsStringId& val) -> unsigned {
      return llvm::DenseMapInfo<int32_t>::getHashValue(val.index);
    }

    static auto isEqual(const SemanticsStringId& lhs,
                        const SemanticsStringId& rhs) -> bool {
      return lhs == rhs;
    }
  };

  // Adds a cross-reference for a node_id in the current block.
  auto AddCrossReference(SemanticsNodeId node_id) -> SemanticsNodeId;

  // Adds a node to the current block, returning the produced ID.
  auto AddNode(SemanticsNode node) -> SemanticsNodeId;

  // Binds a DeclaredName to a target node with the given type.
  auto BindName(ParseTree::Node name_node, SemanticsNodeId type_id,
                SemanticsNodeId target_id) -> void;

  // Pushes a parse tree node onto the stack. Used when there is no IR generated
  // by the node.
  auto Push(ParseTree::Node parse_node) -> void;

  // Pushes a parse tree node onto the stack, storing the SemanticsNode as the
  // result.
  auto Push(ParseTree::Node parse_node, SemanticsNode node) -> void;

  // Pushes a parse tree node onto the stack with an already-built node ID.
  auto Push(ParseTree::Node parse_node, SemanticsNodeId node_id) -> void;

  // Pops the top of the stack, verifying that it's the expected kind.
  auto Pop(ParseNodeKind pop_parse_kind) -> void;

  // Pops the top of the stack, returning the result_id. Must only be called for
  // nodes that have results.
  auto PopWithResult() -> SemanticsNodeId;

  // Pops the top of the stack, verifying that it's the expected kind and
  // returning the result_id. Must only be called for nodes that have results.
  auto PopWithResult(ParseNodeKind pop_parse_kind) -> SemanticsNodeId;

  // Pops the top of the stack, verifying that it's the expected kind and
  // returning the result_id. Must only be called for nodes that have results.
  auto PopWithResultIf(ParseNodeKind pop_parse_kind)
      -> std::optional<SemanticsNodeId>;

  // Attempts a type conversion between arguments of the two arguments with
  // provided types, returning the result type. The result type will be invalid
  // for errors; this handles printing diagnostics.
  auto TryTypeConversion(ParseTree::Node parse_node, SemanticsNodeId lhs_id,
                         SemanticsNodeId rhs_id, bool can_convert_lhs)
      -> SemanticsNodeId;

  // Parse node handlers.
#define CARBON_PARSE_NODE_KIND(Name) \
  auto Handle##Name(ParseTree::Node parse_node)->void;
#include "toolchain/parser/parse_node_kind.def"

  // Tokens for getting data on literals.
  const TokenizedBuffer* tokens_;

  // Handles diagnostics.
  TokenDiagnosticEmitter* emitter_;

  // The file's parse tree.
  const ParseTree* parse_tree_;

  // The SemanticsIR being added to.
  SemanticsIR* semantics_;

  // Whether to print verbose output.
  llvm::raw_ostream* vlog_stream_;

  // The stack during Build. Will contain file-level parse nodes on return.
  llvm::SmallVector<TraversalStackEntry> node_stack_;

  // The stack of node blocks during build. Only updated on ParseTree nodes that
  // affect the stack.
  llvm::SmallVector<SemanticsNodeBlockId> node_block_stack_;

  // Provides name lookup functionality. Each string maps to a stack of
  // SemanticsNodeIDs, where the last node added will correspond to the current
  // block. All SemanticsNodeIDs must be cross-references, because names will
  // frequently be used across blocks.
  llvm::DenseMap<SemanticsStringId, llvm::SmallVector<SemanticsNodeId>,
                 SemanticsStringIdMapInfo>
      name_lookup_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_PARSE_TREE_HANDLER_H_

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_PARSE_TREE_HANDLER_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_PARSE_TREE_HANDLER_H_

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/semantics_ir.h"
#include "toolchain/semantics/semantics_node.h"
#include "toolchain/semantics/semantics_node_stack.h"

namespace Carbon {

// Handles processing of a ParseTree for semantics.
class SemanticsParseTreeHandler {
 public:
  // Stores references for work.
  explicit SemanticsParseTreeHandler(
      const TokenizedBuffer& tokens,
      DiagnosticEmitter<ParseTree::Node>& emitter, const ParseTree& parse_tree,
      SemanticsIR& semantics, llvm::raw_ostream* vlog_stream)
      : tokens_(&tokens),
        emitter_(&emitter),
        parse_tree_(&parse_tree),
        semantics_(&semantics),
        vlog_stream_(vlog_stream),
        node_stack_(parse_tree, vlog_stream) {}

  // Outputs the ParseTree information into SemanticsIR.
  auto Build() -> void;

 private:
  // Prints the node_stack_ on stack dumps.
  class PrettyStackTraceNodeStack;
  // Prints the node_block_stack_ on stack dumps.
  class PrettyStackTraceNodeBlockStack;

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

  // An entry in scope_stack_.
  struct ScopeStackEntry {
    // Names which are registered with name_lookup_, and will need to be
    // deregistered when the scope ends.
    llvm::DenseSet<SemanticsStringId, SemanticsStringIdMapInfo> names;

    // TODO: This likely needs to track things which need to be destructed.
  };

  // Adds a node to the current block, returning the produced ID.
  auto AddNode(SemanticsNode node) -> SemanticsNodeId;

  // Pushes a parse tree node onto the stack, storing the SemanticsNode as the
  // result.
  auto AddNodeAndPush(ParseTree::Node parse_node, SemanticsNode node) -> void;

  // Adds a name to name lookup. This is typically done through BindName, but
  // can also be used to restore removed names.
  auto AddNameToLookup(SemanticsStringId name_id, SemanticsNodeId storage_id)
      -> void {
    name_lookup_[name_id].push_back(storage_id);
  }

  // Binds a DeclaredName to a target node with the given type.
  auto BindName(ParseTree::Node name_node, SemanticsNodeId type_id,
                SemanticsNodeId target_id) -> SemanticsStringId;

  // Pushes a new scope onto scope_stack_.
  auto PushScope() -> void;

  // Pops the top scope from scope_stack_, cleaning up names from name_lookup_.
  auto PopScope() -> void;

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

  auto current_block_id() -> SemanticsNodeBlockId {
    return node_block_stack_.back();
  }

  auto current_scope() -> ScopeStackEntry& { return scope_stack_.back(); }

  // Tokens for getting data on literals.
  const TokenizedBuffer* tokens_;

  // Handles diagnostics.
  DiagnosticEmitter<ParseTree::Node>* emitter_;

  // The file's parse tree.
  const ParseTree* parse_tree_;

  // The SemanticsIR being added to.
  SemanticsIR* semantics_;

  // Whether to print verbose output.
  llvm::raw_ostream* vlog_stream_;

  // The stack during Build. Will contain file-level parse nodes on return.
  SemanticsNodeStack node_stack_;

  // The stack of node blocks during build. Only updated on ParseTree nodes that
  // affect the stack.
  llvm::SmallVector<SemanticsNodeBlockId> node_block_stack_;

  // A stack for scope context.
  llvm::SmallVector<ScopeStackEntry> scope_stack_;

  // Maps identifiers to name lookup results. Values are a stack of name lookup
  // results in the ancestor scopes. This offers constant-time lookup of names,
  // regardless of how many scopes exist between the name declaration and
  // reference.
  //
  // Names which no longer have lookup results are erased.
  llvm::DenseMap<SemanticsStringId, llvm::SmallVector<SemanticsNodeId>,
                 SemanticsStringIdMapInfo>
      name_lookup_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_PARSE_TREE_HANDLER_H_

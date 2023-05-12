// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_CONTEXT_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_CONTEXT_H_

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/semantics_ir.h"
#include "toolchain/semantics/semantics_node.h"
#include "toolchain/semantics/semantics_node_block_stack.h"
#include "toolchain/semantics/semantics_node_stack.h"

namespace Carbon {

// Context and shared functionality for semantics handlers.
class SemanticsContext {
 public:
  // Stores references for work.
  explicit SemanticsContext(const TokenizedBuffer& tokens,
                            DiagnosticEmitter<ParseTree::Node>& emitter,
                            const ParseTree& parse_tree, SemanticsIR& semantics,
                            llvm::raw_ostream* vlog_stream);

  // Marks an implementation TODO. Always returns false.
  auto TODO(ParseTree::Node parse_node, std::string label) -> bool;

  // Runs verification that the processing cleanly finished.
  auto VerifyOnFinish() -> void;

  // Adds a node to the current block, returning the produced ID.
  auto AddNode(SemanticsNode node) -> SemanticsNodeId;

  // Pushes a parse tree node onto the stack, storing the SemanticsNode as the
  // result.
  auto AddNodeAndPush(ParseTree::Node parse_node, SemanticsNode node) -> void;

  // Adds a name to name lookup.
  auto AddNameToLookup(ParseTree::Node name_node, SemanticsStringId name_id,
                       SemanticsNodeId target_id) -> void;

  // Binds a DeclaredName to a target node with the given type.
  auto BindName(ParseTree::Node name_node, SemanticsNodeId type_id,
                SemanticsNodeId target_id) -> SemanticsStringId;

  // Temporarily remove name lookup entries added by the `var`. These will be
  // restored by `VariableDeclaration` using `ReaddNameToLookup`.
  auto TempRemoveLatestNameFromLookup() -> SemanticsNodeId;

  // Re-adds a name to name lookup. This is typically done through BindName, but
  // can also be used to restore removed names.
  auto ReaddNameToLookup(SemanticsStringId name_id, SemanticsNodeId storage_id)
      -> void {
    name_lookup_[name_id].push_back(storage_id);
  }

  // Lookup up a name, returning the referenced node.
  auto LookupName(ParseTree::Node parse_node, llvm::StringRef name)
      -> SemanticsNodeId;

  // Pushes a new scope onto scope_stack_.
  auto PushScope() -> void;

  // Pops the top scope from scope_stack_, cleaning up names from name_lookup_.
  auto PopScope() -> void;

  // Runs ImplicitAsImpl for a set of arguments and parameters.
  //
  // This will eventually need to support checking against multiple possible
  // overloads, multiple of which may be possible but not "best". While this can
  // currently be done by calling twice, toggling `apply_implicit_as`, in the
  // future we may want to remember the right implicit conversions to do for
  // valid cases in order to efficiently handle generics.
  auto ImplicitAsForArgs(
      SemanticsNodeBlockId arg_ir_id, SemanticsNodeBlockId arg_refs_id,
      ParseTree::Node param_parse_node, SemanticsNodeBlockId param_refs_id,
      DiagnosticEmitter<ParseTree::Node>::DiagnosticBuilder* diagnostic)
      -> bool;

  // Runs ImplicitAsImpl for a situation where a cast is required, returning the
  // updated `value_id`. Prints a diagnostic and returns an InvalidType if
  // unsupported.
  auto ImplicitAsRequired(ParseTree::Node parse_node, SemanticsNodeId value_id,
                          SemanticsNodeId as_type_id) -> SemanticsNodeId;

  // Starts handling parameters or arguments.
  auto ParamOrArgStart() -> void;

  // On a comma, pushes the entry. On return, the top of node_stack_ will be
  // start_kind.
  auto ParamOrArgComma(bool for_args) -> void;

  // Detects whether there's an entry to push. On return, the top of
  // node_stack_ will be start_kind, and the caller should do type-specific
  // processing. Returns a pair of {ir_id, refs_id}.
  auto ParamOrArgEnd(bool for_args, ParseNodeKind start_kind)
      -> std::pair<SemanticsNodeBlockId, SemanticsNodeBlockId>;

  // Saves a parameter from the top block in node_stack_ to the top block in
  // params_or_args_stack_. If for_args, adds a StubReference of the previous
  // node's result to the IR.
  //
  // This should only be called by other ParamOrArg functions, not directly.
  auto ParamOrArgSave(bool for_args) -> void;

  // Prints information for a stack dump.
  auto PrintForStackDump(llvm::raw_ostream& output) const -> void;

  auto tokens() -> const TokenizedBuffer& { return *tokens_; }

  auto emitter() -> DiagnosticEmitter<ParseTree::Node>& { return *emitter_; }

  auto parse_tree() -> const ParseTree& { return *parse_tree_; }

  auto semantics() -> SemanticsIR& { return *semantics_; }

  auto node_stack() -> SemanticsNodeStack& { return node_stack_; }

  auto node_block_stack() -> SemanticsNodeBlockStack& {
    return node_block_stack_;
  }

  auto args_type_info_stack() -> SemanticsNodeBlockStack& {
    return args_type_info_stack_;
  }

  auto finished_params_stack() -> llvm::SmallVector<
      std::pair<SemanticsNodeBlockId, SemanticsNodeBlockId>>& {
    return finished_params_stack_;
  }

  auto return_scope_stack() -> llvm::SmallVector<SemanticsNodeId>& {
    return return_scope_stack_;
  }

 private:
  // For CanImplicitAs, the detected conversion to apply.
  enum ImplicitAsKind {
    // Incompatible types.
    Incompatible,
    // No conversion required.
    Identical,
    // ImplicitAs is required.
    Compatible,
  };

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

  // Runs ImplicitAs behavior to convert `value` to `as_type`, returning the
  // result type. The result will be the node to use to replace `value`.
  //
  // If `output_value_id` is null, then this only checks if the conversion is
  // possible.
  //
  // If `output_value_id` is not null, then it will be set if there is a need to
  // cast.
  auto ImplicitAsImpl(SemanticsNodeId value_id, SemanticsNodeId as_type_id,
                      SemanticsNodeId* output_value_id) -> ImplicitAsKind;

  // Returns true if the ImplicitAs can use struct conversion.
  // TODO: This currently only supports struct types that precisely match.
  auto CanImplicitAsStruct(SemanticsNode value_type, SemanticsNode as_type)
      -> bool;

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

  // The stack of node blocks being used for general IR generation.
  SemanticsNodeBlockStack node_block_stack_;

  // The stack of node blocks being used for per-element tracking of nodes in
  // parameter and argument node blocks. Versus node_block_stack_, an element
  // will have 1 or more nodes in blocks in node_block_stack_, but only ever 1
  // node in blocks here.
  SemanticsNodeBlockStack params_or_args_stack_;

  // The stack of node blocks being used for type information while processing
  // arguments. This is used in parallel with params_or_args_stack_. It's
  // currently only used for struct literals, where we need to track names
  // for a type separate from the literal arguments.
  SemanticsNodeBlockStack args_type_info_stack_;

  // Completed parameters that are held temporarily on a side-channel for a
  // function. This can't use node_stack_ because it has space for only one
  // value, whereas parameters return two values.
  llvm::SmallVector<std::pair<SemanticsNodeBlockId, SemanticsNodeBlockId>>
      finished_params_stack_;

  // A stack of return scopes; i.e., targets for `return`. Inside a function,
  // this will be a FunctionDeclaration.
  llvm::SmallVector<SemanticsNodeId> return_scope_stack_;

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

// Parse node handlers. Returns false for unrecoverable errors.
#define CARBON_PARSE_NODE_KIND(Name)                     \
  auto SemanticsHandle##Name(SemanticsContext& context,  \
                             ParseTree::Node parse_node) \
      ->bool;
#include "toolchain/parser/parse_node_kind.def"

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_CONTEXT_H_

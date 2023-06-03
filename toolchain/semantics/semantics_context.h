// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_CONTEXT_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_CONTEXT_H_

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FoldingSet.h"
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

  // Adds a name to name lookup. Prints a diagnostic for name conflicts.
  auto AddNameToLookup(ParseTree::Node name_node, SemanticsStringId name_id,
                       SemanticsNodeId target_id) -> void;

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
      SemanticsNodeBlockId arg_refs_id, ParseTree::Node param_parse_node,
      SemanticsNodeBlockId param_refs_id,
      DiagnosticEmitter<ParseTree::Node>::DiagnosticBuilder* diagnostic)
      -> bool;

  // Runs ImplicitAsImpl for a situation where a cast is required, returning the
  // updated `value_id`. Prints a diagnostic and returns an InvalidType if
  // unsupported.
  auto ImplicitAsRequired(ParseTree::Node parse_node, SemanticsNodeId value_id,
                          SemanticsTypeId as_type_id) -> SemanticsNodeId;

  // Canonicalizes a type which is tracked as a single node.
  // TODO: This should eventually return a type ID.
  auto CanonicalizeType(SemanticsNodeId node_id) -> SemanticsTypeId;

  // Handles canonicalization of struct types. This may create a new struct type
  // when it has a new structure, or reference an existing struct type when it
  // duplicates a prior type.
  //
  // Individual struct type fields aren't canonicalized because they may have
  // name conflicts or other diagnostics during creation, which can use the
  // parse node.
  auto CanonicalizeStructType(ParseTree::Node parse_node,
                              SemanticsNodeBlockId refs_id) -> SemanticsTypeId;

  // Converts an expression for use as a type.
  // TODO: This should eventually return a type ID.
  auto ExpressionAsType(ParseTree::Node parse_node, SemanticsNodeId value_id)
      -> SemanticsTypeId {
    return CanonicalizeType(
        ImplicitAsRequired(parse_node, value_id, SemanticsTypeId::TypeType));
  }

  // Starts handling parameters or arguments.
  auto ParamOrArgStart() -> void;

  // On a comma, pushes the entry. On return, the top of node_stack_ will be
  // start_kind.
  auto ParamOrArgComma(bool for_args) -> void;

  // Detects whether there's an entry to push. On return, the top of
  // node_stack_ will be start_kind, and the caller should do type-specific
  // processing. Returns refs_id.
  auto ParamOrArgEnd(bool for_args, ParseNodeKind start_kind)
      -> SemanticsNodeBlockId;

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

  auto semantics_ir() -> SemanticsIR& { return *semantics_ir_; }

  auto node_stack() -> SemanticsNodeStack& { return node_stack_; }

  auto node_block_stack() -> SemanticsNodeBlockStack& {
    return node_block_stack_;
  }

  auto args_type_info_stack() -> SemanticsNodeBlockStack& {
    return args_type_info_stack_;
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

  // A FoldingSet node for a struct type.
  class StructTypeNode : public llvm::FastFoldingSetNode {
   public:
    explicit StructTypeNode(const llvm::FoldingSetNodeID& node_id,
                            SemanticsTypeId type_id)
        : llvm::FastFoldingSetNode(node_id), type_id_(type_id) {}

    auto type_id() -> SemanticsTypeId { return type_id_; }

   private:
    SemanticsTypeId type_id_;
  };

  // An entry in scope_stack_.
  struct ScopeStackEntry {
    // Names which are registered with name_lookup_, and will need to be
    // deregistered when the scope ends.
    llvm::DenseSet<SemanticsStringId> names;

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
  auto ImplicitAsImpl(SemanticsNodeId value_id, SemanticsTypeId as_type_id,
                      SemanticsNodeId* output_value_id) -> ImplicitAsKind;

  auto current_scope() -> ScopeStackEntry& { return scope_stack_.back(); }

  // Tokens for getting data on literals.
  const TokenizedBuffer* tokens_;

  // Handles diagnostics.
  DiagnosticEmitter<ParseTree::Node>* emitter_;

  // The file's parse tree.
  const ParseTree* parse_tree_;

  // The SemanticsIR being added to.
  SemanticsIR* semantics_ir_;

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
  llvm::DenseMap<SemanticsStringId, llvm::SmallVector<SemanticsNodeId>>
      name_lookup_;

  // Tracks types which have been used, so that they aren't repeatedly added to
  // SemanticsIR.
  llvm::DenseMap<SemanticsNodeId, SemanticsTypeId> canonical_types_;

  // Tracks struct type literals which have been defined, so that they aren't
  // repeatedly redefined.
  llvm::FoldingSet<StructTypeNode> canonical_struct_types_;

  // Storage for the nodes in canonical_struct_types_. This stores in pointers
  // so that canonical_struct_types_ can have stable pointers.
  llvm::SmallVector<std::unique_ptr<StructTypeNode>>
      canonical_struct_types_nodes_;
};

// Parse node handlers. Returns false for unrecoverable errors.
#define CARBON_PARSE_NODE_KIND(Name)                     \
  auto SemanticsHandle##Name(SemanticsContext& context,  \
                             ParseTree::Node parse_node) \
      ->bool;
#include "toolchain/parser/parse_node_kind.def"

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_CONTEXT_H_

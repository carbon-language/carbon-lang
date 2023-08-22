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
#include "toolchain/semantics/semantics_declaration_name_stack.h"
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

  // Adds a node to the given block, returning the produced ID.
  auto AddNodeToBlock(SemanticsNodeBlockId block, SemanticsNode node)
      -> SemanticsNodeId;

  // Pushes a parse tree node onto the stack, storing the SemanticsNode as the
  // result.
  auto AddNodeAndPush(ParseTree::Node parse_node, SemanticsNode node) -> void;

  // Adds a name to name lookup. Prints a diagnostic for name conflicts.
  auto AddNameToLookup(ParseTree::Node name_node, SemanticsStringId name_id,
                       SemanticsNodeId target_id) -> void;

  // Performs name lookup in a specified scope, returning the referenced node.
  // If scope_id is invalid, uses the current contextual scope.
  auto LookupName(ParseTree::Node parse_node, SemanticsStringId name_id,
                  SemanticsNameScopeId scope_id, bool print_diagnostics)
      -> SemanticsNodeId;

  // Prints a diagnostic for a duplicate name.
  auto DiagnoseDuplicateName(ParseTree::Node parse_node,
                             SemanticsNodeId prev_def_id) -> void;

  // Prints a diagnostic for a missing name.
  auto DiagnoseNameNotFound(ParseTree::Node parse_node,
                            SemanticsStringId name_id) -> void;

  // Pushes a new scope onto scope_stack_.
  auto PushScope() -> void;

  // Pops the top scope from scope_stack_, cleaning up names from name_lookup_.
  auto PopScope() -> void;

  // Adds a `Branch` node branching to a new node block, and returns the ID of
  // the new block. All paths to the branch target must go through the current
  // block, though not necessarily through this branch.
  auto AddDominatedBlockAndBranch(ParseTree::Node parse_node)
      -> SemanticsNodeBlockId;

  // Adds a `Branch` node branching to a new node block with a value, and
  // returns the ID of the new block. All paths to the branch target must go
  // through the current block.
  auto AddDominatedBlockAndBranchWithArg(ParseTree::Node parse_node,
                                         SemanticsNodeId arg_id)
      -> SemanticsNodeBlockId;

  // Adds a `BranchIf` node branching to a new node block, and returns the ID
  // of the new block. All paths to the branch target must go through the
  // current block.
  auto AddDominatedBlockAndBranchIf(ParseTree::Node parse_node,
                                    SemanticsNodeId cond_id)
      -> SemanticsNodeBlockId;

  // Adds branches from the given list of blocks to a new block, for
  // reconvergence of control flow, and pushes the new block onto the node
  // block stack.
  auto AddConvergenceBlockAndPush(
      ParseTree::Node parse_node,
      std::initializer_list<SemanticsNodeBlockId> blocks) -> void;

  // Adds branches from the given list of blocks and values to a new block, for
  // reconvergence of control flow with a result value, and pushes the new
  // block onto the node block stack. Returns a node referring to the result
  // value.
  auto AddConvergenceBlockWithArgAndPush(
      ParseTree::Node parse_node,
      std::initializer_list<std::pair<SemanticsNodeBlockId, SemanticsNodeId>>
          blocks_and_args) -> SemanticsNodeId;

  // Add the current code block to the enclosing function.
  auto AddCurrentCodeBlockToFunction() -> void;

  // Returns whether the current position in the current block is reachable.
  auto is_current_position_reachable() -> bool;

  // Converts the given expression to an ephemeral reference to a temporary if
  // it is an initializing expression.
  auto MaterializeIfInitializing(SemanticsNodeId expr_id) -> SemanticsNodeId {
    if (GetSemanticsExpressionCategory(semantics_ir(), expr_id) ==
        SemanticsExpressionCategory::Initializing) {
      return FinalizeTemporary(expr_id, /*discarded=*/false);
    }
    return expr_id;
  }

  // Convert the given expression to a value expression of the same type.
  auto ConvertToValueExpression(SemanticsNodeId expr_id) -> SemanticsNodeId;

  // Performs initialization of `target_id` from `value_id`. Returns the
  // converted initializer value.
  auto Initialize(ParseTree::Node parse_node, SemanticsNodeId target_id,
                  SemanticsNodeId value_id) -> SemanticsNodeId;

  // Converts `value_id` to a value expression of type `type_id`.
  auto ConvertToValueOfType(ParseTree::Node parse_node,
                            SemanticsNodeId value_id, SemanticsTypeId type_id)
      -> SemanticsNodeId {
    return ConvertToValueExpression(
        ImplicitAsRequired(parse_node, value_id, type_id));
  }

  // Converts `value_id` to a value expression of type `bool`.
  auto ConvertToBoolValue(ParseTree::Node parse_node, SemanticsNodeId value_id)
      -> SemanticsNodeId {
    return ConvertToValueOfType(
        parse_node, value_id,
        CanonicalizeType(SemanticsNodeId::BuiltinBoolType));
  }

  // Handles an expression whose result is discarded.
  auto HandleDiscardedExpression(SemanticsNodeId id) -> void;

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
  // updated `value_id`. Prints a diagnostic and returns an Error if
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

  // Handles canonicalization of tuple types. This may create a new tuple type
  // if the `type_ids` doesn't match an existing tuple type.
  auto CanonicalizeTupleType(ParseTree::Node parse_node,
                             llvm::SmallVector<SemanticsTypeId>&& type_ids)
      -> SemanticsTypeId;

  // Returns a pointer type whose pointee type is `pointee_type_id`.
  auto GetPointerType(ParseTree::Node parse_node,
                      SemanticsTypeId pointee_type_id) -> SemanticsTypeId;

  // Converts an expression for use as a type.
  // TODO: This should eventually return a type ID.
  auto ExpressionAsType(ParseTree::Node parse_node, SemanticsNodeId value_id)
      -> SemanticsTypeId {
    auto node = semantics_ir_->GetNode(value_id);
    if (node.kind() == SemanticsNodeKind::StubReference) {
      value_id = node.GetAsStubReference();
      CARBON_CHECK(semantics_ir_->GetNode(value_id).kind() !=
                   SemanticsNodeKind::StubReference)
          << "Stub reference should not point to another stub reference";
    }

    return CanonicalizeType(
        ConvertToValueOfType(parse_node, value_id, SemanticsTypeId::TypeType));
  }

  // Removes any top-level `const` qualifiers from a type.
  auto GetUnqualifiedType(SemanticsTypeId type_id) -> SemanticsTypeId;

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

  auto declaration_name_stack() -> SemanticsDeclarationNameStack& {
    return declaration_name_stack_;
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

  // A FoldingSet node for a type.
  class TypeNode : public llvm::FastFoldingSetNode {
   public:
    explicit TypeNode(const llvm::FoldingSetNodeID& node_id,
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

  // Commits to using a temporary to store the result of the initializing
  // expression described by `init_id`, and returns the location of the
  // temporary. If `discarded` is `true`, the result is discarded, and no
  // temporary will be created if possible; if no temporary is created, the
  // return value will be `SemanticsNodeId::Invalid`.
  auto FinalizeTemporary(SemanticsNodeId init_id, bool discarded)
      -> SemanticsNodeId;

  // Marks the initializer `init_id` as initializing `target_id`.
  auto MarkInitializerFor(SemanticsNodeId target_id, SemanticsNodeId init_id)
      -> void;

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

  // Forms a canonical type ID for a type. This function is given two
  // callbacks:
  //
  // `profile_type(canonical_id)` is called to build a fingerprint for this
  // type. The ID should be distinct for all distinct type values with the same
  // `kind`.
  //
  // `make_node()` is called to obtain a `SemanticsNodeId` that describes the
  // type. It is only called if the type does not already exist, so can be used
  // to lazily build the `SemanticsNode`. `make_node()` is not permitted to
  // directly or indirectly canonicalize any types.
  auto CanonicalizeTypeImpl(
      SemanticsNodeKind kind,
      llvm::function_ref<void(llvm::FoldingSetNodeID& canonical_id)>
          profile_type,
      llvm::function_ref<SemanticsNodeId()> make_node) -> SemanticsTypeId;

  // Forms a canonical type ID for a type. If the type is new, adds the node to
  // the current block.
  auto CanonicalizeTypeAndAddNodeIfNew(SemanticsNode node) -> SemanticsTypeId;

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

  // The stack used for qualified declaration name construction.
  SemanticsDeclarationNameStack declaration_name_stack_;

  // Maps identifiers to name lookup results. Values are a stack of name lookup
  // results in the ancestor scopes. This offers constant-time lookup of names,
  // regardless of how many scopes exist between the name declaration and
  // reference.
  //
  // Names which no longer have lookup results are erased.
  llvm::DenseMap<SemanticsStringId, llvm::SmallVector<SemanticsNodeId>>
      name_lookup_;

  // Cache of the mapping from nodes to types, to avoid recomputing the folding
  // set ID.
  llvm::DenseMap<SemanticsNodeId, SemanticsTypeId> canonical_types_;

  // Tracks the canonical representation of types that have been defined.
  llvm::FoldingSet<TypeNode> canonical_type_nodes_;

  // Storage for the nodes in canonical_type_nodes_. This stores in pointers so
  // that FoldingSet can have stable pointers.
  llvm::SmallVector<std::unique_ptr<TypeNode>> type_node_storage_;
};

// Parse node handlers. Returns false for unrecoverable errors.
#define CARBON_PARSE_NODE_KIND(Name)                     \
  auto SemanticsHandle##Name(SemanticsContext& context,  \
                             ParseTree::Node parse_node) \
      ->bool;
#include "toolchain/parser/parse_node_kind.def"

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_CONTEXT_H_

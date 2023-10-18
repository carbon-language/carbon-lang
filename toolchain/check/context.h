// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CONTEXT_H_
#define CARBON_TOOLCHAIN_CHECK_CONTEXT_H_

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/check/declaration_name_stack.h"
#include "toolchain/check/node_block_stack.h"
#include "toolchain/check/node_stack.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/node.h"

namespace Carbon::Check {

// Context and shared functionality for semantics handlers.
class Context {
 public:
  using DiagnosticEmitter = Carbon::DiagnosticEmitter<Parse::Node>;
  using DiagnosticBuilder = DiagnosticEmitter::DiagnosticBuilder;

  // A scope in which `break` and `continue` can be used.
  struct BreakContinueScope {
    SemIR::NodeBlockId break_target;
    SemIR::NodeBlockId continue_target;
  };

  // Stores references for work.
  explicit Context(const Lex::TokenizedBuffer& tokens,
                   DiagnosticEmitter& emitter, const Parse::Tree& parse_tree,
                   SemIR::File& semantics, llvm::raw_ostream* vlog_stream);

  // Marks an implementation TODO. Always returns false.
  auto TODO(Parse::Node parse_node, std::string label) -> bool;

  // Runs verification that the processing cleanly finished.
  auto VerifyOnFinish() -> void;

  // Adds a node to the current block, returning the produced ID.
  auto AddNode(SemIR::Node node) -> SemIR::NodeId;

  // Pushes a parse tree node onto the stack, storing the SemIR::Node as the
  // result.
  auto AddNodeAndPush(Parse::Node parse_node, SemIR::Node node) -> void;

  // Adds a name to name lookup. Prints a diagnostic for name conflicts.
  auto AddNameToLookup(Parse::Node name_node, SemIR::StringId name_id,
                       SemIR::NodeId target_id) -> void;

  // Performs name lookup in a specified scope, returning the referenced node.
  // If scope_id is invalid, uses the current contextual scope.
  auto LookupName(Parse::Node parse_node, SemIR::StringId name_id,
                  SemIR::NameScopeId scope_id, bool print_diagnostics)
      -> SemIR::NodeId;

  // Prints a diagnostic for a duplicate name.
  auto DiagnoseDuplicateName(Parse::Node parse_node, SemIR::NodeId prev_def_id)
      -> void;

  // Prints a diagnostic for a missing name.
  auto DiagnoseNameNotFound(Parse::Node parse_node, SemIR::StringId name_id)
      -> void;

  // Adds a note to a diagnostic explaining that a class is incomplete.
  auto NoteIncompleteClass(SemIR::ClassDeclaration class_decl,
                           DiagnosticBuilder& builder) -> void;

  // Pushes a new scope onto scope_stack_.
  auto PushScope(SemIR::NameScopeId scope_id = SemIR::NameScopeId::Invalid)
      -> void;

  // Pops the top scope from scope_stack_, cleaning up names from name_lookup_.
  auto PopScope() -> void;

  auto current_scope_id() const -> SemIR::NameScopeId {
    return scope_stack_.back().scope_id;
  }

  // Follows NameReference nodes to find the value named by a given node.
  auto FollowNameReferences(SemIR::NodeId node_id) -> SemIR::NodeId;

  // Adds a `Branch` node branching to a new node block, and returns the ID of
  // the new block. All paths to the branch target must go through the current
  // block, though not necessarily through this branch.
  auto AddDominatedBlockAndBranch(Parse::Node parse_node) -> SemIR::NodeBlockId;

  // Adds a `Branch` node branching to a new node block with a value, and
  // returns the ID of the new block. All paths to the branch target must go
  // through the current block.
  auto AddDominatedBlockAndBranchWithArg(Parse::Node parse_node,
                                         SemIR::NodeId arg_id)
      -> SemIR::NodeBlockId;

  // Adds a `BranchIf` node branching to a new node block, and returns the ID
  // of the new block. All paths to the branch target must go through the
  // current block.
  auto AddDominatedBlockAndBranchIf(Parse::Node parse_node,
                                    SemIR::NodeId cond_id)
      -> SemIR::NodeBlockId;

  // Handles recovergence of control flow. Adds branches from the top
  // `num_blocks` on the node block stack to a new block, pops the existing
  // blocks, and pushes the new block onto the node block stack.
  auto AddConvergenceBlockAndPush(Parse::Node parse_node, int num_blocks)
      -> void;

  // Handles recovergence of control flow with a result value. Adds branches
  // from the top few blocks on the node block stack to a new block, pops the
  // existing blocks, and pushes the new block onto the node block stack. The
  // number of blocks popped is the size of `block_args`, and the corresponding
  // result values are the elements of `block_args`. Returns a node referring
  // to the result value.
  auto AddConvergenceBlockWithArgAndPush(
      Parse::Node parse_node,
      std::initializer_list<SemIR::NodeId> blocks_and_args) -> SemIR::NodeId;

  // Add the current code block to the enclosing function.
  auto AddCurrentCodeBlockToFunction() -> void;

  // Returns whether the current position in the current block is reachable.
  auto is_current_position_reachable() -> bool;

  // Canonicalizes a type which is tracked as a single node.
  auto CanonicalizeType(SemIR::NodeId node_id) -> SemIR::TypeId;

  // Handles canonicalization of struct types. This may create a new struct type
  // when it has a new structure, or reference an existing struct type when it
  // duplicates a prior type.
  //
  // Individual struct type fields aren't canonicalized because they may have
  // name conflicts or other diagnostics during creation, which can use the
  // parse node.
  auto CanonicalizeStructType(Parse::Node parse_node,
                              SemIR::NodeBlockId refs_id) -> SemIR::TypeId;

  // Handles canonicalization of tuple types. This may create a new tuple type
  // if the `type_ids` doesn't match an existing tuple type.
  auto CanonicalizeTupleType(Parse::Node parse_node,
                             llvm::ArrayRef<SemIR::TypeId> type_ids)
      -> SemIR::TypeId;

  // Attempts to complete the type `type_id`. Returns `true` if the type is
  // complete, or `false` if it could not be completed. A complete type has
  // known object and value representations.
  //
  // If the type is not complete, `diagnoser` is invoked to diagnose the issue.
  // The builder it returns will be annotated to describe the reason why the
  // type is not complete.
  auto TryToCompleteType(
      SemIR::TypeId type_id,
      std::optional<llvm::function_ref<auto()->DiagnosticBuilder>> diagnoser =
          std::nullopt) -> bool;

  // Gets a builtin type. The returned type will be complete.
  auto GetBuiltinType(SemIR::BuiltinKind kind) -> SemIR::TypeId;

  // Returns a pointer type whose pointee type is `pointee_type_id`.
  auto GetPointerType(Parse::Node parse_node, SemIR::TypeId pointee_type_id)
      -> SemIR::TypeId;

  // Removes any top-level `const` qualifiers from a type.
  auto GetUnqualifiedType(SemIR::TypeId type_id) -> SemIR::TypeId;

  // Starts handling parameters or arguments.
  auto ParamOrArgStart() -> void;

  // On a comma, pushes the entry. On return, the top of node_stack_ will be
  // start_kind.
  auto ParamOrArgComma() -> void;

  // Detects whether there's an entry to push from the end of a parameter or
  // argument list, and if so, moves it to the current parameter or argument
  // list. Does not pop the list. `start_kind` is the node kind at the start
  // of the parameter or argument list, and will be at the top of the parse node
  // stack when this function returns.
  auto ParamOrArgEndNoPop(Parse::NodeKind start_kind) -> void;

  // Pops the current parameter or argument list. Should only be called after
  // `ParamOrArgEndNoPop`.
  auto ParamOrArgPop() -> SemIR::NodeBlockId;

  // Detects whether there's an entry to push. Pops and returns the argument
  // list. This is the same as `ParamOrArgEndNoPop` followed by `ParamOrArgPop`.
  auto ParamOrArgEnd(Parse::NodeKind start_kind) -> SemIR::NodeBlockId;

  // Saves a parameter from the top block in node_stack_ to the top block in
  // params_or_args_stack_.
  auto ParamOrArgSave(SemIR::NodeId node_id) -> void {
    params_or_args_stack_.AddNodeId(node_id);
  }

  // Prints information for a stack dump.
  auto PrintForStackDump(llvm::raw_ostream& output) const -> void;

  auto tokens() -> const Lex::TokenizedBuffer& { return *tokens_; }

  auto emitter() -> DiagnosticEmitter& { return *emitter_; }

  auto parse_tree() -> const Parse::Tree& { return *parse_tree_; }

  auto semantics_ir() -> SemIR::File& { return *semantics_ir_; }

  auto node_stack() -> NodeStack& { return node_stack_; }

  auto node_block_stack() -> NodeBlockStack& { return node_block_stack_; }

  auto args_type_info_stack() -> NodeBlockStack& {
    return args_type_info_stack_;
  }

  auto return_scope_stack() -> llvm::SmallVector<SemIR::NodeId>& {
    return return_scope_stack_;
  }

  auto break_continue_stack() -> llvm::SmallVector<BreakContinueScope>& {
    return break_continue_stack_;
  }

  auto declaration_name_stack() -> DeclarationNameStack& {
    return declaration_name_stack_;
  }

 private:
  // A FoldingSet node for a type.
  class TypeNode : public llvm::FastFoldingSetNode {
   public:
    explicit TypeNode(const llvm::FoldingSetNodeID& node_id,
                      SemIR::TypeId type_id)
        : llvm::FastFoldingSetNode(node_id), type_id_(type_id) {}

    auto type_id() -> SemIR::TypeId { return type_id_; }

   private:
    SemIR::TypeId type_id_;
  };

  // An entry in scope_stack_.
  struct ScopeStackEntry {
    // The name scope associated with this entry, if any.
    SemIR::NameScopeId scope_id;

    // Names which are registered with name_lookup_, and will need to be
    // deregistered when the scope ends.
    llvm::DenseSet<SemIR::StringId> names;

    // TODO: This likely needs to track things which need to be destructed.
  };

  // Forms a canonical type ID for a type. This function is given two
  // callbacks:
  //
  // `profile_type(canonical_id)` is called to build a fingerprint for this
  // type. The ID should be distinct for all distinct type values with the same
  // `kind`.
  //
  // `make_node()` is called to obtain a `SemIR::NodeId` that describes the
  // type. It is only called if the type does not already exist, so can be used
  // to lazily build the `SemIR::Node`. `make_node()` is not permitted to
  // directly or indirectly canonicalize any types.
  auto CanonicalizeTypeImpl(
      SemIR::NodeKind kind,
      llvm::function_ref<void(llvm::FoldingSetNodeID& canonical_id)>
          profile_type,
      llvm::function_ref<SemIR::NodeId()> make_node) -> SemIR::TypeId;

  // Forms a canonical type ID for a type. If the type is new, adds the node to
  // the current block.
  auto CanonicalizeTypeAndAddNodeIfNew(SemIR::Node node) -> SemIR::TypeId;

  auto current_scope() -> ScopeStackEntry& { return scope_stack_.back(); }

  // Tokens for getting data on literals.
  const Lex::TokenizedBuffer* tokens_;

  // Handles diagnostics.
  DiagnosticEmitter* emitter_;

  // The file's parse tree.
  const Parse::Tree* parse_tree_;

  // The SemIR::File being added to.
  SemIR::File* semantics_ir_;

  // Whether to print verbose output.
  llvm::raw_ostream* vlog_stream_;

  // The stack during Build. Will contain file-level parse nodes on return.
  NodeStack node_stack_;

  // The stack of node blocks being used for general IR generation.
  NodeBlockStack node_block_stack_;

  // The stack of node blocks being used for per-element tracking of nodes in
  // parameter and argument node blocks. Versus node_block_stack_, an element
  // will have 1 or more nodes in blocks in node_block_stack_, but only ever 1
  // node in blocks here.
  NodeBlockStack params_or_args_stack_;

  // The stack of node blocks being used for type information while processing
  // arguments. This is used in parallel with params_or_args_stack_. It's
  // currently only used for struct literals, where we need to track names
  // for a type separate from the literal arguments.
  NodeBlockStack args_type_info_stack_;

  // A stack of return scopes; i.e., targets for `return`. Inside a function,
  // this will be a FunctionDeclaration.
  llvm::SmallVector<SemIR::NodeId> return_scope_stack_;

  // A stack of `break` and `continue` targets.
  llvm::SmallVector<BreakContinueScope> break_continue_stack_;

  // A stack for scope context.
  llvm::SmallVector<ScopeStackEntry> scope_stack_;

  // The stack used for qualified declaration name construction.
  DeclarationNameStack declaration_name_stack_;

  // Maps identifiers to name lookup results. Values are a stack of name lookup
  // results in the ancestor scopes. This offers constant-time lookup of names,
  // regardless of how many scopes exist between the name declaration and
  // reference.
  //
  // Names which no longer have lookup results are erased.
  llvm::DenseMap<SemIR::StringId, llvm::SmallVector<SemIR::NodeId>>
      name_lookup_;

  // Cache of the mapping from nodes to types, to avoid recomputing the folding
  // set ID.
  llvm::DenseMap<SemIR::NodeId, SemIR::TypeId> canonical_types_;

  // Tracks the canonical representation of types that have been defined.
  llvm::FoldingSet<TypeNode> canonical_type_nodes_;

  // Storage for the nodes in canonical_type_nodes_. This stores in pointers so
  // that FoldingSet can have stable pointers.
  llvm::SmallVector<std::unique_ptr<TypeNode>> type_node_storage_;
};

// Parse node handlers. Returns false for unrecoverable errors.
#define CARBON_PARSE_NODE_KIND(Name) \
  auto Handle##Name(Context& context, Parse::Node parse_node)->bool;
#include "toolchain/parse/node_kind.def"

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CONTEXT_H_

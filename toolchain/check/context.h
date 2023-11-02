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
#include "toolchain/check/inst_block_stack.h"
#include "toolchain/check/node_stack.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

// Context and shared functionality for semantics handlers.
class Context {
 public:
  using DiagnosticEmitter = Carbon::DiagnosticEmitter<Parse::Node>;
  using DiagnosticBuilder = DiagnosticEmitter::DiagnosticBuilder;

  // A scope in which `break` and `continue` can be used.
  struct BreakContinueScope {
    SemIR::InstBlockId break_target;
    SemIR::InstBlockId continue_target;
  };

  // Stores references for work.
  explicit Context(const Lex::TokenizedBuffer& tokens,
                   DiagnosticEmitter& emitter, const Parse::Tree& parse_tree,
                   SemIR::File& semantics, llvm::raw_ostream* vlog_stream);

  // Marks an implementation TODO. Always returns false.
  auto TODO(Parse::Node parse_node, std::string label) -> bool;

  // Runs verification that the processing cleanly finished.
  auto VerifyOnFinish() -> void;

  // Adds an instruction to the current block, returning the produced ID.
  auto AddInst(SemIR::Inst inst) -> SemIR::InstId;

  // Pushes a parse tree node onto the stack, storing the SemIR::Inst as the
  // result.
  auto AddInstAndPush(Parse::Node parse_node, SemIR::Inst inst) -> void;

  // Adds a name to name lookup. Prints a diagnostic for name conflicts.
  auto AddNameToLookup(Parse::Node name_node, IdentifierId name_id,
                       SemIR::InstId target_id) -> void;

  // Performs name lookup in a specified scope, returning the referenced
  // instruction. If scope_id is invalid, uses the current contextual scope.
  auto LookupName(Parse::Node parse_node, IdentifierId name_id,
                  SemIR::NameScopeId scope_id, bool print_diagnostics)
      -> SemIR::InstId;

  // Prints a diagnostic for a duplicate name.
  auto DiagnoseDuplicateName(Parse::Node parse_node, SemIR::InstId prev_def_id)
      -> void;

  // Prints a diagnostic for a missing name.
  auto DiagnoseNameNotFound(Parse::Node parse_node, IdentifierId name_id)
      -> void;

  // Adds a note to a diagnostic explaining that a class is incomplete.
  auto NoteIncompleteClass(SemIR::ClassId class_id, DiagnosticBuilder& builder)
      -> void;

  // Pushes a new scope onto scope_stack_.
  auto PushScope(SemIR::InstId scope_inst_id = SemIR::InstId::Invalid,
                 SemIR::NameScopeId scope_id = SemIR::NameScopeId::Invalid)
      -> void;

  // Pops the top scope from scope_stack_, cleaning up names from name_lookup_.
  auto PopScope() -> void;

  // Returns the name scope associated with the current lexical scope, if any.
  auto current_scope_id() const -> SemIR::NameScopeId {
    return current_scope().scope_id;
  }

  // Returns the current scope, if it is of the specified kind. Otherwise,
  // returns nullopt.
  template <typename InstT>
  auto GetCurrentScopeAs() -> std::optional<InstT> {
    auto current_scope_inst_id = current_scope().scope_inst_id;
    if (!current_scope_inst_id.is_valid()) {
      return std::nullopt;
    }
    return insts().Get(current_scope_inst_id).TryAs<InstT>();
  }

  // Follows NameReference instructions to find the value named by a given
  // instruction.
  auto FollowNameReferences(SemIR::InstId inst_id) -> SemIR::InstId;

  // Gets the constant value of the given instruction, if it has one.
  auto GetConstantValue(SemIR::InstId inst_id) -> SemIR::InstId;

  // Adds a `Branch` instruction branching to a new instruction block, and
  // returns the ID of the new block. All paths to the branch target must go
  // through the current block, though not necessarily through this branch.
  auto AddDominatedBlockAndBranch(Parse::Node parse_node) -> SemIR::InstBlockId;

  // Adds a `Branch` instruction branching to a new instruction block with a
  // value, and returns the ID of the new block. All paths to the branch target
  // must go through the current block.
  auto AddDominatedBlockAndBranchWithArg(Parse::Node parse_node,
                                         SemIR::InstId arg_id)
      -> SemIR::InstBlockId;

  // Adds a `BranchIf` instruction branching to a new instruction block, and
  // returns the ID of the new block. All paths to the branch target must go
  // through the current block.
  auto AddDominatedBlockAndBranchIf(Parse::Node parse_node,
                                    SemIR::InstId cond_id)
      -> SemIR::InstBlockId;

  // Handles recovergence of control flow. Adds branches from the top
  // `num_blocks` on the instruction block stack to a new block, pops the
  // existing blocks, and pushes the new block onto the instruction block stack.
  auto AddConvergenceBlockAndPush(Parse::Node parse_node, int num_blocks)
      -> void;

  // Handles recovergence of control flow with a result value. Adds branches
  // from the top few blocks on the instruction block stack to a new block, pops
  // the existing blocks, and pushes the new block onto the instruction block
  // stack. The number of blocks popped is the size of `block_args`, and the
  // corresponding result values are the elements of `block_args`. Returns an
  // instruction referring to the result value.
  auto AddConvergenceBlockWithArgAndPush(
      Parse::Node parse_node,
      std::initializer_list<SemIR::InstId> blocks_and_args) -> SemIR::InstId;

  // Add the current code block to the enclosing function.
  auto AddCurrentCodeBlockToFunction() -> void;

  // Returns whether the current position in the current block is reachable.
  auto is_current_position_reachable() -> bool;

  // Canonicalizes a type which is tracked as a single instruction.
  auto CanonicalizeType(SemIR::InstId inst_id) -> SemIR::TypeId;

  // Handles canonicalization of struct types. This may create a new struct type
  // when it has a new structure, or reference an existing struct type when it
  // duplicates a prior type.
  //
  // Individual struct type fields aren't canonicalized because they may have
  // name conflicts or other diagnostics during creation, which can use the
  // parse node.
  auto CanonicalizeStructType(Parse::Node parse_node,
                              SemIR::InstBlockId refs_id) -> SemIR::TypeId;

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
  auto ParamOrArgPop() -> SemIR::InstBlockId;

  // Detects whether there's an entry to push. Pops and returns the argument
  // list. This is the same as `ParamOrArgEndNoPop` followed by `ParamOrArgPop`.
  auto ParamOrArgEnd(Parse::NodeKind start_kind) -> SemIR::InstBlockId;

  // Saves a parameter from the top block in node_stack_ to the top block in
  // params_or_args_stack_.
  auto ParamOrArgSave(SemIR::InstId inst_id) -> void {
    params_or_args_stack_.AddInstId(inst_id);
  }

  // Prints information for a stack dump.
  auto PrintForStackDump(llvm::raw_ostream& output) const -> void;

  auto tokens() -> const Lex::TokenizedBuffer& { return *tokens_; }

  auto emitter() -> DiagnosticEmitter& { return *emitter_; }

  auto parse_tree() -> const Parse::Tree& { return *parse_tree_; }

  auto sem_ir() -> SemIR::File& { return *sem_ir_; }

  auto node_stack() -> NodeStack& { return node_stack_; }

  auto inst_block_stack() -> InstBlockStack& { return inst_block_stack_; }

  auto params_or_args_stack() -> InstBlockStack& {
    return params_or_args_stack_;
  }

  auto args_type_info_stack() -> InstBlockStack& {
    return args_type_info_stack_;
  }

  auto return_scope_stack() -> llvm::SmallVector<SemIR::InstId>& {
    return return_scope_stack_;
  }

  auto break_continue_stack() -> llvm::SmallVector<BreakContinueScope>& {
    return break_continue_stack_;
  }

  auto declaration_name_stack() -> DeclarationNameStack& {
    return declaration_name_stack_;
  }

  // Directly expose SemIR::File data accessors for brevity in calls.
  auto identifiers() -> StringStoreWrapper<IdentifierId>& {
    return sem_ir().identifiers();
  }
  auto integers() -> ValueStore<IntegerId>& { return sem_ir().integers(); }
  auto reals() -> ValueStore<RealId>& { return sem_ir().reals(); }
  auto string_literals() -> StringStoreWrapper<StringLiteralId>& {
    return sem_ir().string_literals();
  }
  auto functions() -> ValueStore<SemIR::FunctionId, SemIR::Function>& {
    return sem_ir().functions();
  }
  auto classes() -> ValueStore<SemIR::ClassId, SemIR::Class>& {
    return sem_ir().classes();
  }
  auto name_scopes() -> SemIR::NameScopeStore& {
    return sem_ir().name_scopes();
  }
  auto types() -> ValueStore<SemIR::TypeId, SemIR::TypeInfo>& {
    return sem_ir().types();
  }
  auto type_blocks()
      -> SemIR::BlockValueStore<SemIR::TypeBlockId, SemIR::TypeId>& {
    return sem_ir().type_blocks();
  }
  auto insts() -> SemIR::InstStore& { return sem_ir().insts(); }
  auto inst_blocks() -> SemIR::InstBlockStore& {
    return sem_ir().inst_blocks();
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
    // The instruction associated with this entry, if any. This can be one of:
    //
    // - A `ClassDeclaration`, for a class definition scope.
    // - A `FunctionDeclaration`, for the outermost scope in a function
    //   definition.
    // - Invalid, for any other scope.
    SemIR::InstId scope_inst_id;

    // The name scope associated with this entry, if any.
    SemIR::NameScopeId scope_id;

    // Names which are registered with name_lookup_, and will need to be
    // deregistered when the scope ends.
    llvm::DenseSet<IdentifierId> names;

    // TODO: This likely needs to track things which need to be destructed.
  };

  // Forms a canonical type ID for a type. This function is given two
  // callbacks:
  //
  // `profile_type(canonical_id)` is called to build a fingerprint for this
  // type. The ID should be distinct for all distinct type values with the same
  // `kind`.
  //
  // `make_inst()` is called to obtain a `SemIR::InstId` that describes the
  // type. It is only called if the type does not already exist, so can be used
  // to lazily build the `SemIR::Inst`. `make_inst()` is not permitted to
  // directly or indirectly canonicalize any types.
  auto CanonicalizeTypeImpl(
      SemIR::InstKind kind,
      llvm::function_ref<void(llvm::FoldingSetNodeID& canonical_id)>
          profile_type,
      llvm::function_ref<SemIR::InstId()> make_inst) -> SemIR::TypeId;

  // Forms a canonical type ID for a type. If the type is new, adds the
  // instruction to the current block.
  auto CanonicalizeTypeAndAddInstIfNew(SemIR::Inst inst) -> SemIR::TypeId;

  auto current_scope() -> ScopeStackEntry& { return scope_stack_.back(); }
  auto current_scope() const -> const ScopeStackEntry& {
    return scope_stack_.back();
  }

  // Tokens for getting data on literals.
  const Lex::TokenizedBuffer* tokens_;

  // Handles diagnostics.
  DiagnosticEmitter* emitter_;

  // The file's parse tree.
  const Parse::Tree* parse_tree_;

  // The SemIR::File being added to.
  SemIR::File* sem_ir_;

  // Whether to print verbose output.
  llvm::raw_ostream* vlog_stream_;

  // The stack during Build. Will contain file-level parse nodes on return.
  NodeStack node_stack_;

  // The stack of instruction blocks being used for general IR generation.
  InstBlockStack inst_block_stack_;

  // The stack of instruction blocks being used for per-element tracking of
  // instructions in parameter and argument instruction blocks. Versus
  // inst_block_stack_, an element will have 1 or more instructions in blocks in
  // inst_block_stack_, but only ever 1 instruction in blocks here.
  InstBlockStack params_or_args_stack_;

  // The stack of instruction blocks being used for type information while
  // processing arguments. This is used in parallel with params_or_args_stack_.
  // It's currently only used for struct literals, where we need to track names
  // for a type separate from the literal arguments.
  InstBlockStack args_type_info_stack_;

  // A stack of return scopes; i.e., targets for `return`. Inside a function,
  // this will be a FunctionDeclaration.
  llvm::SmallVector<SemIR::InstId> return_scope_stack_;

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
  llvm::DenseMap<IdentifierId, llvm::SmallVector<SemIR::InstId>> name_lookup_;

  // Cache of the mapping from instructions to types, to avoid recomputing the
  // folding set ID.
  llvm::DenseMap<SemIR::InstId, SemIR::TypeId> canonical_types_;

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

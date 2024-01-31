// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CONTEXT_H_
#define CARBON_TOOLCHAIN_CHECK_CONTEXT_H_

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/decl_state.h"
#include "toolchain/check/inst_block_stack.h"
#include "toolchain/check/lexical_lookup.h"
#include "toolchain/check/node_stack.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/parse/tree.h"
#include "toolchain/parse/tree_node_location_translator.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/value_stores.h"

namespace Carbon::Check {

// Diagnostic locations produced by checking may be either a parse node
// directly, or an inst ID which is later translated to a parse node.
struct SemIRLocation {
  // NOLINTNEXTLINE(google-explicit-constructor)
  SemIRLocation(SemIR::InstId inst_id) : inst_id(inst_id), is_inst_id(true) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  SemIRLocation(Parse::NodeLocation node_location)
      : node_location(node_location), is_inst_id(false) {}
  // NOLINTNEXTLINE(google-explicit-constructor)
  SemIRLocation(Parse::NodeId node_id)
      : SemIRLocation(Parse::NodeLocation(node_id)) {}

  union {
    SemIR::InstId inst_id;
    Parse::NodeLocation node_location;
  };

  bool is_inst_id;
};

// Context and shared functionality for semantics handlers.
class Context {
 public:
  using DiagnosticEmitter = Carbon::DiagnosticEmitter<SemIRLocation>;
  using DiagnosticBuilder = DiagnosticEmitter::DiagnosticBuilder;

  // A scope in which `break` and `continue` can be used.
  struct BreakContinueScope {
    SemIR::InstBlockId break_target;
    SemIR::InstBlockId continue_target;
  };

  // A scope in which `return` can be used.
  struct ReturnScope {
    // The declaration from which we can return. Inside a function, this will
    // be a `FunctionDecl`.
    SemIR::InstId decl_id;

    // The value corresponding to the current `returned var`, if any. Will be
    // set and unset as `returned var`s are declared and go out of scope.
    SemIR::InstId returned_var = SemIR::InstId::Invalid;
  };

  // Stores references for work.
  explicit Context(const Lex::TokenizedBuffer& tokens,
                   DiagnosticEmitter& emitter, const Parse::Tree& parse_tree,
                   SemIR::File& sem_ir, llvm::raw_ostream* vlog_stream);

  // Marks an implementation TODO. Always returns false.
  auto TODO(Parse::NodeId parse_node, std::string label) -> bool;

  // Runs verification that the processing cleanly finished.
  auto VerifyOnFinish() -> void;

  // Adds an instruction to the current block, returning the produced ID.
  auto AddInst(SemIR::ParseNodeAndInst parse_node_and_inst) -> SemIR::InstId;

  // Adds an instruction in no block, returning the produced ID. Should be used
  // rarely.
  auto AddInstInNoBlock(SemIR::ParseNodeAndInst parse_node_and_inst)
      -> SemIR::InstId;

  // Adds an instruction to the current block, returning the produced ID. The
  // instruction is a placeholder that is expected to be replaced by
  // `ReplaceInstBeforeConstantUse`.
  auto AddPlaceholderInst(SemIR::ParseNodeAndInst parse_node_and_inst)
      -> SemIR::InstId;

  // Adds an instruction in no block, returning the produced ID. Should be used
  // rarely. The instruction is a placeholder that is expected to be replaced by
  // `ReplaceInstBeforeConstantUse`.
  auto AddPlaceholderInstInNoBlock(SemIR::ParseNodeAndInst parse_node_and_inst)
      -> SemIR::InstId;

  // Adds an instruction to the constants block, returning the produced ID.
  auto AddConstant(SemIR::Inst inst, bool is_symbolic) -> SemIR::ConstantId;

  // Pushes a parse tree node onto the stack, storing the SemIR::Inst as the
  // result.
  auto AddInstAndPush(SemIR::ParseNodeAndInst parse_node_and_inst) -> void;

  // Replaces the value of the instruction `inst_id` with `parse_node_and_inst`.
  // The instruction is required to not have been used in any constant
  // evaluation, either because it's newly created and entirely unused, or
  // because it's only used in a position that constant evaluation ignores, such
  // as a return slot.
  auto ReplaceInstBeforeConstantUse(SemIR::InstId inst_id,
                                    SemIR::ParseNodeAndInst parse_node_and_inst)
      -> void;

  // Sets only the parse node of an instruction. This is only used when setting
  // the parse node of an imported namespace. Versus
  // ReplaceInstBeforeConstantUse, it is safe to use after the namespace is used
  // in constant evaluation. It's exposed this way mainly so that `insts()` can
  // remain const.
  auto SetNamespaceParseNode(SemIR::InstId inst_id, Parse::NodeId parse_node)
      -> void {
    sem_ir().insts().SetParseNode(inst_id, parse_node);
  }

  // Adds a package's imports to name lookup, with all libraries together.
  // sem_irs will all be non-null; has_load_error must be used for any errors.
  auto AddPackageImports(Parse::NodeId import_node, IdentifierId package_id,
                         llvm::ArrayRef<const SemIR::File*> sem_irs,
                         bool has_load_error) -> void;

  // Adds a name to name lookup. Prints a diagnostic for name conflicts.
  auto AddNameToLookup(SemIR::NameId name_id, SemIR::InstId target_id) -> void;

  // Performs name lookup in a specified scope for a name appearing in a
  // declaration, returning the referenced instruction. If scope_id is invalid,
  // uses the current contextual scope.
  auto LookupNameInDecl(Parse::NodeId parse_node, SemIR::NameId name_id,
                        SemIR::NameScopeId scope_id) -> SemIR::InstId;

  // Performs an unqualified name lookup, returning the referenced instruction.
  auto LookupUnqualifiedName(Parse::NodeId parse_node, SemIR::NameId name_id)
      -> SemIR::InstId;

  // Performs a name lookup in a specified scope, returning the referenced
  // instruction. Does not look into extended scopes. Returns an invalid
  // instruction if the name is not found.
  auto LookupNameInExactScope(SemIR::NameId name_id,
                              const SemIR::NameScope& scope) -> SemIR::InstId;

  // Performs a qualified name lookup in a specified scope and in scopes that
  // it extends, returning the referenced instruction.
  auto LookupQualifiedName(Parse::NodeId parse_node, SemIR::NameId name_id,
                           SemIR::NameScopeId scope_id, bool required = true)
      -> SemIR::InstId;

  // Prints a diagnostic for a duplicate name.
  auto DiagnoseDuplicateName(SemIR::InstId dup_def_id,
                             SemIR::InstId prev_def_id) -> void;

  // Prints a diagnostic for a missing name.
  auto DiagnoseNameNotFound(Parse::NodeId parse_node, SemIR::NameId name_id)
      -> void;

  // Adds a note to a diagnostic explaining that a class is incomplete.
  auto NoteIncompleteClass(SemIR::ClassId class_id, DiagnosticBuilder& builder)
      -> void;

  // Pushes a scope onto scope_stack_. NameScopeId::Invalid is used for new
  // scopes. lexical_lookup_has_load_error is used to limit diagnostics when a
  // given namespace may contain a mix of both successful and failed name
  // imports.
  auto PushScope(SemIR::InstId scope_inst_id = SemIR::InstId::Invalid,
                 SemIR::NameScopeId scope_id = SemIR::NameScopeId::Invalid,
                 bool lexical_lookup_has_load_error = false) -> void;

  // Pops the top scope from scope_stack_, cleaning up names from
  // lexical_lookup_.
  auto PopScope() -> void;

  // Pops the top scope from scope_stack_ if it contains no names.
  auto PopScopeIfEmpty() -> void {
    if (scope_stack_.back().names.empty()) {
      PopScope();
    }
  }

  // Pops scopes until we return to the specified scope index.
  auto PopToScope(ScopeIndex index) -> void;

  // Returns the scope index associated with the current scope.
  auto current_scope_index() const -> ScopeIndex {
    return current_scope().index;
  }

  // Returns the name scope associated with the current lexical scope, if any.
  auto current_scope_id() const -> SemIR::NameScopeId {
    return current_scope().scope_id;
  }

  // Returns the instruction associated with the current scope, or Invalid if
  // there is no such instruction, such as for a block scope.
  auto current_scope_inst_id() const -> SemIR::InstId {
    return current_scope().scope_inst_id;
  }

  auto GetCurrentScopeParseNode() const -> Parse::NodeId {
    auto inst_id = current_scope_inst_id();
    if (!inst_id.is_valid()) {
      return Parse::NodeId::Invalid;
    }
    return sem_ir_->insts().GetParseNode(inst_id);
  }

  // Returns the current scope, if it is of the specified kind. Otherwise,
  // returns nullopt.
  template <typename InstT>
  auto GetCurrentScopeAs() -> std::optional<InstT> {
    auto inst_id = current_scope_inst_id();
    if (!inst_id.is_valid()) {
      return std::nullopt;
    }
    return insts().TryGetAs<InstT>(inst_id);
  }

  // If there is no `returned var` in scope, sets the given instruction to be
  // the current `returned var` and returns an invalid instruction ID. If there
  // is already a `returned var`, returns it instead.
  auto SetReturnedVarOrGetExisting(SemIR::InstId inst_id) -> SemIR::InstId;

  // Adds a `Branch` instruction branching to a new instruction block, and
  // returns the ID of the new block. All paths to the branch target must go
  // through the current block, though not necessarily through this branch.
  auto AddDominatedBlockAndBranch(Parse::NodeId parse_node)
      -> SemIR::InstBlockId;

  // Adds a `Branch` instruction branching to a new instruction block with a
  // value, and returns the ID of the new block. All paths to the branch target
  // must go through the current block.
  auto AddDominatedBlockAndBranchWithArg(Parse::NodeId parse_node,
                                         SemIR::InstId arg_id)
      -> SemIR::InstBlockId;

  // Adds a `BranchIf` instruction branching to a new instruction block, and
  // returns the ID of the new block. All paths to the branch target must go
  // through the current block.
  auto AddDominatedBlockAndBranchIf(Parse::NodeId parse_node,
                                    SemIR::InstId cond_id)
      -> SemIR::InstBlockId;

  // Handles recovergence of control flow. Adds branches from the top
  // `num_blocks` on the instruction block stack to a new block, pops the
  // existing blocks, and pushes the new block onto the instruction block stack.
  auto AddConvergenceBlockAndPush(Parse::NodeId parse_node, int num_blocks)
      -> void;

  // Handles recovergence of control flow with a result value. Adds branches
  // from the top few blocks on the instruction block stack to a new block, pops
  // the existing blocks, and pushes the new block onto the instruction block
  // stack. The number of blocks popped is the size of `block_args`, and the
  // corresponding result values are the elements of `block_args`. Returns an
  // instruction referring to the result value.
  auto AddConvergenceBlockWithArgAndPush(
      Parse::NodeId parse_node, std::initializer_list<SemIR::InstId> block_args)
      -> SemIR::InstId;

  // Add the current code block to the enclosing function.
  // TODO: The parse_node is taken for expressions, which can occur in
  // non-function contexts. This should be refactored to support non-function
  // contexts, and parse_node removed.
  auto AddCurrentCodeBlockToFunction(
      Parse::NodeId parse_node = Parse::NodeId::Invalid) -> void;

  // Returns whether the current position in the current block is reachable.
  auto is_current_position_reachable() -> bool;

  // Returns the type ID for a constant of type `type`.
  auto GetTypeIdForTypeConstant(SemIR::ConstantId constant_id) -> SemIR::TypeId;

  // Attempts to complete the type `type_id`. Returns `true` if the type is
  // complete, or `false` if it could not be completed. A complete type has
  // known object and value representations.
  //
  // If the type is not complete, `diagnoser` is invoked to diagnose the issue,
  // if a `diagnoser` is provided. The builder it returns will be annotated to
  // describe the reason why the type is not complete.
  auto TryToCompleteType(
      SemIR::TypeId type_id,
      std::optional<llvm::function_ref<auto()->DiagnosticBuilder>> diagnoser =
          std::nullopt) -> bool;

  // Returns the type `type_id` as a complete type, or produces an incomplete
  // type error and returns an error type. This is a convenience wrapper around
  // TryToCompleteType.
  auto AsCompleteType(SemIR::TypeId type_id,
                      llvm::function_ref<auto()->DiagnosticBuilder> diagnoser)
      -> SemIR::TypeId {
    return TryToCompleteType(type_id, diagnoser) ? type_id
                                                 : SemIR::TypeId::Error;
  }

  // TODO: Consider moving these `Get*Type` functions to a separate class.

  // Gets a builtin type. The returned type will be complete.
  auto GetBuiltinType(SemIR::BuiltinKind kind) -> SemIR::TypeId;

  // Returns a class type for the class described by `class_id`.
  // TODO: Support generic arguments.
  auto GetClassType(SemIR::ClassId class_id) -> SemIR::TypeId;

  // Returns a pointer type whose pointee type is `pointee_type_id`.
  auto GetPointerType(SemIR::TypeId pointee_type_id) -> SemIR::TypeId;

  // Returns a struct type with the given fields, which should be a block of
  // `StructTypeField`s.
  auto GetStructType(SemIR::InstBlockId refs_id) -> SemIR::TypeId;

  // Returns a tuple type with the given element types.
  auto GetTupleType(llvm::ArrayRef<SemIR::TypeId> type_ids) -> SemIR::TypeId;

  // Returns an unbound element type.
  auto GetUnboundElementType(SemIR::TypeId class_type_id,
                             SemIR::TypeId element_type_id) -> SemIR::TypeId;

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

  // Adds an exported name.
  auto AddExport(SemIR::InstId inst_id) -> void { exports_.push_back(inst_id); }

  // Finalizes the list of exports on the IR.
  auto FinalizeExports() -> void {
    inst_blocks().Set(SemIR::InstBlockId::Exports, exports_);
  }

  // Prints information for a stack dump.
  auto PrintForStackDump(llvm::raw_ostream& output) const -> void;

  // Get the Lex::TokenKind of a node for diagnostics.
  auto token_kind(Parse::NodeId parse_node) -> Lex::TokenKind {
    return tokens().GetKind(parse_tree().node_token(parse_node));
  }

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

  auto return_scope_stack() -> llvm::SmallVector<ReturnScope>& {
    return return_scope_stack_;
  }

  auto break_continue_stack() -> llvm::SmallVector<BreakContinueScope>& {
    return break_continue_stack_;
  }

  auto decl_name_stack() -> DeclNameStack& { return decl_name_stack_; }

  auto decl_state_stack() -> DeclStateStack& { return decl_state_stack_; }

  auto lexical_lookup() -> LexicalLookup& { return lexical_lookup_; }

  // Directly expose SemIR::File data accessors for brevity in calls.

  auto identifiers() -> StringStoreWrapper<IdentifierId>& {
    return sem_ir().identifiers();
  }
  auto ints() -> ValueStore<IntId>& { return sem_ir().ints(); }
  auto reals() -> ValueStore<RealId>& { return sem_ir().reals(); }
  auto string_literal_values() -> StringStoreWrapper<StringLiteralValueId>& {
    return sem_ir().string_literal_values();
  }
  auto bind_names() -> ValueStore<SemIR::BindNameId>& {
    return sem_ir().bind_names();
  }
  auto functions() -> ValueStore<SemIR::FunctionId>& {
    return sem_ir().functions();
  }
  auto classes() -> ValueStore<SemIR::ClassId>& { return sem_ir().classes(); }
  auto interfaces() -> ValueStore<SemIR::InterfaceId>& {
    return sem_ir().interfaces();
  }
  auto import_irs() -> ValueStore<SemIR::ImportIRId>& {
    return sem_ir().import_irs();
  }
  auto names() -> SemIR::NameStoreWrapper { return sem_ir().names(); }
  auto name_scopes() -> SemIR::NameScopeStore& {
    return sem_ir().name_scopes();
  }
  auto types() -> SemIR::TypeStore& { return sem_ir().types(); }
  auto type_blocks() -> SemIR::BlockValueStore<SemIR::TypeBlockId>& {
    return sem_ir().type_blocks();
  }
  // Instructions should be added with `AddInst` or `AddInstInNoBlock`. This is
  // `const` to prevent accidental misuse.
  auto insts() -> const SemIR::InstStore& { return sem_ir().insts(); }
  auto constant_values() -> SemIR::ConstantValueStore& {
    return sem_ir().constant_values();
  }
  auto inst_blocks() -> SemIR::InstBlockStore& {
    return sem_ir().inst_blocks();
  }
  auto constants() -> SemIR::ConstantStore& { return sem_ir().constants(); }

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
    // The sequential index of this scope entry within the file.
    ScopeIndex index;

    // The instruction associated with this entry, if any. This can be one of:
    //
    // - A `ClassDecl`, for a class definition scope.
    // - A `FunctionDecl`, for the outermost scope in a function
    //   definition.
    // - Invalid, for any other scope.
    SemIR::InstId scope_inst_id;

    // The name scope associated with this entry, if any.
    SemIR::NameScopeId scope_id;

    // The previous state of lexical_lookup_has_load_error_, restored on pop.
    bool prev_lexical_lookup_has_load_error;

    // Names which are registered with lexical_lookup_, and will need to be
    // unregistered when the scope ends.
    llvm::DenseSet<SemIR::NameId> names;

    // Whether a `returned var` was introduced in this scope, and needs to be
    // unregistered when the scope ends.
    bool has_returned_var = false;

    // TODO: This likely needs to track things which need to be destructed.
  };

  // If the passed in instruction ID is a ImportRefUnused, resolves it for use.
  // Called when name lookup intends to return an inst_id.
  auto ResolveIfImportRefUnused(SemIR::InstId inst_id) -> void;

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

  // A stack of scopes from which we can `return`.
  llvm::SmallVector<ReturnScope> return_scope_stack_;

  // A stack of `break` and `continue` targets.
  llvm::SmallVector<BreakContinueScope> break_continue_stack_;

  // A stack for scope context.
  llvm::SmallVector<ScopeStackEntry> scope_stack_;

  // Information about non-lexical scopes. This is a subset of the entries and
  // the information in scope_stack_.
  llvm::SmallVector<std::pair<ScopeIndex, SemIR::NameScopeId>>
      non_lexical_scope_stack_;

  // The index of the next scope that will be pushed onto scope_stack_. The
  // first is always the package scope.
  ScopeIndex next_scope_index_ = ScopeIndex::Package;

  // The stack used for qualified declaration name construction.
  DeclNameStack decl_name_stack_;

  // The stack of declarations that could have modifiers.
  DeclStateStack decl_state_stack_;

  // Tracks lexical lookup results.
  LexicalLookup lexical_lookup_;

  // Whether lexical_lookup_ has load errors, updated whenever scope_stack_ is
  // pushed or popped.
  bool lexical_lookup_has_load_error_ = false;

  // Cache of reverse mapping from type constants to types.
  //
  // TODO: Instead of mapping to a dense `TypeId` space, we could make `TypeId`
  // be a thin wrapper around `ConstantId` and only perform the lookup only when
  // we want to access the completeness and value representation of a type. It's
  // not clear whether that would result in more or fewer lookups.
  //
  // TODO: Should this be part of the `TypeStore`?
  llvm::DenseMap<SemIR::ConstantId, SemIR::TypeId> type_ids_for_type_constants_;

  // The list which will form NodeBlockId::Exports.
  llvm::SmallVector<SemIR::InstId> exports_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CONTEXT_H_

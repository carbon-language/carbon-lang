// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CONTEXT_H_
#define CARBON_TOOLCHAIN_CHECK_CONTEXT_H_

#include "common/map.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/check/decl_introducer_state.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/generic_region_stack.h"
#include "toolchain/check/global_init.h"
#include "toolchain/check/inst_block_stack.h"
#include "toolchain/check/node_stack.h"
#include "toolchain/check/param_and_arg_refs_stack.h"
#include "toolchain/check/scope_stack.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/parse/tree.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/import_ir.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Information about a scope in which we can perform name lookup.
struct LookupScope {
  // The name scope in which names are searched.
  SemIR::NameScopeId name_scope_id;
  // The specific for the name scope, or `Invalid` if the name scope is not
  // defined by a generic or we should perform lookup into the generic itself.
  SemIR::SpecificId specific_id;
};

// A result produced by name lookup.
struct LookupResult {
  // The specific in which the lookup result was found. `Invalid` if the result
  // was not found in a specific.
  SemIR::SpecificId specific_id;
  // The declaration that was found by name lookup.
  SemIR::InstId inst_id;
};

// Information about an access.
struct AccessInfo {
  // The constant being accessed.
  SemIR::ConstantId constant_id;

  // The highest allowed access for a lookup. For example, `Protected` allows
  // access to `Public` and `Protected` names, but not `Private`.
  SemIR::AccessKind highest_allowed_access;
};

// Context and shared functionality for semantics handlers.
class Context {
 public:
  using DiagnosticEmitter = Carbon::DiagnosticEmitter<SemIRLoc>;
  using DiagnosticBuilder = DiagnosticEmitter::DiagnosticBuilder;
  // A function that forms a diagnostic for some kind of problem. The
  // DiagnosticBuilder is returned rather than emitted so that the caller can
  // add contextual notes as appropriate.
  using BuildDiagnosticFn =
      llvm::function_ref<auto()->Context::DiagnosticBuilder>;

  // Stores references for work.
  explicit Context(const Lex::TokenizedBuffer& tokens,
                   DiagnosticEmitter& emitter, const Parse::Tree& parse_tree,
                   llvm::function_ref<const Parse::TreeAndSubtrees&()>
                       get_parse_tree_and_subtrees,
                   SemIR::File& sem_ir, llvm::raw_ostream* vlog_stream);

  // Marks an implementation TODO. Always returns false.
  auto TODO(SemIRLoc loc, std::string label) -> bool;

  // Runs verification that the processing cleanly finished.
  auto VerifyOnFinish() -> void;

  // Adds an instruction to the current block, returning the produced ID.
  auto AddInst(SemIR::LocIdAndInst loc_id_and_inst) -> SemIR::InstId;

  // Convenience for AddInst with typed nodes.
  template <typename InstT, typename LocT>
  auto AddInst(LocT loc, InstT inst)
      -> decltype(AddInst(SemIR::LocIdAndInst(loc, inst))) {
    return AddInst(SemIR::LocIdAndInst(loc, inst));
  }

  // Returns a LocIdAndInst for an instruction with an imported location. Checks
  // that the imported location is compatible with the kind of instruction being
  // created.
  template <typename InstT>
    requires SemIR::Internal::HasNodeId<InstT>
  auto MakeImportedLocAndInst(SemIR::ImportIRInstId imported_loc_id, InstT inst)
      -> SemIR::LocIdAndInst {
    if constexpr (!SemIR::Internal::HasUntypedNodeId<InstT>) {
      CheckCompatibleImportedNodeKind(imported_loc_id, InstT::Kind);
    }
    return SemIR::LocIdAndInst::UncheckedLoc(imported_loc_id, inst);
  }

  // Adds an instruction in no block, returning the produced ID. Should be used
  // rarely.
  auto AddInstInNoBlock(SemIR::LocIdAndInst loc_id_and_inst) -> SemIR::InstId;

  // Convenience for AddInstInNoBlock with typed nodes.
  template <typename InstT, typename LocT>
  auto AddInstInNoBlock(LocT loc, InstT inst)
      -> decltype(AddInstInNoBlock(SemIR::LocIdAndInst(loc, inst))) {
    return AddInstInNoBlock(SemIR::LocIdAndInst(loc, inst));
  }

  // Adds an instruction to the current block, returning the produced ID. The
  // instruction is a placeholder that is expected to be replaced by
  // `ReplaceInstBeforeConstantUse`.
  auto AddPlaceholderInst(SemIR::LocIdAndInst loc_id_and_inst) -> SemIR::InstId;

  // Adds an instruction in no block, returning the produced ID. Should be used
  // rarely. The instruction is a placeholder that is expected to be replaced by
  // `ReplaceInstBeforeConstantUse`.
  auto AddPlaceholderInstInNoBlock(SemIR::LocIdAndInst loc_id_and_inst)
      -> SemIR::InstId;

  // Adds an instruction to the current pattern block, returning the produced
  // ID.
  auto AddPatternInst(SemIR::LocIdAndInst loc_id_and_inst) -> SemIR::InstId;

  // Convenience for AddPatternInst with typed nodes.
  template <typename InstT>
    requires(SemIR::Internal::HasNodeId<InstT>)
  auto AddPatternInst(decltype(InstT::Kind)::TypedNodeId node_id, InstT inst)
      -> SemIR::InstId {
    return AddPatternInst(SemIR::LocIdAndInst(node_id, inst));
  }

  // Adds an instruction to the constants block, returning the produced ID.
  auto AddConstant(SemIR::Inst inst, bool is_symbolic) -> SemIR::ConstantId;

  // Pushes a parse tree node onto the stack, storing the SemIR::Inst as the
  // result.
  template <typename InstT>
    requires(SemIR::Internal::HasNodeId<InstT>)
  auto AddInstAndPush(decltype(InstT::Kind)::TypedNodeId node_id, InstT inst)
      -> void {
    node_stack_.Push(node_id, AddInst(node_id, inst));
  }

  // Replaces the instruction `inst_id` with `loc_id_and_inst`. The instruction
  // is required to not have been used in any constant evaluation, either
  // because it's newly created and entirely unused, or because it's only used
  // in a position that constant evaluation ignores, such as a return slot.
  auto ReplaceLocIdAndInstBeforeConstantUse(SemIR::InstId inst_id,
                                            SemIR::LocIdAndInst loc_id_and_inst)
      -> void;

  // Replaces the instruction `inst_id` with `inst`, not affecting location.
  // The instruction is required to not have been used in any constant
  // evaluation, either because it's newly created and entirely unused, or
  // because it's only used in a position that constant evaluation ignores, such
  // as a return slot.
  auto ReplaceInstBeforeConstantUse(SemIR::InstId inst_id, SemIR::Inst inst)
      -> void;

  // Sets only the parse node of an instruction. This is only used when setting
  // the parse node of an imported namespace. Versus
  // ReplaceInstBeforeConstantUse, it is safe to use after the namespace is used
  // in constant evaluation. It's exposed this way mainly so that `insts()` can
  // remain const.
  auto SetNamespaceNodeId(SemIR::InstId inst_id, Parse::NodeId node_id)
      -> void {
    sem_ir().insts().SetLocId(inst_id, SemIR::LocId(node_id));
  }

  // Adds a name to name lookup. Prints a diagnostic for name conflicts.
  auto AddNameToLookup(SemIR::NameId name_id, SemIR::InstId target_id) -> void;

  // Performs name lookup in a specified scope for a name appearing in a
  // declaration, returning the referenced instruction. If scope_id is invalid,
  // uses the current contextual scope.
  auto LookupNameInDecl(SemIR::LocId loc_id, SemIR::NameId name_id,
                        SemIR::NameScopeId scope_id) -> SemIR::InstId;

  // Performs an unqualified name lookup, returning the referenced instruction.
  auto LookupUnqualifiedName(Parse::NodeId node_id, SemIR::NameId name_id,
                             bool required = true) -> LookupResult;

  // Performs a name lookup in a specified scope, returning the referenced
  // instruction. Does not look into extended scopes. Returns an invalid
  // instruction if the name is not found.
  auto LookupNameInExactScope(SemIRLoc loc, SemIR::NameId name_id,
                              SemIR::NameScopeId scope_id,
                              const SemIR::NameScope& scope)
      -> std::pair<SemIR::InstId, SemIR::AccessKind>;

  // Performs a qualified name lookup in a specified scope and in scopes that
  // it extends, returning the referenced instruction.
  auto LookupQualifiedName(SemIRLoc loc, SemIR::NameId name_id,
                           LookupScope scope, bool required = true,
                           std::optional<AccessInfo> access_info = std::nullopt)
      -> LookupResult;

  // Returns the instruction corresponding to a name in the core package, or
  // BuiltinError if not found.
  auto LookupNameInCore(SemIRLoc loc, llvm::StringRef name) -> SemIR::InstId;

  // Prints a diagnostic for a duplicate name.
  auto DiagnoseDuplicateName(SemIRLoc dup_def, SemIRLoc prev_def) -> void;

  // Prints a diagnostic for a missing name.
  auto DiagnoseNameNotFound(SemIRLoc loc, SemIR::NameId name_id) -> void;

  // Adds a note to a diagnostic explaining that a class is incomplete.
  auto NoteIncompleteClass(SemIR::ClassId class_id, DiagnosticBuilder& builder)
      -> void;

  // Adds a note to a diagnostic explaining that an interface is not defined.
  auto NoteUndefinedInterface(SemIR::InterfaceId interface_id,
                              DiagnosticBuilder& builder) -> void;

  // Returns the current scope, if it is of the specified kind. Otherwise,
  // returns nullopt.
  template <typename InstT>
  auto GetCurrentScopeAs() -> std::optional<InstT> {
    return scope_stack().GetCurrentScopeAs<InstT>(sem_ir());
  }

  // Adds a `Branch` instruction branching to a new instruction block, and
  // returns the ID of the new block. All paths to the branch target must go
  // through the current block, though not necessarily through this branch.
  auto AddDominatedBlockAndBranch(Parse::NodeId node_id) -> SemIR::InstBlockId;

  // Adds a `Branch` instruction branching to a new instruction block with a
  // value, and returns the ID of the new block. All paths to the branch target
  // must go through the current block.
  auto AddDominatedBlockAndBranchWithArg(Parse::NodeId node_id,
                                         SemIR::InstId arg_id)
      -> SemIR::InstBlockId;

  // Adds a `BranchIf` instruction branching to a new instruction block, and
  // returns the ID of the new block. All paths to the branch target must go
  // through the current block.
  auto AddDominatedBlockAndBranchIf(Parse::NodeId node_id,
                                    SemIR::InstId cond_id)
      -> SemIR::InstBlockId;

  // Handles recovergence of control flow. Adds branches from the top
  // `num_blocks` on the instruction block stack to a new block, pops the
  // existing blocks, and pushes the new block onto the instruction block stack.
  auto AddConvergenceBlockAndPush(Parse::NodeId node_id, int num_blocks)
      -> void;

  // Handles recovergence of control flow with a result value. Adds branches
  // from the top few blocks on the instruction block stack to a new block, pops
  // the existing blocks, and pushes the new block onto the instruction block
  // stack. The number of blocks popped is the size of `block_args`, and the
  // corresponding result values are the elements of `block_args`. Returns an
  // instruction referring to the result value.
  auto AddConvergenceBlockWithArgAndPush(
      Parse::NodeId node_id, std::initializer_list<SemIR::InstId> block_args)
      -> SemIR::InstId;

  // Sets the constant value of a block argument created as the result of a
  // branch.  `select_id` should be a `BlockArg` that selects between two
  // values. `cond_id` is the condition, `if_false` is the value to use if the
  // condition is false, and `if_true` is the value to use if the condition is
  // true.  We don't track enough information in the `BlockArg` inst for
  // `TryEvalInst` to do this itself.
  auto SetBlockArgResultBeforeConstantUse(SemIR::InstId select_id,
                                          SemIR::InstId cond_id,
                                          SemIR::InstId if_true,
                                          SemIR::InstId if_false) -> void;

  // Add the current code block to the enclosing function.
  // TODO: The node_id is taken for expressions, which can occur in
  // non-function contexts. This should be refactored to support non-function
  // contexts, and node_id removed.
  auto AddCurrentCodeBlockToFunction(
      Parse::NodeId node_id = Parse::NodeId::Invalid) -> void;

  // Returns whether the current position in the current block is reachable.
  auto is_current_position_reachable() -> bool;

  // Returns the type ID for a constant of type `type`.
  auto GetTypeIdForTypeConstant(SemIR::ConstantId constant_id) -> SemIR::TypeId;

  // Returns the type ID for an instruction whose constant value is of type
  // `type`.
  auto GetTypeIdForTypeInst(SemIR::InstId inst_id) -> SemIR::TypeId {
    return GetTypeIdForTypeConstant(constant_values().Get(inst_id));
  }

  // Attempts to complete the type `type_id`. Returns `true` if the type is
  // complete, or `false` if it could not be completed. A complete type has
  // known object and value representations.
  //
  // If the type is not complete, `diagnoser` is invoked to diagnose the issue,
  // if a `diagnoser` is provided. The builder it returns will be annotated to
  // describe the reason why the type is not complete.
  auto TryToCompleteType(
      SemIR::TypeId type_id,
      std::optional<BuildDiagnosticFn> diagnoser = std::nullopt) -> bool;

  // Attempts to complete and define the type `type_id`. Returns `true` if the
  // type is defined, or `false` if no definition is available. A defined type
  // has known members.
  //
  // This is the same as `TryToCompleteType` except for interfaces, which are
  // complete before they are fully defined.
  auto TryToDefineType(
      SemIR::TypeId type_id,
      std::optional<BuildDiagnosticFn> diagnoser = std::nullopt) -> bool;

  // Returns the type `type_id` as a complete type, or produces an incomplete
  // type error and returns an error type. This is a convenience wrapper around
  // TryToCompleteType.
  auto AsCompleteType(SemIR::TypeId type_id, BuildDiagnosticFn diagnoser)
      -> SemIR::TypeId {
    return TryToCompleteType(type_id, diagnoser) ? type_id
                                                 : SemIR::TypeId::Error;
  }

  // Returns whether `type_id` represents a facet type.
  auto IsFacetType(SemIR::TypeId type_id) -> bool {
    return type_id == SemIR::TypeId::TypeType ||
           types().Is<SemIR::InterfaceType>(type_id);
  }

  // TODO: Consider moving these `Get*Type` functions to a separate class.

  // Gets the type for the name of an associated entity.
  auto GetAssociatedEntityType(SemIR::TypeId interface_type_id,
                               SemIR::TypeId entity_type_id) -> SemIR::TypeId;

  // Gets a builtin type. The returned type will be complete.
  auto GetBuiltinType(SemIR::BuiltinInstKind kind) -> SemIR::TypeId;

  // Gets a function type. The returned type will be complete.
  auto GetFunctionType(SemIR::FunctionId fn_id, SemIR::SpecificId specific_id)
      -> SemIR::TypeId;

  // Gets a generic class type, which is the type of a name of a generic class,
  // such as the type of `Vector` given `class Vector(T:! type)`. The returned
  // type will be complete.
  auto GetGenericClassType(SemIR::ClassId class_id,
                           SemIR::SpecificId enclosing_specific_id)
      -> SemIR::TypeId;

  // Gets a generic interface type, which is the type of a name of a generic
  // interface, such as the type of `AddWith` given
  // `interface AddWith(T:! type)`. The returned type will be complete.
  auto GetGenericInterfaceType(SemIR::InterfaceId interface_id,
                               SemIR::SpecificId enclosing_specific_id)
      -> SemIR::TypeId;

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

  // Adds an exported name.
  auto AddExport(SemIR::InstId inst_id) -> void { exports_.push_back(inst_id); }

  auto Finalize() -> void;

  // Sets the total number of IRs which exist. This is used to prepare a map
  // from IR to imported IR.
  auto SetTotalIRCount(int num_irs) -> void {
    CARBON_CHECK(check_ir_map_.empty(), "SetTotalIRCount is only called once");
    check_ir_map_.resize(num_irs, SemIR::ImportIRId::Invalid);
  }

  // Returns the imported IR ID for an IR, or invalid if not imported.
  auto GetImportIRId(const SemIR::File& sem_ir) -> SemIR::ImportIRId& {
    return check_ir_map_[sem_ir.check_ir_id().index];
  }

  // True if the current file is an impl file.
  auto IsImplFile() -> bool {
    return sem_ir_->import_irs().Get(SemIR::ImportIRId::ApiForImpl).sem_ir !=
           nullptr;
  }

  // Prints information for a stack dump.
  auto PrintForStackDump(llvm::raw_ostream& output) const -> void;

  // Prints the the formatted sem_ir to stderr.
  LLVM_DUMP_METHOD auto DumpFormattedFile() const -> void;

  // Get the Lex::TokenKind of a node for diagnostics.
  auto token_kind(Parse::NodeId node_id) -> Lex::TokenKind {
    return tokens().GetKind(parse_tree().node_token(node_id));
  }

  auto tokens() -> const Lex::TokenizedBuffer& { return *tokens_; }

  auto emitter() -> DiagnosticEmitter& { return *emitter_; }

  auto parse_tree() -> const Parse::Tree& { return *parse_tree_; }

  auto parse_tree_and_subtrees() -> const Parse::TreeAndSubtrees& {
    return get_parse_tree_and_subtrees_();
  }

  auto sem_ir() -> SemIR::File& { return *sem_ir_; }

  auto node_stack() -> NodeStack& { return node_stack_; }

  auto inst_block_stack() -> InstBlockStack& { return inst_block_stack_; }
  auto pattern_block_stack() -> InstBlockStack& { return pattern_block_stack_; }

  auto param_and_arg_refs_stack() -> ParamAndArgRefsStack& {
    return param_and_arg_refs_stack_;
  }

  auto args_type_info_stack() -> InstBlockStack& {
    return args_type_info_stack_;
  }

  auto decl_name_stack() -> DeclNameStack& { return decl_name_stack_; }

  auto decl_introducer_state_stack() -> DeclIntroducerStateStack& {
    return decl_introducer_state_stack_;
  }

  auto scope_stack() -> ScopeStack& { return scope_stack_; }

  auto return_scope_stack() -> llvm::SmallVector<ScopeStack::ReturnScope>& {
    return scope_stack().return_scope_stack();
  }

  auto break_continue_stack()
      -> llvm::SmallVector<ScopeStack::BreakContinueScope>& {
    return scope_stack().break_continue_stack();
  }

  auto generic_region_stack() -> GenericRegionStack& {
    return generic_region_stack_;
  }

  auto import_ir_constant_values()
      -> llvm::SmallVector<SemIR::ConstantValueStore, 0>& {
    return import_ir_constant_values_;
  }

  // Directly expose SemIR::File data accessors for brevity in calls.

  auto identifiers() -> CanonicalValueStore<IdentifierId>& {
    return sem_ir().identifiers();
  }
  auto ints() -> CanonicalValueStore<IntId>& { return sem_ir().ints(); }
  auto reals() -> ValueStore<RealId>& { return sem_ir().reals(); }
  auto floats() -> FloatValueStore& { return sem_ir().floats(); }
  auto string_literal_values() -> CanonicalValueStore<StringLiteralValueId>& {
    return sem_ir().string_literal_values();
  }
  auto entity_names() -> SemIR::EntityNameStore& {
    return sem_ir().entity_names();
  }
  auto functions() -> ValueStore<SemIR::FunctionId>& {
    return sem_ir().functions();
  }
  auto classes() -> ValueStore<SemIR::ClassId>& { return sem_ir().classes(); }
  auto interfaces() -> ValueStore<SemIR::InterfaceId>& {
    return sem_ir().interfaces();
  }
  auto impls() -> SemIR::ImplStore& { return sem_ir().impls(); }
  auto generics() -> SemIR::GenericStore& { return sem_ir().generics(); }
  auto specifics() -> SemIR::SpecificStore& { return sem_ir().specifics(); }
  auto import_irs() -> ValueStore<SemIR::ImportIRId>& {
    return sem_ir().import_irs();
  }
  auto import_ir_insts() -> ValueStore<SemIR::ImportIRInstId>& {
    return sem_ir().import_ir_insts();
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

  auto definitions_required() -> llvm::SmallVector<SemIR::InstId>& {
    return definitions_required_;
  }

  auto global_init() -> GlobalInit& { return global_init_; }

  auto import_ref_ids() -> llvm::SmallVector<SemIR::InstId>& {
    return import_ref_ids_;
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

  // Checks that the provided imported location has a node kind that is
  // compatible with that of the given instruction.
  auto CheckCompatibleImportedNodeKind(SemIR::ImportIRInstId imported_loc_id,
                                       SemIR::InstKind kind) -> void;

  // Finish producing an instruction. Set its constant value, and register it in
  // any applicable instruction lists.
  auto FinishInst(SemIR::InstId inst_id, SemIR::Inst inst) -> void;

  // Tokens for getting data on literals.
  const Lex::TokenizedBuffer* tokens_;

  // Handles diagnostics.
  DiagnosticEmitter* emitter_;

  // The file's parse tree.
  const Parse::Tree* parse_tree_;

  // Returns a lazily constructed TreeAndSubtrees.
  llvm::function_ref<const Parse::TreeAndSubtrees&()>
      get_parse_tree_and_subtrees_;

  // The SemIR::File being added to.
  SemIR::File* sem_ir_;

  // Whether to print verbose output.
  llvm::raw_ostream* vlog_stream_;

  // The stack during Build. Will contain file-level parse nodes on return.
  NodeStack node_stack_;

  // The stack of instruction blocks being used for general IR generation.
  InstBlockStack inst_block_stack_;

  // The stack of instruction blocks that contain pattern instructions.
  InstBlockStack pattern_block_stack_;

  // The stack of instruction blocks being used for param and arg ref blocks.
  ParamAndArgRefsStack param_and_arg_refs_stack_;

  // The stack of instruction blocks being used for type information while
  // processing arguments. This is used in parallel with
  // param_and_arg_refs_stack_. It's currently only used for struct literals,
  // where we need to track names for a type separate from the literal
  // arguments.
  InstBlockStack args_type_info_stack_;

  // The stack used for qualified declaration name construction.
  DeclNameStack decl_name_stack_;

  // The stack of declarations that could have modifiers.
  DeclIntroducerStateStack decl_introducer_state_stack_;

  // The stack of scopes we are currently within.
  ScopeStack scope_stack_;

  // The stack of generic regions we are currently within.
  GenericRegionStack generic_region_stack_;

  // Cache of reverse mapping from type constants to types.
  //
  // TODO: Instead of mapping to a dense `TypeId` space, we could make `TypeId`
  // be a thin wrapper around `ConstantId` and only perform the lookup only when
  // we want to access the completeness and value representation of a type. It's
  // not clear whether that would result in more or fewer lookups.
  //
  // TODO: Should this be part of the `TypeStore`?
  Map<SemIR::ConstantId, SemIR::TypeId> type_ids_for_type_constants_;

  // The list which will form NodeBlockId::Exports.
  llvm::SmallVector<SemIR::InstId> exports_;

  // Maps CheckIRId to ImportIRId.
  llvm::SmallVector<SemIR::ImportIRId> check_ir_map_;

  // Per-import constant values. These refer to the main IR and mainly serve as
  // a lookup table for quick access.
  //
  // Inline 0 elements because it's expected to require heap allocation.
  llvm::SmallVector<SemIR::ConstantValueStore, 0> import_ir_constant_values_;

  // Declaration instructions of entities that should have definitions by the
  // end of the current source file.
  llvm::SmallVector<SemIR::InstId> definitions_required_;

  // State for global initialization.
  GlobalInit global_init_;

  // A list of import refs which can't be inserted into their current context.
  // They're typically added during name lookup or import ref resolution, where
  // the current block on inst_block_stack_ is unrelated.
  //
  // These are instead added here because they're referenced by other
  // instructions and needs to be visible in textual IR.
  // FinalizeImportRefBlock() will produce an inst block for them.
  llvm::SmallVector<SemIR::InstId> import_ref_ids_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CONTEXT_H_

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CONTEXT_H_
#define CARBON_TOOLCHAIN_CHECK_CONTEXT_H_

#include <string>

#include "common/map.h"
#include "common/ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/check/decl_introducer_state.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/full_pattern_stack.h"
#include "toolchain/check/generic_region_stack.h"
#include "toolchain/check/global_init.h"
#include "toolchain/check/inst_block_stack.h"
#include "toolchain/check/node_stack.h"
#include "toolchain/check/param_and_arg_refs_stack.h"
#include "toolchain/check/region_stack.h"
#include "toolchain/check/scope_stack.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
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

// Context stored during check.
//
// This file stores state, and members objects may provide an API. Other files
// may also have helpers that operate on Context. To keep this file manageable,
// please put logic into other files.
//
// For example, consider the API for functions:
// - `context.functions()`: Exposes storage of `SemIR::Function` objects.
// - `toolchain/check/function.h`: Contains helper functions which use
//   `Check::Context`.
// - `toolchain/sem_ir/function.h`: Contains helper functions which only need
//   `SemIR` objects, for which it's helpful not to depend on `Check::Context`
//   (for example, shared with lowering).
class Context {
 public:
  // Stores references for work.
  explicit Context(DiagnosticEmitter<SemIRLoc>* emitter,
                   Parse::GetTreeAndSubtreesFn tree_and_subtrees_getter,
                   SemIR::File* sem_ir, int imported_ir_count,
                   int total_ir_count, llvm::raw_ostream* vlog_stream);

  // Marks an implementation TODO. Always returns false.
  auto TODO(SemIRLoc loc, std::string label) -> bool;

  // Runs verification that the processing cleanly finished.
  auto VerifyOnFinish() const -> void;

  // Prints information for a stack dump.
  auto PrintForStackDump(llvm::raw_ostream& output) const -> void;

  // Get the Lex::TokenKind of a node for diagnostics.
  auto token_kind(Parse::NodeId node_id) -> Lex::TokenKind {
    return tokens().GetKind(parse_tree().node_token(node_id));
  }

  auto emitter() -> DiagnosticEmitter<SemIRLoc>& { return *emitter_; }

  auto parse_tree_and_subtrees() -> const Parse::TreeAndSubtrees& {
    return tree_and_subtrees_getter_();
  }

  auto sem_ir() -> SemIR::File& { return *sem_ir_; }
  auto sem_ir() const -> const SemIR::File& { return *sem_ir_; }

  // Convenience functions for major phase data.
  auto parse_tree() const -> const Parse::Tree& {
    return sem_ir_->parse_tree();
  }
  auto tokens() const -> const Lex::TokenizedBuffer& {
    return parse_tree().tokens();
  }

  auto vlog_stream() -> llvm::raw_ostream* { return vlog_stream_; }

  auto node_stack() -> NodeStack& { return node_stack_; }

  auto inst_block_stack() -> InstBlockStack& { return inst_block_stack_; }
  auto pattern_block_stack() -> InstBlockStack& { return pattern_block_stack_; }

  auto param_and_arg_refs_stack() -> ParamAndArgRefsStack& {
    return param_and_arg_refs_stack_;
  }

  auto args_type_info_stack() -> InstBlockStack& {
    return args_type_info_stack_;
  }

  auto struct_type_fields_stack() -> ArrayStack<SemIR::StructTypeField>& {
    return struct_type_fields_stack_;
  }

  auto field_decls_stack() -> ArrayStack<SemIR::InstId>& {
    return field_decls_stack_;
  }

  auto decl_name_stack() -> DeclNameStack& { return decl_name_stack_; }

  auto decl_introducer_state_stack() -> DeclIntroducerStateStack& {
    return decl_introducer_state_stack_;
  }

  auto scope_stack() -> ScopeStack& { return scope_stack_; }

  // Conveneicne functions for frequently-used `scope_stack` members.
  auto return_scope_stack() -> llvm::SmallVector<ScopeStack::ReturnScope>& {
    return scope_stack().return_scope_stack();
  }
  auto break_continue_stack()
      -> llvm::SmallVector<ScopeStack::BreakContinueScope>& {
    return scope_stack().break_continue_stack();
  }
  auto full_pattern_stack() -> FullPatternStack& {
    return scope_stack_.full_pattern_stack();
  }

  auto generic_region_stack() -> GenericRegionStack& {
    return generic_region_stack_;
  }

  auto vtable_stack() -> InstBlockStack& { return vtable_stack_; }

  auto exports() -> llvm::SmallVector<SemIR::InstId>& { return exports_; }

  auto check_ir_map() -> llvm::MutableArrayRef<SemIR::ImportIRId> {
    return check_ir_map_;
  }

  auto import_ir_constant_values()
      -> llvm::SmallVector<SemIR::ConstantValueStore, 0>& {
    return import_ir_constant_values_;
  }

  auto definitions_required() -> llvm::SmallVector<SemIR::InstId>& {
    return definitions_required_;
  }

  auto global_init() -> GlobalInit& { return global_init_; }

  auto import_ref_ids() -> llvm::SmallVector<SemIR::InstId>& {
    return import_ref_ids_;
  }

  // Pre-computed parts of a binding pattern.
  // TODO: Consider putting this behind a narrower API to guard against emitting
  // multiple times.
  struct BindingPatternInfo {
    // The corresponding AnyBindName inst.
    SemIR::InstId bind_name_id;
    // The region of insts that computes the type of the binding.
    SemIR::ExprRegionId type_expr_region_id;
  };
  auto bind_name_map() -> Map<SemIR::InstId, BindingPatternInfo>& {
    return bind_name_map_;
  }

  auto var_storage_map() -> Map<SemIR::InstId, SemIR::InstId>& {
    return var_storage_map_;
  }

  // During Choice typechecking, each alternative turns into a name binding on
  // the Choice type, but this can't be done until the full Choice type is
  // known. This represents each binding to be done at the end of checking the
  // Choice type.
  struct ChoiceDeferredBinding {
    Parse::NodeIdOneOf<Parse::ChoiceAlternativeListCommaId,
                       Parse::ChoiceDefinitionId>
        node_id;
    NameComponent name_component;
  };
  auto choice_deferred_bindings() -> llvm::SmallVector<ChoiceDeferredBinding>& {
    return choice_deferred_bindings_;
  }

  auto region_stack() -> RegionStack& { return region_stack_; }

  // An ongoing impl lookup, used to ensure termination.
  struct ImplLookupStackEntry {
    SemIR::ConstantId query_self_const_id;
    SemIR::ConstantId query_facet_type_const_id;
    // The location of the impl being looked at for the stack entry.
    SemIR::InstId impl_loc = SemIR::InstId::None;
  };
  auto impl_lookup_stack() -> llvm::SmallVector<ImplLookupStackEntry>& {
    return impl_lookup_stack_;
  }

  // --------------------------------------------------------------------------
  // Directly expose SemIR::File data accessors for brevity in calls.
  // --------------------------------------------------------------------------

  auto identifiers() -> SharedValueStores::IdentifierStore& {
    return sem_ir().identifiers();
  }
  auto ints() -> SharedValueStores::IntStore& { return sem_ir().ints(); }
  auto reals() -> SharedValueStores::RealStore& { return sem_ir().reals(); }
  auto floats() -> SharedValueStores::FloatStore& { return sem_ir().floats(); }
  auto string_literal_values() -> SharedValueStores::StringLiteralStore& {
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
  auto associated_constants() -> ValueStore<SemIR::AssociatedConstantId>& {
    return sem_ir().associated_constants();
  }
  auto facet_types() -> CanonicalValueStore<SemIR::FacetTypeId>& {
    return sem_ir().facet_types();
  }
  auto complete_facet_types() -> SemIR::File::CompleteFacetTypeStore& {
    return sem_ir().complete_facet_types();
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
  auto struct_type_fields() -> SemIR::StructTypeFieldsStore& {
    return sem_ir().struct_type_fields();
  }
  auto types() -> SemIR::TypeStore& { return sem_ir().types(); }
  auto type_blocks() -> SemIR::BlockValueStore<SemIR::TypeBlockId>& {
    return sem_ir().type_blocks();
  }
  // Instructions should be added with `AddInst` or `AddInstInNoBlock` from
  // `inst.h`. This is `const` to prevent accidental misuse.
  auto insts() -> const SemIR::InstStore& { return sem_ir().insts(); }
  auto constant_values() -> SemIR::ConstantValueStore& {
    return sem_ir().constant_values();
  }
  auto inst_blocks() -> SemIR::InstBlockStore& {
    return sem_ir().inst_blocks();
  }
  auto constants() -> SemIR::ConstantStore& { return sem_ir().constants(); }

  // --------------------------------------------------------------------------
  // End of SemIR::File members.
  // --------------------------------------------------------------------------

 private:
  // Handles diagnostics.
  DiagnosticEmitter<SemIRLoc>* emitter_;

  // Returns a lazily constructed TreeAndSubtrees.
  Parse::GetTreeAndSubtreesFn tree_and_subtrees_getter_;

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

  // The stack of StructTypeFields for in-progress StructTypeLiterals.
  ArrayStack<SemIR::StructTypeField> struct_type_fields_stack_;

  // The stack of FieldDecls for in-progress Class definitions.
  ArrayStack<SemIR::InstId> field_decls_stack_;

  // The stack used for qualified declaration name construction.
  DeclNameStack decl_name_stack_;

  // The stack of declarations that could have modifiers.
  DeclIntroducerStateStack decl_introducer_state_stack_;

  // The stack of scopes we are currently within.
  ScopeStack scope_stack_;

  // The stack of generic regions we are currently within.
  GenericRegionStack generic_region_stack_;

  // Contains a vtable block for each `class` scope which is currently being
  // defined, regardless of whether the class can have virtual functions.
  InstBlockStack vtable_stack_;

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

  // Map from an AnyBindingPattern inst to precomputed parts of the
  // pattern-match SemIR for it.
  Map<SemIR::InstId, BindingPatternInfo> bind_name_map_;

  // Map from VarPattern insts to the corresponding VarStorage insts. The
  // VarStorage insts are allocated, emitted, and stored in the map after
  // processing the enclosing full-pattern.
  Map<SemIR::InstId, SemIR::InstId> var_storage_map_;

  // Each alternative in a Choice gets an entry here, they are stored in
  // declaration order. The vector is consumed and emptied at the end of the
  // Choice definition.
  //
  // TODO: This may need to be a stack of vectors if it becomes possible to
  // define a Choice type inside an alternative's parameter set.
  llvm::SmallVector<ChoiceDeferredBinding> choice_deferred_bindings_;

  // Stack of single-entry regions being built.
  RegionStack region_stack_;

  // Tracks all ongoing impl lookups in order to ensure that lookup terminates
  // via the acyclic rule and the termination rule.
  llvm::SmallVector<ImplLookupStackEntry> impl_lookup_stack_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CONTEXT_H_

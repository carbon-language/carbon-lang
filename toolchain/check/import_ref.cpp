// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/import_ref.h"

#include "common/check.h"
#include "toolchain/check/context.h"
#include "toolchain/check/eval.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/typed_insts.h"
#include "toolchain/sem_ir/value_stores.h"

namespace Carbon::Check {

// Resolves an instruction from an imported IR into a constant referring to the
// current IR.
class ImportRefResolver {
 public:
  explicit ImportRefResolver(Context& context, SemIR::ImportIRId import_ir_id)
      : context_(context),
        import_ir_id_(import_ir_id),
        import_ir_(*context_.import_irs().Get(import_ir_id)),
        import_ir_constant_values_(
            context_.import_ir_constant_values()[import_ir_id.index]) {}

  // Iteratively resolves an imported instruction's inner references until a
  // constant ID referencing the current IR is produced. When an outer
  // instruction has unresolved inner references, it will add them to the stack
  // for inner evaluation and reattempt outer evaluation after.
  auto Resolve(SemIR::InstId inst_id) -> SemIR::ConstantId {
    work_stack_.push_back(inst_id);
    while (!work_stack_.empty()) {
      auto inst_id = work_stack_.back();
      CARBON_CHECK(inst_id.is_valid());

      // Double-check that the constant still doesn't have a calculated value.
      // This should typically be checked before adding it, but a given constant
      // may be added multiple times before its constant is evaluated.
      if (auto current_const_id = import_ir_constant_values_.Get(inst_id);
          current_const_id.is_valid()) {
        work_stack_.pop_back();
      } else if (auto new_const_id = TryResolveInst(inst_id);
                 new_const_id.is_valid()) {
        import_ir_constant_values_.Set(inst_id, new_const_id);
        work_stack_.pop_back();
      }
    }
    auto constant_id = import_ir_constant_values_.Get(inst_id);
    CARBON_CHECK(constant_id.is_valid());
    return constant_id;
  }

  // Wraps constant evaluation with logic to handle types.
  auto ResolveType(SemIR::TypeId import_type_id) -> SemIR::TypeId {
    if (!import_type_id.is_valid()) {
      return import_type_id;
    }

    auto import_type_inst_id = import_ir_.types().GetInstId(import_type_id);
    CARBON_CHECK(import_type_inst_id.is_valid());

    if (import_type_inst_id.is_builtin()) {
      // Builtins don't require constant resolution; we can use them directly.
      return context_.GetBuiltinType(import_type_inst_id.builtin_kind());
    } else {
      return context_.GetTypeIdForTypeConstant(Resolve(import_type_inst_id));
    }
  }

 private:
  // For imported entities, we use an invalid enclosing scope. This will be okay
  // if the scope isn't used later, but we may need to change logic for this if
  // the behavior changes.
  static constexpr SemIR::NameScopeId NoEnclosingScopeForImports =
      SemIR::NameScopeId::Invalid;

  // Returns true if new unresolved constants were found.
  //
  // At the start of a function, do:
  //   auto initial_work = work_stack_.size();
  // Then when determining:
  //   if (HasUnresolved(initial_work)) { ... }
  auto HasUnresolved(size_t initial_work) -> bool {
    CARBON_CHECK(initial_work <= work_stack_.size());
    return initial_work < work_stack_.size();
  }

  // Returns the ConstantId for an InstId. Adds it to the stack and sets
  // has_unresolved to true if it's not ready.
  auto GetConstantId(SemIR::InstId inst_id) -> SemIR::ConstantId {
    auto const_id = import_ir_constant_values_.Get(inst_id);
    if (!const_id.is_valid()) {
      work_stack_.push_back(inst_id);
    }
    return const_id;
  }

  // Returns the ConstantId for a TypeId. Adds it to the stack and sets
  // has_unresolved to true if it's not ready.
  auto GetConstantId(SemIR::TypeId type_id) -> SemIR::ConstantId {
    return GetConstantId(import_ir_.types().GetInstId(type_id));
  }

  // Given a param_refs_id, returns the necessary constants to convert it. Sets
  // has_unresolved to true if any aren't ready.
  auto GetParamConstantIds(SemIR::InstBlockId param_refs_id)
      -> llvm::SmallVector<SemIR::ConstantId> {
    if (param_refs_id == SemIR::InstBlockId::Empty) {
      return {};
    }
    const auto& param_refs = import_ir_.inst_blocks().Get(param_refs_id);
    llvm::SmallVector<SemIR::ConstantId> const_ids;
    const_ids.reserve(param_refs.size());
    for (auto inst_id : param_refs) {
      const_ids.push_back(
          GetConstantId(import_ir_.insts().Get(inst_id).type_id()));
    }
    return const_ids;
  }

  // Given a param_refs_id and const_ids from GetParamConstantIds, returns a
  // version localized to the current IR.
  auto GetParamRefsId(SemIR::InstBlockId param_refs_id,
                      const llvm::SmallVector<SemIR::ConstantId>& const_ids)
      -> SemIR::InstBlockId {
    if (param_refs_id == SemIR::InstBlockId::Empty) {
      return SemIR::InstBlockId::Empty;
    }
    const auto& param_refs = import_ir_.inst_blocks().Get(param_refs_id);
    llvm::SmallVector<SemIR::InstId> new_param_refs;
    for (auto [ref_id, const_id] : llvm::zip(param_refs, const_ids)) {
      new_param_refs.push_back(context_.AddInstInNoBlock(
          {SemIR::ImportRefUsed{context_.GetTypeIdForTypeConstant(const_id),
                                import_ir_id_, ref_id}}));
    }
    return context_.inst_blocks().Add(new_param_refs);
  }

  // Translates a NameId from the import IR to a local NameId.
  auto GetNameId(SemIR::NameId name_id) -> SemIR::NameId {
    if (auto ident_id = name_id.AsIdentifierId(); ident_id.is_valid()) {
      return SemIR::NameId::ForIdentifier(
          context_.identifiers().Add(import_ir_.identifiers().Get(ident_id)));
    }
    return name_id;
  }

  // Tries to resolve the InstId, returning a constant when ready, or Invalid if
  // more has been added to the stack. A similar API is followed for all
  // following TryResolveTypedInst helper functions.
  //
  // Logic for each TryResolveTypedInst will be in two phases:
  //   1. Gather all input constants.
  //   2. Produce an output constant.
  //
  // Although it's possible TryResolveTypedInst could complete in a single call
  // when all input constants are ready, a common scenario is that some inputs
  // will still be unresolved, and it'll return Invalid between phases. On the
  // second call, all previously unready constants will have been resolved, so
  // it should run to completion. As a consequence, it's important to reserve
  // all expensive logic for the second phase; this in particular includes
  // GetTypeIdForTypeConstant calls which do a hash table lookup.
  //
  // TODO: Error is returned when support is missing, but that should go away.
  auto TryResolveInst(SemIR::InstId inst_id) -> SemIR::ConstantId {
    if (inst_id.is_builtin()) {
      // Constants for builtins can be directly copied.
      return context_.constant_values().Get(inst_id);
    }

    auto inst = import_ir_.insts().Get(inst_id);
    switch (inst.kind()) {
      case SemIR::InstKind::BindAlias:
        return TryResolveTypedInst(inst.As<SemIR::BindAlias>());

      case SemIR::InstKind::ConstType:
        return TryResolveTypedInst(inst.As<SemIR::ConstType>());

      case SemIR::InstKind::FunctionDecl:
        return TryResolveTypedInst(inst.As<SemIR::FunctionDecl>());

      case SemIR::InstKind::PointerType:
        return TryResolveTypedInst(inst.As<SemIR::PointerType>());

      case SemIR::InstKind::StructType:
        return TryResolveTypedInst(inst.As<SemIR::StructType>());

      case SemIR::InstKind::TupleType:
        return TryResolveTypedInst(inst.As<SemIR::TupleType>());

      case SemIR::InstKind::BindName:
      case SemIR::InstKind::BindSymbolicName:
        // Can use TryEvalInst because the resulting constant doesn't really use
        // `inst`.
        return TryEvalInst(context_, inst_id, inst);

      case SemIR::InstKind::ClassDecl:
      case SemIR::InstKind::InterfaceDecl:
        // TODO: Not implemented.
        return SemIR::ConstantId::Error;

      default:
        context_.TODO(
            Parse::NodeId::Invalid,
            llvm::formatv("TryResolveInst on {0}", inst.kind()).str());
        return SemIR::ConstantId::Error;
    }
  }

  auto TryResolveTypedInst(SemIR::BindAlias inst) -> SemIR::ConstantId {
    auto initial_work = work_stack_.size();
    auto value_id = GetConstantId(inst.value_id);
    if (HasUnresolved(initial_work)) {
      return SemIR::ConstantId::Invalid;
    }
    return value_id;
  }

  auto TryResolveTypedInst(SemIR::ConstType inst) -> SemIR::ConstantId {
    auto initial_work = work_stack_.size();
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto inner_const_id = GetConstantId(inst.inner_id);
    if (HasUnresolved(initial_work)) {
      return SemIR::ConstantId::Invalid;
    }
    auto inner_type_id = context_.GetTypeIdForTypeConstant(inner_const_id);
    // TODO: Should ConstType have a wrapper for this similar to the others?
    return TryEvalInst(
        context_, SemIR::InstId::Invalid,
        SemIR::ConstType{SemIR::TypeId::TypeType, inner_type_id});
  }

  auto TryResolveTypedInst(SemIR::FunctionDecl inst) -> SemIR::ConstantId {
    auto initial_work = work_stack_.size();
    auto type_const_id = GetConstantId(inst.type_id);

    const auto& function = import_ir_.functions().Get(inst.function_id);
    auto return_type_const_id = SemIR::ConstantId::Invalid;
    if (function.return_type_id.is_valid()) {
      return_type_const_id = GetConstantId(function.return_type_id);
    }
    auto return_slot_const_id = SemIR::ConstantId::Invalid;
    if (function.return_slot_id.is_valid()) {
      return_slot_const_id = GetConstantId(function.return_slot_id);
    }
    llvm::SmallVector<SemIR::ConstantId> implicit_param_const_ids =
        GetParamConstantIds(function.implicit_param_refs_id);
    llvm::SmallVector<SemIR::ConstantId> param_const_ids =
        GetParamConstantIds(function.param_refs_id);

    if (HasUnresolved(initial_work)) {
      return SemIR::ConstantId::Invalid;
    }

    // Add the function declaration.
    auto function_decl =
        SemIR::FunctionDecl{context_.GetTypeIdForTypeConstant(type_const_id),
                            SemIR::FunctionId::Invalid};
    auto function_decl_id =
        context_.AddPlaceholderInst({Parse::NodeId::Invalid, function_decl});

    auto new_return_type_id =
        return_type_const_id.is_valid()
            ? context_.GetTypeIdForTypeConstant(return_type_const_id)
            : SemIR::TypeId::Invalid;
    auto new_return_slot = SemIR::InstId::Invalid;
    if (function.return_slot_id.is_valid()) {
      context_.AddInstInNoBlock({SemIR::ImportRefUsed{
          context_.GetTypeIdForTypeConstant(return_slot_const_id),
          import_ir_id_, function.return_slot_id}});
    }
    function_decl.function_id = context_.functions().Add(
        {.name_id = GetNameId(function.name_id),
         .enclosing_scope_id = NoEnclosingScopeForImports,
         .decl_id = function_decl_id,
         .implicit_param_refs_id = GetParamRefsId(
             function.implicit_param_refs_id, implicit_param_const_ids),
         .param_refs_id =
             GetParamRefsId(function.param_refs_id, param_const_ids),
         .return_type_id = new_return_type_id,
         .return_slot_id = new_return_slot});
    // Write the function ID into the FunctionDecl.
    context_.ReplaceInstBeforeConstantUse(
        function_decl_id, {Parse::NodeId::Invalid, function_decl});
    return context_.constant_values().Get(function_decl_id);
  }

  auto TryResolveTypedInst(SemIR::PointerType inst) -> SemIR::ConstantId {
    auto initial_work = work_stack_.size();
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto pointee_const_id = GetConstantId(inst.pointee_id);
    if (HasUnresolved(initial_work)) {
      return SemIR::ConstantId::Invalid;
    }

    auto pointee_type_id = context_.GetTypeIdForTypeConstant(pointee_const_id);
    return context_.types().GetConstantId(
        context_.GetPointerType(pointee_type_id));
  }

  auto TryResolveTypedInst(SemIR::StructType inst) -> SemIR::ConstantId {
    // Collect all constants first, locating unresolved ones in a single pass.
    auto initial_work = work_stack_.size();
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto orig_fields = import_ir_.inst_blocks().Get(inst.fields_id);
    llvm::SmallVector<SemIR::ConstantId> field_const_ids;
    field_const_ids.reserve(orig_fields.size());
    for (auto field_id : orig_fields) {
      auto field = import_ir_.insts().GetAs<SemIR::StructTypeField>(field_id);
      field_const_ids.push_back(GetConstantId(field.field_type_id));
    }
    if (HasUnresolved(initial_work)) {
      return SemIR::ConstantId::Invalid;
    }

    // Prepare a vector of fields for GetStructType.
    // TODO: Should we have field constants so that we can deduplicate fields
    // without creating instructions here?
    llvm::SmallVector<SemIR::InstId> fields;
    fields.reserve(orig_fields.size());
    for (auto [field_id, field_const_id] :
         llvm::zip(orig_fields, field_const_ids)) {
      auto field = import_ir_.insts().GetAs<SemIR::StructTypeField>(field_id);
      auto name_id = GetNameId(field.name_id);
      auto field_type_id = context_.GetTypeIdForTypeConstant(field_const_id);
      fields.push_back(context_.AddInstInNoBlock(
          {Parse::NodeId::Invalid,
           SemIR::StructTypeField{.name_id = name_id,
                                  .field_type_id = field_type_id}}));
    }

    return context_.types().GetConstantId(
        context_.GetStructType(context_.inst_blocks().Add(fields)));
  }

  auto TryResolveTypedInst(SemIR::TupleType inst) -> SemIR::ConstantId {
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);

    // Collect all constants first, locating unresolved ones in a single pass.
    auto initial_work = work_stack_.size();
    auto orig_elem_type_ids = import_ir_.type_blocks().Get(inst.elements_id);
    llvm::SmallVector<SemIR::ConstantId> elem_const_ids;
    elem_const_ids.reserve(orig_elem_type_ids.size());
    for (auto elem_type_id : orig_elem_type_ids) {
      elem_const_ids.push_back(GetConstantId(elem_type_id));
    }
    if (HasUnresolved(initial_work)) {
      return SemIR::ConstantId::Invalid;
    }

    // Prepare a vector of the tuple types for GetTupleType.
    llvm::SmallVector<SemIR::TypeId> elem_type_ids;
    elem_type_ids.reserve(orig_elem_type_ids.size());
    for (auto elem_const_id : elem_const_ids) {
      elem_type_ids.push_back(context_.GetTypeIdForTypeConstant(elem_const_id));
    }

    return context_.types().GetConstantId(context_.GetTupleType(elem_type_ids));
  }

  Context& context_;
  SemIR::ImportIRId import_ir_id_;
  const SemIR::File& import_ir_;
  SemIR::ConstantValueStore& import_ir_constant_values_;
  llvm::SmallVector<SemIR::InstId> work_stack_;
};

auto TryResolveImportRefUnused(Context& context, SemIR::InstId inst_id)
    -> void {
  auto inst = context.insts().Get(inst_id);
  auto import_ref = inst.TryAs<SemIR::ImportRefUnused>();
  if (!import_ref) {
    return;
  }

  const SemIR::File& import_ir = *context.import_irs().Get(import_ref->ir_id);
  auto import_inst = import_ir.insts().Get(import_ref->inst_id);

  ImportRefResolver resolver(context, import_ref->ir_id);
  auto type_id = resolver.ResolveType(import_inst.type_id());
  auto constant_id = resolver.Resolve(import_ref->inst_id);

  // TODO: Once ClassDecl/InterfaceDecl are supported (no longer return Error),
  // remove this.
  if (constant_id == SemIR::ConstantId::Error) {
    type_id = SemIR::TypeId::Error;
  }

  // Replace the ImportRefUnused instruction with an ImportRefUsed. This doesn't
  // use ReplaceInstBeforeConstantUse because it would trigger TryEvalInst, and
  // we're instead doing constant evaluation here in order to minimize recursion
  // risks.
  context.sem_ir().insts().Set(
      inst_id,
      SemIR::ImportRefUsed{type_id, import_ref->ir_id, import_ref->inst_id});

  // Store the constant for both the ImportRefUsed and imported instruction.
  context.constant_values().Set(inst_id, constant_id);
}

}  // namespace Carbon::Check

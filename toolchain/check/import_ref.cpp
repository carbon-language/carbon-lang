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
  explicit ImportRefResolver(
      Context& context, const SemIR::File& import_ir,
      SemIR::ConstantValueStore& import_ir_constant_values)
      : context_(context),
        import_ir_(import_ir),
        import_ir_constant_values_(import_ir_constant_values) {}

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
  // Returns the ConstantId for an instruction, or adds it to the stack and
  // returns Invalid if the ConstantId is not ready.
  auto GetConstantId(SemIR::TypeId type_id) -> SemIR::ConstantId {
    auto inst_id = import_ir_.types().GetInstId(type_id);
    auto const_id = import_ir_constant_values_.Get(inst_id);
    if (!const_id.is_valid()) {
      work_stack_.push_back(inst_id);
    }
    return const_id;
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
      case SemIR::InstKind::ConstType:
        return TryResolveTypedInst(inst.As<SemIR::ConstType>());

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

      case SemIR::InstKind::FunctionDecl:
        // TODO: Allowed to work for testing, but not really implemented.
        return SemIR::ConstantId::NotConstant;

      default:
        context_.TODO(
            Parse::NodeId::Invalid,
            llvm::formatv("TryResolveInst on {0}", inst.kind()).str());
        return SemIR::ConstantId::Error;
    }
  }

  auto TryResolveTypedInst(SemIR::ConstType inst) -> SemIR::ConstantId {
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto inner_const_id = GetConstantId(inst.inner_id);
    if (!inner_const_id.is_valid()) {
      return SemIR::ConstantId::Invalid;
    }
    auto inner_type_id = context_.GetTypeIdForTypeConstant(inner_const_id);
    // TODO: Should ConstType have a wrapper for this similar to the others?
    return TryEvalInst(
        context_, SemIR::InstId::Invalid,
        SemIR::ConstType{SemIR::TypeId::TypeType, inner_type_id});
  }

  auto TryResolveTypedInst(SemIR::PointerType inst) -> SemIR::ConstantId {
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto pointee_const_id = GetConstantId(inst.pointee_id);
    if (!pointee_const_id.is_valid()) {
      return SemIR::ConstantId::Invalid;
    }
    auto pointee_type_id = context_.GetTypeIdForTypeConstant(pointee_const_id);
    return context_.types().GetConstantId(
        context_.GetPointerType(pointee_type_id));
  }

  auto TryResolveTypedInst(SemIR::StructType inst) -> SemIR::ConstantId {
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);

    // Collect all constants first, locating unresolved ones in a single pass.
    bool has_unresolved = false;
    auto orig_fields = import_ir_.inst_blocks().Get(inst.fields_id);
    llvm::SmallVector<SemIR::ConstantId> field_const_ids;
    field_const_ids.reserve(orig_fields.size());
    for (auto field_id : orig_fields) {
      auto field = import_ir_.insts().GetAs<SemIR::StructTypeField>(field_id);
      auto field_const_id = GetConstantId(field.field_type_id);
      if (field_const_id.is_valid()) {
        field_const_ids.push_back(field_const_id);
      } else {
        has_unresolved = true;
      }
    }
    if (has_unresolved) {
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
      auto name_str = import_ir_.names().GetAsStringIfIdentifier(field.name_id);
      auto name_id = name_str ? SemIR::NameId::ForIdentifier(
                                    context_.identifiers().Add(*name_str))
                              : field.name_id;
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
    bool has_unresolved = false;
    auto orig_elem_type_ids = import_ir_.type_blocks().Get(inst.elements_id);
    llvm::SmallVector<SemIR::ConstantId> elem_const_ids;
    elem_const_ids.reserve(orig_elem_type_ids.size());
    for (auto elem_type_id : orig_elem_type_ids) {
      auto elem_const_id = GetConstantId(elem_type_id);
      if (elem_const_id.is_valid()) {
        elem_const_ids.push_back(elem_const_id);
      } else {
        has_unresolved = true;
      }
    }
    if (has_unresolved) {
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
  auto& import_ir_constant_values =
      context.import_ir_constant_values()[import_ref->ir_id.index];
  auto import_inst = import_ir.insts().Get(import_ref->inst_id);

  ImportRefResolver resolver(context, import_ir, import_ir_constant_values);
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

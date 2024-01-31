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
  // instruction has unresolved inner references, it will add them to the todo
  // list for inner evaluation and reattempt outer evaluation after.
  auto Resolve(SemIR::InstId inst_id) -> SemIR::ConstantId {
    todo_.push_back(inst_id);
    while (!todo_.empty()) {
      auto inst_id = todo_.back();
      CARBON_CHECK(inst_id.is_valid());

      // Double-check that the constant still doesn't have a calculated value.
      // This should typically be checked before adding it, but a given constant
      // may be added multiple times before its constant is evaluated.
      if (auto current_const_id = import_ir_constant_values_.Get(inst_id);
          current_const_id.is_valid()) {
        todo_.pop_back();
      } else if (auto new_const_id = TryResolveInst(inst_id);
                 new_const_id.is_valid()) {
        import_ir_constant_values_.Set(inst_id, new_const_id);
        todo_.pop_back();
      }
    }
    auto constant_id = import_ir_constant_values_.Get(inst_id);
    CARBON_CHECK(constant_id.is_valid());
    return constant_id;
  }

 private:
  // Returns the ConstantId for an instruction, or adds it to the todo list and
  // returns Invalid if the ConstantId is not ready.
  auto GetConstantId(SemIR::TypeId type_id) -> SemIR::ConstantId {
    auto inst_id = import_ir_.types().GetInstId(type_id);
    auto const_id = import_ir_constant_values_.Get(inst_id);
    if (!const_id.is_valid()) {
      todo_.push_back(inst_id);
    }
    return const_id;
  }

  // Tries to resolve the InstId, returning a constant when ready, or Invalid if
  // more has been added to the todo list. A similar API is followed for all
  // following TryResolveTypedInst helper functions.
  //
  // Resolving will often call GetTypeIdForTypeConstant. It involves a hash
  // table lookup, so only call it when ready to provide a constant.
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

      default:
        context_.TODO(
            Parse::NodeId::Invalid,
            llvm::formatv("TryResolveNextInst on {0}", inst.kind()).str());
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

    // First verify that all used types are ready.
    bool added_todos = false;
    auto orig_fields = import_ir_.inst_blocks().Get(inst.fields_id);
    llvm::SmallVector<SemIR::ConstantId> field_const_ids;
    field_const_ids.reserve(orig_fields.size());
    for (auto field_id : orig_fields) {
      auto field = import_ir_.insts().GetAs<SemIR::StructTypeField>(field_id);
      auto field_const_id = GetConstantId(field.field_type_id);
      if (field_const_id.is_valid()) {
        field_const_ids.push_back(field_const_id);
      } else {
        added_todos = true;
      }
    }
    if (added_todos) {
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

    // First verify that all used types are ready.
    bool added_todos = false;
    auto orig_el_type_ids = import_ir_.type_blocks().Get(inst.elements_id);
    llvm::SmallVector<SemIR::ConstantId> el_const_ids;
    el_const_ids.reserve(orig_el_type_ids.size());
    for (auto el_type_id : orig_el_type_ids) {
      auto el_const_id = GetConstantId(el_type_id);
      if (el_const_id.is_valid()) {
        el_const_ids.push_back(el_const_id);
      } else {
        added_todos = true;
      }
    }
    if (added_todos) {
      return SemIR::ConstantId::Invalid;
    }

    // Prepare a vector of the tuple types for GetTupleType.
    llvm::SmallVector<SemIR::TypeId> el_type_ids;
    el_type_ids.reserve(orig_el_type_ids.size());
    for (auto el_const_id : el_const_ids) {
      el_type_ids.push_back(context_.GetTypeIdForTypeConstant(el_const_id));
    }

    return context_.types().GetConstantId(context_.GetTupleType(el_type_ids));
  }

  Context& context_;
  const SemIR::File& import_ir_;
  SemIR::ConstantValueStore& import_ir_constant_values_;
  llvm::SmallVector<SemIR::InstId> todo_;
};

auto TryResolveImportRefUnused(Context& context, SemIR::InstId inst_id)
    -> void {
  auto inst = context.insts().Get(inst_id);
  auto unused_inst = inst.TryAs<SemIR::ImportRefUnused>();
  if (!unused_inst) {
    return;
  }

  const SemIR::File& import_ir = *context.import_irs().Get(unused_inst->ir_id);
  auto import_inst = import_ir.insts().Get(unused_inst->inst_id);

  // TODO: Types need to be specifically addressed here to prevent crashes in
  // constant evaluation while support is incomplete. Functions are also
  // incomplete, but are allowed to fail differently because they aren't a type.
  // The partial function support is useful for some namespace validation.
  if (import_inst.Is<SemIR::ClassDecl>() ||
      import_inst.Is<SemIR::InterfaceDecl>()) {
    context.TODO(
        Parse::NodeId::Invalid,
        llvm::formatv("TryResolveImportRefUnused on {0}", import_inst.kind())
            .str());
    context.ReplaceInstBeforeConstantUse(
        inst_id, {SemIR::ImportRefUsed{SemIR::TypeId::Error, unused_inst->ir_id,
                                       unused_inst->inst_id}});
    return;
  }

  // If the type ID isn't a normal value, forward it directly.
  if (!import_inst.type_id().is_valid()) {
    context.ReplaceInstBeforeConstantUse(
        inst_id,
        {SemIR::ImportRefUsed{import_inst.type_id(), unused_inst->ir_id,
                              unused_inst->inst_id}});
    return;
  }

  auto import_type_inst_id = import_ir.types().GetInstId(import_inst.type_id());
  CARBON_CHECK(import_type_inst_id.is_valid());

  auto type_id = SemIR::TypeId::Invalid;
  if (import_type_inst_id.is_builtin()) {
    // Builtins don't require constant resolution; we can use them directly.
    type_id = context.GetBuiltinType(import_type_inst_id.builtin_kind());
  } else {
    ImportRefResolver resolver(
        context, import_ir,
        context.import_ir_constant_values()[unused_inst->ir_id.index]);
    type_id =
        context.GetTypeIdForTypeConstant(resolver.Resolve(import_type_inst_id));
  }
  // TODO: Add breadcrumbs for lowering.
  context.ReplaceInstBeforeConstantUse(
      inst_id, {SemIR::ImportRefUsed{type_id, unused_inst->ir_id,
                                     unused_inst->inst_id}});
}

}  // namespace Carbon::Check

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/convert.h"

#include <optional>
#include <string>
#include <utility>

#include "common/check.h"
#include "common/map.h"
#include "llvm/ADT/STLExtras.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/action.h"
#include "toolchain/check/context.h"
#include "toolchain/check/control_flow.h"
#include "toolchain/check/core_identifier.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/impl_lookup.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/member_access.h"
#include "toolchain/check/operator.h"
#include "toolchain/check/pattern_match.h"
#include "toolchain/check/pending_block.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/diagnostics/emitter.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/sem_ir/copy_on_write_block.h"
#include "toolchain/sem_ir/expr_info.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/type.h"
#include "toolchain/sem_ir/type_info.h"
#include "toolchain/sem_ir/typed_insts.h"

// TODO: This contains a lot of recursion. Consider removing it in order to
// prevent accidents.
// NOLINTBEGIN(misc-no-recursion)

namespace Carbon::Check {

// If the initializing expression `init_id` has a storage argument that refers
// to a temporary, overwrites it with the inst at `target.storage_id`, and
// returns the ID that should now be used to refer to `init_id`'s storage. Has
// no effect and returns `target.storage_id` unchanged if `target.storage_id` is
// None, if `init_id` doesn't have a storage arg, or if the storage argument
// doesn't point to a temporary. In the latter case, we assume it was set
// correctly when the instruction was created.
static auto OverwriteTemporaryStorageArg(SemIR::File& sem_ir,
                                         SemIR::InstId init_id,
                                         const ConversionTarget& target)
    -> SemIR::InstId {
  CARBON_CHECK(target.is_initializer());
  if (!target.storage_id.has_value()) {
    return SemIR::InstId::None;
  }
  auto storage_arg_id = FindStorageArgForInitializer(sem_ir, init_id);
  if (!storage_arg_id.has_value() || storage_arg_id == target.storage_id ||
      !sem_ir.insts().Is<SemIR::TemporaryStorage>(storage_arg_id)) {
    return target.storage_id;
  }
  // Replace the temporary in the storage argument with a reference to our
  // target.
  return target.storage_access_block->MergeReplacing(storage_arg_id,
                                                     target.storage_id);
}

// Materializes and returns a temporary initialized from the initializer
// `init_id`. If `init_id` has a storage arg, it must be a `TemporaryStorage`;
// if not, this function allocates one for it.
static auto MaterializeTemporary(Context& context, SemIR::InstId init_id)
    -> SemIR::InstId {
  auto& sem_ir = context.sem_ir();
  auto category = SemIR::GetExprCategory(sem_ir, init_id);
  CARBON_CHECK(SemIR::IsInitializerCategory(category));
  auto init = sem_ir.insts().Get(init_id);
  auto storage_id = FindStorageArgForInitializer(sem_ir, init_id);
  if (!storage_id.has_value()) {
    CARBON_CHECK(category == SemIR::ExprCategory::ReprInitializing);
    // The initializer has no storage arg, but we want to produce an ephemeral
    // reference, so we need to allocate temporary storage.
    storage_id = AddInst<SemIR::TemporaryStorage>(
        context, SemIR::LocId(init_id), {.type_id = init.type_id()});
  }

  CARBON_CHECK(
      sem_ir.insts().Get(storage_id).kind() == SemIR::TemporaryStorage::Kind,
      "Storage arg for initializer does not contain a temporary; "
      "initialized multiple times? Have {0}",
      sem_ir.insts().Get(storage_id));
  return AddInstWithCleanup<SemIR::Temporary>(context, SemIR::LocId(init_id),
                                              {.type_id = init.type_id(),
                                               .storage_id = storage_id,
                                               .init_id = init_id});
}

// Discards the initializer `init_id`. If `init_id` intrinsically writes to
// memory, this materializes a temporary for it and starts its lifetime.
//
// TODO: We should probably start its lifetime unconditionally, because
// types with by-copy representations can still have nontrivial destructors.
static auto DiscardInitializer(Context& context, SemIR::InstId init_id)
    -> void {
  auto& sem_ir = context.sem_ir();
  auto storage_id = FindStorageArgForInitializer(sem_ir, init_id);
  if (!storage_id.has_value()) {
    CARBON_CHECK(SemIR::GetExprCategory(sem_ir, init_id) ==
                 SemIR::ExprCategory::ReprInitializing);
    return;
  }

  // init_id writes to temporary storage, so we need to materialize a temporary
  // for it.
  MaterializeTemporary(context, init_id);
}

// If `expr_id` is an initializer, materializes it and returns the resulting
// ephemeral reference expression. Otherwise, returns `expr_id`.
static auto MaterializeIfInitializer(Context& context, SemIR::InstId expr_id)
    -> SemIR::InstId {
  if (SemIR::IsInitializerCategory(
          SemIR::GetExprCategory(context.sem_ir(), expr_id))) {
    return MaterializeTemporary(context, expr_id);
  } else {
    return expr_id;
  }
}

// Helper to allow `MakeElementAccessInst` to call `AddInst` with either a
// `PendingBlock` or `Context` (defined in `inst.h`).
template <typename AccessInstT>
static auto AddInst(PendingBlock& block, SemIR::LocId loc_id, AccessInstT inst)
    -> SemIR::InstId {
  return block.AddInst<AccessInstT>(loc_id, inst);
}

// Creates and adds an instruction to perform element access into an aggregate.
template <typename AccessInstT, typename InstBlockT>
static auto MakeElementAccessInst(Context& context, SemIR::LocId loc_id,
                                  SemIR::InstId aggregate_id,
                                  SemIR::TypeId elem_type_id, InstBlockT& block,
                                  size_t i) -> SemIR::InstId {
  if (!aggregate_id.has_value()) {
    return SemIR::InstId::None;
  }
  if constexpr (std::is_same_v<AccessInstT, SemIR::ArrayIndex>) {
    // TODO: Add a new instruction kind for indexing an array at a constant
    // index so that we don't need an integer literal instruction here, and
    // remove this special case.
    auto index_id = block.template AddInst<SemIR::IntValue>(
        loc_id, {.type_id = GetSingletonType(context,
                                             SemIR::IntLiteralType::TypeInstId),
                 .int_id = context.ints().Add(static_cast<int64_t>(i))});
    return AddInst<AccessInstT>(block, loc_id,
                                {elem_type_id, aggregate_id, index_id});
  } else {
    return AddInst<AccessInstT>(
        block, loc_id, {elem_type_id, aggregate_id, SemIR::ElementIndex(i)});
  }
}

// Get the conversion target kind to use when initializing an element of an
// aggregate.
static auto GetAggregateElementConversionTargetKind(SemIR::File& sem_ir,
                                                    ConversionTarget target)
    -> ConversionTarget::Kind {
  // If we're forming an initializer, then we want an initializer for each
  // element.
  if (target.is_initializer()) {
    // Perform a final destination store if we're performing an in-place
    // initialization.
    auto init_repr = SemIR::InitRepr::ForType(sem_ir, target.type_id);
    CARBON_CHECK(init_repr.kind != SemIR::InitRepr::Dependent,
                 "Aggregate should not have dependent init kind");
    if (init_repr.kind == SemIR::InitRepr::InPlace) {
      return ConversionTarget::InPlaceInitializing;
    }
    return ConversionTarget::Initializing;
  }

  // Otherwise, we want a value representation for each element.
  return ConversionTarget::Value;
}

// Converts an element of one aggregate so that it can be used as an element of
// another aggregate.
//
// For the source: `src_id` is the source aggregate, `src_elem_type` is the
// element type, `src_field_index` is the index, and `SourceAccessInstT` is the
// kind of instruction used to access the source element.
//
// For the target: `kind` is the kind of conversion or initialization,
// `target_elem_type` is the element type. For initialization, `target_id` is
// the destination, `target_block` is a pending block for target location
// calculations that will be spliced as the return slot of the initializer if
// necessary, `target_field_index` is the index, and `TargetAccessInstT` is the
// kind of instruction used to access the destination element.
template <typename SourceAccessInstT, typename TargetAccessInstT>
static auto ConvertAggregateElement(
    Context& context, SemIR::LocId loc_id, SemIR::InstId src_id,
    SemIR::TypeInstId src_elem_type_inst,
    llvm::ArrayRef<SemIR::InstId> src_literal_elems,
    ConversionTarget::Kind kind, SemIR::InstId target_id,
    SemIR::TypeInstId target_elem_type_inst, PendingBlock* target_block,
    size_t src_field_index, size_t target_field_index) -> SemIR::InstId {
  auto src_elem_type =
      context.types().GetTypeIdForTypeInstId(src_elem_type_inst);
  auto target_elem_type =
      context.types().GetTypeIdForTypeInstId(target_elem_type_inst);

  // Compute the location of the source element. This goes into the current code
  // block, not into the target block.
  // TODO: Ideally we would discard this instruction if it's unused.
  auto src_elem_id = !src_literal_elems.empty()
                         ? src_literal_elems[src_field_index]
                         : MakeElementAccessInst<SourceAccessInstT>(
                               context, loc_id, src_id, src_elem_type, context,
                               src_field_index);

  // If we're performing a conversion rather than an initialization, we won't
  // have or need a target.
  ConversionTarget target = {.kind = kind, .type_id = target_elem_type};
  if (!target.is_initializer()) {
    return Convert(context, loc_id, src_elem_id, target);
  }

  // Compute the location of the target element and initialize it.
  PendingBlock::DiscardUnusedInstsScope scope(target_block);
  target.storage_access_block = target_block;
  target.storage_id = MakeElementAccessInst<TargetAccessInstT>(
      context, loc_id, target_id, target_elem_type, *target_block,
      target_field_index);
  return Convert(context, loc_id, src_elem_id, target);
}

// Performs a conversion from a tuple to an array type. This function only
// converts the type, and does not perform a final conversion to the requested
// expression category.
static auto ConvertTupleToArray(Context& context, SemIR::TupleType tuple_type,
                                SemIR::ArrayType array_type,
                                SemIR::InstId value_id, ConversionTarget target)
    -> SemIR::InstId {
  auto& sem_ir = context.sem_ir();
  auto tuple_elem_types = sem_ir.inst_blocks().Get(tuple_type.type_elements_id);

  auto value = sem_ir.insts().Get(value_id);
  SemIR::LocId value_loc_id(value_id);

  // If we're initializing from a tuple literal, we will use its elements
  // directly. Otherwise, materialize a temporary if needed and index into the
  // result.
  llvm::ArrayRef<SemIR::InstId> literal_elems;
  if (auto tuple_literal = value.TryAs<SemIR::TupleLiteral>()) {
    literal_elems = sem_ir.inst_blocks().Get(tuple_literal->elements_id);
  } else {
    value_id = MaterializeIfInitializer(context, value_id);
  }

  // Check that the tuple is the right size.
  std::optional<uint64_t> array_bound =
      sem_ir.GetZExtIntValue(array_type.bound_id);
  if (!array_bound) {
    // TODO: Should this fall back to using `ImplicitAs`?
    if (target.diagnose) {
      CARBON_DIAGNOSTIC(ArrayInitDependentBound, Error,
                        "cannot initialize array with dependent bound from a "
                        "list of initializers");
      context.emitter().Emit(value_loc_id, ArrayInitDependentBound);
    }
    return SemIR::ErrorInst::InstId;
  }
  if (tuple_elem_types.size() != array_bound) {
    if (target.diagnose) {
      CARBON_DIAGNOSTIC(ArrayInitFromLiteralArgCountMismatch, Error,
                        "cannot initialize array of {0} element{0:s} from {1} "
                        "initializer{1:s}",
                        Diagnostics::IntAsSelect, Diagnostics::IntAsSelect);
      CARBON_DIAGNOSTIC(
          ArrayInitFromExprArgCountMismatch, Error,
          "cannot initialize array of {0} element{0:s} from tuple "
          "with {1} element{1:s}",
          Diagnostics::IntAsSelect, Diagnostics::IntAsSelect);
      context.emitter().Emit(value_loc_id,
                             literal_elems.empty()
                                 ? ArrayInitFromExprArgCountMismatch
                                 : ArrayInitFromLiteralArgCountMismatch,
                             *array_bound, tuple_elem_types.size());
    }
    return SemIR::ErrorInst::InstId;
  }

  PendingBlock target_block_storage(&context);
  PendingBlock* target_block = target.storage_access_block
                                   ? target.storage_access_block
                                   : &target_block_storage;

  // Arrays are always initialized in-place. Allocate a temporary as the
  // destination for the array initialization if we weren't given one.
  SemIR::InstId return_slot_arg_id = target.storage_id;
  if (!target.storage_id.has_value()) {
    return_slot_arg_id = target_block->AddInst<SemIR::TemporaryStorage>(
        value_loc_id, {.type_id = target.type_id});
  }

  // Initialize each element of the array from the corresponding element of the
  // tuple.
  // TODO: Annotate diagnostics coming from here with the array element index,
  // if initializing from a tuple literal.
  llvm::SmallVector<SemIR::InstId> inits;
  inits.reserve(*array_bound + 1);
  for (auto [i, src_type_inst_id] : llvm::enumerate(
           context.types().GetBlockAsTypeInstIds(tuple_elem_types))) {
    // TODO: This call recurses back into conversion. Switch to an iterative
    // approach.
    auto init_id =
        ConvertAggregateElement<SemIR::TupleAccess, SemIR::ArrayIndex>(
            context, value_loc_id, value_id, src_type_inst_id, literal_elems,
            ConversionTarget::InPlaceInitializing, return_slot_arg_id,
            array_type.element_type_inst_id, target_block, i, i);
    if (init_id == SemIR::ErrorInst::InstId) {
      return SemIR::ErrorInst::InstId;
    }
    inits.push_back(init_id);
  }

  // Flush the temporary here if we didn't insert it earlier, so we can add a
  // reference to the return slot.
  target_block->InsertHere();
  return AddInst<SemIR::ArrayInit>(context, value_loc_id,
                                   {.type_id = target.type_id,
                                    .inits_id = sem_ir.inst_blocks().Add(inits),
                                    .dest_id = return_slot_arg_id});
}

// Performs a conversion from a tuple to a tuple type. This function only
// converts the type, and does not perform a final conversion to the requested
// expression category.
static auto ConvertTupleToTuple(Context& context, SemIR::TupleType src_type,
                                SemIR::TupleType dest_type,
                                SemIR::InstId value_id, ConversionTarget target)
    -> SemIR::InstId {
  auto& sem_ir = context.sem_ir();
  auto src_elem_types = sem_ir.inst_blocks().Get(src_type.type_elements_id);
  auto dest_elem_types = sem_ir.inst_blocks().Get(dest_type.type_elements_id);

  auto value = sem_ir.insts().Get(value_id);
  SemIR::LocId value_loc_id(value_id);

  // If we're initializing from a tuple literal, we will use its elements
  // directly. Otherwise, materialize a temporary if needed and index into the
  // result.
  llvm::ArrayRef<SemIR::InstId> literal_elems;
  auto literal_elems_id = SemIR::InstBlockId::None;
  if (auto tuple_literal = value.TryAs<SemIR::TupleLiteral>()) {
    literal_elems_id = tuple_literal->elements_id;
    literal_elems = sem_ir.inst_blocks().Get(literal_elems_id);
  } else {
    value_id = MaterializeIfInitializer(context, value_id);
  }

  // Check that the tuples are the same size.
  if (src_elem_types.size() != dest_elem_types.size()) {
    if (target.diagnose) {
      CARBON_DIAGNOSTIC(
          TupleInitElementCountMismatch, Error,
          "cannot initialize tuple of {0} element{0:s} from tuple "
          "with {1} element{1:s}",
          Diagnostics::IntAsSelect, Diagnostics::IntAsSelect);
      context.emitter().Emit(value_loc_id, TupleInitElementCountMismatch,
                             dest_elem_types.size(), src_elem_types.size());
    }
    return SemIR::ErrorInst::InstId;
  }

  ConversionTarget::Kind inner_kind =
      GetAggregateElementConversionTargetKind(sem_ir, target);

  // Initialize each element of the destination from the corresponding element
  // of the source.
  // TODO: Annotate diagnostics coming from here with the element index.
  auto new_block =
      literal_elems_id.has_value()
          ? SemIR::CopyOnWriteInstBlock(&sem_ir, literal_elems_id)
          : SemIR::CopyOnWriteInstBlock(
                &sem_ir, SemIR::CopyOnWriteInstBlock::UninitializedBlock{
                             src_elem_types.size()});
  for (auto [i, src_type_inst_id, dest_type_inst_id] : llvm::enumerate(
           context.types().GetBlockAsTypeInstIds(src_elem_types),
           context.types().GetBlockAsTypeInstIds(dest_elem_types))) {
    // TODO: This call recurses back into conversion. Switch to an iterative
    // approach.
    auto init_id =
        ConvertAggregateElement<SemIR::TupleAccess, SemIR::TupleAccess>(
            context, value_loc_id, value_id, src_type_inst_id, literal_elems,
            inner_kind, target.storage_id, dest_type_inst_id,
            target.storage_access_block, i, i);
    if (init_id == SemIR::ErrorInst::InstId) {
      return SemIR::ErrorInst::InstId;
    }
    new_block.Set(i, init_id);
  }

  if (target.is_initializer()) {
    target.storage_access_block->InsertHere();
    return AddInst<SemIR::TupleInit>(context, value_loc_id,
                                     {.type_id = target.type_id,
                                      .elements_id = new_block.id(),
                                      .dest_id = target.storage_id});
  } else {
    return AddInst<SemIR::TupleValue>(
        context, value_loc_id,
        {.type_id = target.type_id, .elements_id = new_block.id()});
  }
}

// Converts a tuple of elements that are convertible to `type` into a `type`
// that is a tuple of types.
static auto ConvertTupleToType(Context& context, SemIR::LocId loc_id,
                               SemIR::InstId value_id,
                               SemIR::TypeId value_type_id,
                               ConversionTarget target) -> SemIR::TypeInstId {
  auto value_const_id = context.constant_values().Get(value_id);
  if (!value_const_id.is_constant()) {
    // Types are constants. The input value must have a constant value to
    // convert.
    return SemIR::TypeInstId::None;
  }

  llvm::SmallVector<SemIR::InstId> type_inst_ids;

  if (auto tuple_value =
          context.constant_values().TryGetInstAs<SemIR::TupleValue>(
              value_const_id)) {
    for (auto tuple_inst_id :
         context.inst_blocks().Get(tuple_value->elements_id)) {
      // TODO: This call recurses back into conversion. Switch to an
      // iterative approach.
      type_inst_ids.push_back(
          ExprAsType(context, loc_id, tuple_inst_id, target.diagnose).inst_id);
    }
  } else {
    // A value of type TupleType that isn't a TupleValue must be a symbolic
    // binding.
    CARBON_CHECK(context.constant_values().InstIs<SemIR::SymbolicBinding>(
        value_const_id));
    // Form a TupleAccess for each element in the symbolic value, which is then
    // converted to a `type` or diagnosed as an error.
    auto tuple_type = context.types().GetAs<SemIR::TupleType>(value_type_id);
    auto type_elements = context.types().GetBlockAsTypeIds(
        context.inst_blocks().Get(tuple_type.type_elements_id));
    for (auto [i, type_id] : llvm::enumerate(type_elements)) {
      auto access_inst_id =
          GetOrAddInst<SemIR::TupleAccess>(context, loc_id,
                                           {.type_id = type_id,
                                            .tuple_id = value_id,
                                            .index = SemIR::ElementIndex(i)});
      // TODO: This call recurses back into conversion. Switch to an
      // iterative approach.
      type_inst_ids.push_back(
          ExprAsType(context, loc_id, access_inst_id, target.diagnose).inst_id);
    }
  }

  // TODO: Should we add this as an instruction? It will contain
  // references to local InstIds.
  auto tuple_type_id = GetTupleType(context, type_inst_ids);
  return context.types().GetTypeInstId(tuple_type_id);
}

// Create a reference to the vtable pointer for a class. Returns None if the
// class has no vptr.
static auto CreateVtablePtrRef(Context& context, SemIR::LocId loc_id,
                               SemIR::ClassType vtable_class_type)
    -> SemIR::InstId {
  auto vtable_decl_id =
      context.classes().Get(vtable_class_type.class_id).vtable_decl_id;
  if (!vtable_decl_id.has_value()) {
    return SemIR::InstId::None;
  }

  LoadImportRef(context, vtable_decl_id);
  auto canonical_vtable_decl_id =
      context.constant_values().GetConstantInstId(vtable_decl_id);
  return AddInst<SemIR::VtablePtr>(
      context, loc_id,
      {.type_id = GetPointerType(context, SemIR::VtableType::TypeInstId),
       .vtable_id = context.insts()
                        .GetAs<SemIR::VtableDecl>(canonical_vtable_decl_id)
                        .vtable_id,
       .specific_id = vtable_class_type.specific_id});
}

// Returns whether the given expression performs in-place initialization (or is
// invalid). The category can be passed if known, otherwise it will be computed.
static auto IsInPlaceInitializing(Context& context, SemIR::InstId result_id,
                                  SemIR::ExprCategory category) {
  return category == SemIR::ExprCategory::InPlaceInitializing ||
         (category == SemIR::ExprCategory::ReprInitializing &&
          SemIR::InitRepr::ForType(context.sem_ir(),
                                   context.insts().Get(result_id).type_id())
                  .kind == SemIR::InitRepr::InPlace) ||
         category == SemIR::ExprCategory::Error;
}
static auto IsInPlaceInitializing(Context& context, SemIR::InstId result_id) {
  auto category = SemIR::GetExprCategory(context.sem_ir(), result_id);
  return IsInPlaceInitializing(context, result_id, category);
}

// Returns the index of the vptr field in the given struct type fields, or
// None if there is no vptr field.
static auto GetVptrFieldIndex(llvm::ArrayRef<SemIR::StructTypeField> fields)
    -> SemIR::ElementIndex {
  // If the type introduces a vptr, it will always be the first field.
  bool has_vptr =
      !fields.empty() && fields.front().name_id == SemIR::NameId::Vptr;
  return has_vptr ? SemIR::ElementIndex(0) : SemIR::ElementIndex::None;
}

// Builds a member access expression naming the vptr field of the given class
// object. This is analogous to what `PerformMemberAccess` for `NameId::Vptr`
// would return if the vptr could be found by name lookup.
static auto PerformVptrAccess(Context& context, SemIR::LocId loc_id,
                              SemIR::InstId class_ref_id) -> SemIR::InstId {
  auto class_type_id = context.insts().Get(class_ref_id).type_id();
  while (class_ref_id.has_value()) {
    // The type of `ref_id` must be a class type.
    if (class_type_id == SemIR::ErrorInst::TypeId) {
      return SemIR::ErrorInst::InstId;
    }
    auto class_type = context.types().GetAs<SemIR::ClassType>(class_type_id);
    auto& class_info = context.classes().Get(class_type.class_id);

    // Get the object representation.
    auto object_repr_id =
        class_info.GetObjectRepr(context.sem_ir(), class_type.specific_id);
    if (object_repr_id == SemIR::ErrorInst::TypeId) {
      return SemIR::ErrorInst::InstId;
    }
    if (context.types().Is<SemIR::CustomLayoutType>(object_repr_id)) {
      context.TODO(loc_id, "accessing vptr of custom layout class");
      return SemIR::ErrorInst::InstId;
    }

    // Check to see if this class introduces the vptr.
    auto repr_struct_type =
        context.types().GetAs<SemIR::StructType>(object_repr_id);
    auto repr_fields =
        context.struct_type_fields().Get(repr_struct_type.fields_id);
    if (auto vptr_field_index = GetVptrFieldIndex(repr_fields);
        vptr_field_index.has_value()) {
      return AddInst<SemIR::ClassElementAccess>(
          context, loc_id,
          {.type_id = context.types().GetTypeIdForTypeInstId(
               repr_fields[vptr_field_index.index].type_inst_id),
           .base_id = class_ref_id,
           .index = vptr_field_index});
    }

    // Otherwise, step through to the base class and try again.
    CARBON_CHECK(class_info.base_id.has_value(),
                 "Could not find vptr for dynamic class");
    auto base_decl = context.insts().GetAs<SemIR::BaseDecl>(class_info.base_id);
    class_type_id = context.types().GetTypeIdForTypeInstId(
        repr_fields[base_decl.index.index].type_inst_id);
    class_ref_id =
        AddInst<SemIR::ClassElementAccess>(context, loc_id,
                                           {.type_id = class_type_id,
                                            .base_id = class_ref_id,
                                            .index = base_decl.index});
  }
  return class_ref_id;
}

// Converts an initializer for a type `partial T` to an initializer for `T` by
// initializing the vptr if necessary.
static auto ConvertPartialInitializerToNonPartial(
    Context& context, ConversionTarget target,
    SemIR::ClassType vtable_class_type, SemIR::InstId result_id)
    -> SemIR::InstId {
  auto loc_id = SemIR::LocId(result_id);
  auto vptr_id = CreateVtablePtrRef(context, loc_id, vtable_class_type);
  if (!vptr_id.has_value()) {
    // No vtable pointer in this class, nothing to do.
    return result_id;
  }

  CARBON_CHECK(
      IsInPlaceInitializing(context, result_id),
      "Type with vptr should have in-place initializing representation");

  target.storage_access_block->InsertHere();
  auto dest_id = PerformVptrAccess(context, loc_id, target.storage_id);
  auto vptr_init_id = AddInst<SemIR::InPlaceInit>(
      context, loc_id,
      {.type_id = context.insts().Get(dest_id).type_id(),
       .src_id = vptr_id,
       .dest_id = dest_id});
  return AddInst<SemIR::UpdateInit>(context, loc_id,
                                    {.type_id = target.type_id,
                                     .base_init_id = result_id,
                                     .update_init_id = vptr_init_id});
}

// Common implementation for ConvertStructToStruct and ConvertStructToClass.
template <typename TargetAccessInstT>
static auto ConvertStructToStructOrClass(
    Context& context, SemIR::StructType src_type, SemIR::StructType dest_type,
    SemIR::InstId value_id, ConversionTarget target,
    SemIR::ClassType* vtable_class_type = nullptr) -> SemIR::InstId {
  static_assert(std::is_same_v<SemIR::ClassElementAccess, TargetAccessInstT> ||
                std::is_same_v<SemIR::StructAccess, TargetAccessInstT>);
  constexpr bool ToClass =
      std::is_same_v<SemIR::ClassElementAccess, TargetAccessInstT>;

  auto& sem_ir = context.sem_ir();
  auto src_elem_fields = sem_ir.struct_type_fields().Get(src_type.fields_id);
  auto dest_elem_fields = sem_ir.struct_type_fields().Get(dest_type.fields_id);
  auto dest_vptr_index = GetVptrFieldIndex(dest_elem_fields);
  auto dest_elem_fields_size =
      dest_elem_fields.size() - (dest_vptr_index.has_value() ? 1 : 0);

  auto value = sem_ir.insts().Get(value_id);
  SemIR::LocId value_loc_id(value_id);

  // If we're initializing from a struct literal, we will use its elements
  // directly. Otherwise, materialize a temporary if needed and index into the
  // result.
  llvm::ArrayRef<SemIR::InstId> literal_elems;
  auto literal_elems_id = SemIR::InstBlockId::None;
  if (auto struct_literal = value.TryAs<SemIR::StructLiteral>()) {
    literal_elems_id = struct_literal->elements_id;
    literal_elems = sem_ir.inst_blocks().Get(literal_elems_id);
  } else {
    value_id = MaterializeIfInitializer(context, value_id);
  }

  // Check that the structs are the same size.
  // TODO: If not, include the name of the first source field that doesn't
  // exist in the destination or vice versa in the diagnostic.
  if (src_elem_fields.size() != dest_elem_fields_size) {
    if (target.diagnose) {
      CARBON_DIAGNOSTIC(
          StructInitElementCountMismatch, Error,
          "cannot initialize {0:class|struct} with {1} field{1:s} from struct "
          "with {2} field{2:s}",
          Diagnostics::BoolAsSelect, Diagnostics::IntAsSelect,
          Diagnostics::IntAsSelect);
      context.emitter().Emit(value_loc_id, StructInitElementCountMismatch,
                             ToClass, dest_elem_fields_size,
                             src_elem_fields.size());
    }
    return SemIR::ErrorInst::InstId;
  }

  // Prepare to look up fields in the source by index.
  Map<SemIR::NameId, int32_t> src_field_indexes;
  if (src_type.fields_id != dest_type.fields_id) {
    for (auto [i, field] : llvm::enumerate(src_elem_fields)) {
      auto result = src_field_indexes.Insert(field.name_id, i);
      CARBON_CHECK(result.is_inserted(), "Duplicate field in source structure");
    }
  }

  ConversionTarget::Kind inner_kind =
      GetAggregateElementConversionTargetKind(sem_ir, target);

  // Initialize each element of the destination from the corresponding element
  // of the source.
  // TODO: Annotate diagnostics coming from here with the element index.
  auto new_block =
      literal_elems_id.has_value() && !dest_vptr_index.has_value()
          ? SemIR::CopyOnWriteInstBlock(&sem_ir, literal_elems_id)
          : SemIR::CopyOnWriteInstBlock(
                &sem_ir, SemIR::CopyOnWriteInstBlock::UninitializedBlock{
                             dest_elem_fields.size()});
  for (auto [i, dest_field] : llvm::enumerate(dest_elem_fields)) {
    if (dest_field.name_id == SemIR::NameId::Vptr) {
      if constexpr (!ToClass) {
        CARBON_FATAL("Only classes should have vptrs.");
      }
      target.storage_access_block->InsertHere();
      auto vptr_type_id =
          context.types().GetTypeIdForTypeInstId(dest_field.type_inst_id);
      auto dest_id =
          AddInst<SemIR::ClassElementAccess>(context, value_loc_id,
                                             {.type_id = vptr_type_id,
                                              .base_id = target.storage_id,
                                              .index = SemIR::ElementIndex(i)});
      auto vtable_ptr_id = SemIR::InstId::None;
      if (vtable_class_type) {
        vtable_ptr_id =
            CreateVtablePtrRef(context, value_loc_id, *vtable_class_type);
        // Track that we initialized the vptr so we don't do it again.
        vtable_class_type = nullptr;
      } else {
        // For a partial class type, we leave the vtable pointer uninitialized.
        // TODO: Consider storing a specified value such as null for hardening.
        vtable_ptr_id = AddInst<SemIR::UninitializedValue>(
            context, value_loc_id,
            {.type_id =
                 GetPointerType(context, SemIR::VtableType::TypeInstId)});
      }
      auto init_id = AddInst<SemIR::InPlaceInit>(context, value_loc_id,
                                                 {.type_id = vptr_type_id,
                                                  .src_id = vtable_ptr_id,
                                                  .dest_id = dest_id});
      new_block.Set(i, init_id);
      continue;
    }

    // Find the matching source field.
    auto src_field_index = i;
    if (src_type.fields_id != dest_type.fields_id) {
      if (auto lookup = src_field_indexes.Lookup(dest_field.name_id)) {
        src_field_index = lookup.value();
      } else {
        if (target.diagnose) {
          if (literal_elems_id.has_value()) {
            CARBON_DIAGNOSTIC(
                StructInitMissingFieldInLiteral, Error,
                "missing value for field `{0}` in struct initialization",
                SemIR::NameId);
            context.emitter().Emit(value_loc_id,
                                   StructInitMissingFieldInLiteral,
                                   dest_field.name_id);
          } else {
            CARBON_DIAGNOSTIC(StructInitMissingFieldInConversion, Error,
                              "cannot convert from struct type {0} to {1}: "
                              "missing field `{2}` in source type",
                              TypeOfInstId, SemIR::TypeId, SemIR::NameId);
            context.emitter().Emit(value_loc_id,
                                   StructInitMissingFieldInConversion, value_id,
                                   target.type_id, dest_field.name_id);
          }
        }
        return SemIR::ErrorInst::InstId;
      }
    }
    auto src_field = src_elem_fields[src_field_index];

    // When initializing the `.base` field of a class, the destination type is
    // `partial Base`, not `Base`.
    // TODO: Skip this if the source field is an initializing expression of the
    // non-partial type in order to produce smaller IR.
    auto dest_field_type_inst_id = dest_field.type_inst_id;
    if (dest_field.name_id == SemIR::NameId::Base) {
      auto partial_type_id = GetQualifiedType(
          context,
          context.types().GetTypeIdForTypeInstId(dest_field.type_inst_id),
          SemIR::TypeQualifiers::Partial);
      dest_field_type_inst_id = context.types().GetTypeInstId(partial_type_id);
    }

    // TODO: This call recurses back into conversion. Switch to an iterative
    // approach.
    auto dest_field_index = src_field_index;
    if (dest_vptr_index.has_value() &&
        static_cast<int32_t>(src_field_index) >= dest_vptr_index.index) {
      dest_field_index += 1;
    }
    auto init_id =
        ConvertAggregateElement<SemIR::StructAccess, TargetAccessInstT>(
            context, value_loc_id, value_id, src_field.type_inst_id,
            literal_elems, inner_kind, target.storage_id,
            dest_field_type_inst_id, target.storage_access_block,
            src_field_index, dest_field_index);
    if (init_id == SemIR::ErrorInst::InstId) {
      return SemIR::ErrorInst::InstId;
    }

    // When initializing the base, adjust the type of the initializer from
    // `partial Base` to `Base`. This isn't strictly correct, since we haven't
    // finished initializing a `Base` until we store to the vptr, but is better
    // than having an inconsistent type for the struct field initializer.
    if (dest_field_type_inst_id != dest_field.type_inst_id) {
      init_id = AddInst<SemIR::AsCompatible>(
          context, value_loc_id,
          {.type_id =
               context.types().GetTypeIdForTypeInstId(dest_field.type_inst_id),
           .source_id = init_id});
    }

    new_block.Set(i, init_id);
  }

  bool is_init = target.is_initializer();
  if (ToClass) {
    target.storage_access_block->InsertHere();
    CARBON_CHECK(is_init,
                 "Converting directly to a class value is not supported");
    auto result_id = AddInst<SemIR::ClassInit>(context, value_loc_id,
                                               {.type_id = target.type_id,
                                                .elements_id = new_block.id(),
                                                .dest_id = target.storage_id});
    if (vtable_class_type) {
      result_id = ConvertPartialInitializerToNonPartial(
          context, target, *vtable_class_type, result_id);
    }
    return result_id;
  } else if (is_init) {
    target.storage_access_block->InsertHere();
    return AddInst<SemIR::StructInit>(context, value_loc_id,
                                      {.type_id = target.type_id,
                                       .elements_id = new_block.id(),
                                       .dest_id = target.storage_id});
  } else {
    return AddInst<SemIR::StructValue>(
        context, value_loc_id,
        {.type_id = target.type_id, .elements_id = new_block.id()});
  }
}

// Performs a conversion from a struct to a struct type. This function only
// converts the type, and does not perform a final conversion to the requested
// expression category.
static auto ConvertStructToStruct(Context& context, SemIR::StructType src_type,
                                  SemIR::StructType dest_type,
                                  SemIR::InstId value_id,
                                  ConversionTarget target) -> SemIR::InstId {
  return ConvertStructToStructOrClass<SemIR::StructAccess>(
      context, src_type, dest_type, value_id, target);
}

// Performs a conversion from a struct to a class type. This function only
// converts the type, and does not perform a final conversion to the requested
// expression category.
static auto ConvertStructToClass(Context& context, SemIR::StructType src_type,
                                 SemIR::ClassType dest_type,
                                 SemIR::InstId value_id,
                                 ConversionTarget target,
                                 bool is_partial = false) -> SemIR::InstId {
  CARBON_CHECK(target.kind != ConversionTarget::InPlaceInitializing ||
               target.storage_id.has_value());
  PendingBlock target_block(&context);
  auto& dest_class_info = context.classes().Get(dest_type.class_id);
  CARBON_CHECK(is_partial ||
               dest_class_info.inheritance_kind != SemIR::Class::Abstract);
  auto object_repr_id =
      dest_class_info.GetObjectRepr(context.sem_ir(), dest_type.specific_id);
  if (object_repr_id == SemIR::ErrorInst::TypeId) {
    return SemIR::ErrorInst::InstId;
  }
  if (context.types().Is<SemIR::CustomLayoutType>(object_repr_id)) {
    // Builtin conversion does not apply.
    return value_id;
  }
  auto dest_struct_type =
      context.types().GetAs<SemIR::StructType>(object_repr_id);

  // If we're trying to create a class value, form temporary storage to hold the
  // initializer.
  if (!target.is_initializer()) {
    target.kind = ConversionTarget::Initializing;
    target.storage_access_block = &target_block;
    target.storage_id = target_block.AddInst<SemIR::TemporaryStorage>(
        SemIR::LocId(value_id), {.type_id = target.type_id});
  }

  return ConvertStructToStructOrClass<SemIR::ClassElementAccess>(
      context, src_type, dest_struct_type, value_id, target,
      is_partial ? nullptr : &dest_type);
}

// An inheritance path is a sequence of `BaseDecl`s and corresponding base types
// in order from derived to base.
using InheritancePath =
    llvm::SmallVector<std::pair<SemIR::InstId, SemIR::TypeId>>;

// Computes the inheritance path from class `derived_id` to class `base_id`.
// Returns nullopt if `derived_id` is not a class derived from `base_id`.
static auto ComputeInheritancePath(Context& context, SemIR::LocId loc_id,
                                   SemIR::TypeId derived_id,
                                   SemIR::TypeId base_id)
    -> std::optional<InheritancePath> {
  // We intend for NRVO to be applied to `result`. All `return` statements in
  // this function should `return result;`.
  std::optional<InheritancePath> result(std::in_place);
  if (!TryToCompleteType(context, derived_id, loc_id)) {
    // TODO: Should we give an error here? If we don't, and there is an
    // inheritance path when the class is defined, we may have a coherence
    // problem.
    result = std::nullopt;
    return result;
  }
  while (derived_id != base_id) {
    auto derived_class_type =
        context.types().TryGetAs<SemIR::ClassType>(derived_id);
    if (!derived_class_type) {
      result = std::nullopt;
      break;
    }
    auto& derived_class = context.classes().Get(derived_class_type->class_id);
    auto base_type_id = derived_class.GetBaseType(
        context.sem_ir(), derived_class_type->specific_id);
    if (!base_type_id.has_value()) {
      result = std::nullopt;
      break;
    }
    result->push_back({derived_class.base_id, base_type_id});
    derived_id = base_type_id;
  }
  return result;
}

// Performs a conversion from a derived class value or reference to a base class
// value or reference.
static auto ConvertDerivedToBase(Context& context, SemIR::LocId loc_id,
                                 SemIR::InstId value_id,
                                 const InheritancePath& path) -> SemIR::InstId {
  // Materialize a temporary if necessary.
  value_id = ConvertToValueOrRefExpr(context, value_id);

  // Preserve type qualifiers.
  auto quals = context.types()
                   .GetUnqualifiedTypeAndQualifiers(
                       context.insts().Get(value_id).type_id())
                   .second;

  // Add a series of `.base` accesses.
  for (auto [base_id, base_type_id] : path) {
    auto base_decl = context.insts().GetAs<SemIR::BaseDecl>(base_id);
    value_id = AddInst<SemIR::ClassElementAccess>(
        context, loc_id,
        {.type_id = GetQualifiedType(context, base_type_id, quals),
         .base_id = value_id,
         .index = base_decl.index});
  }
  return value_id;
}

// Performs a conversion from a derived class pointer to a base class pointer.
static auto ConvertDerivedPointerToBasePointer(
    Context& context, SemIR::LocId loc_id, SemIR::PointerType src_ptr_type,
    SemIR::TypeId dest_ptr_type_id, SemIR::InstId ptr_id,
    const InheritancePath& path) -> SemIR::InstId {
  auto pointee_type_id =
      context.types().GetTypeIdForTypeInstId(src_ptr_type.pointee_id);

  // Form `*p`.
  ptr_id = ConvertToValueExpr(context, ptr_id);
  auto ref_id = AddInst<SemIR::Deref>(
      context, loc_id, {.type_id = pointee_type_id, .pointer_id = ptr_id});

  // Convert as a reference expression.
  ref_id = ConvertDerivedToBase(context, loc_id, ref_id, path);

  // Take the address.
  return AddInst<SemIR::AddrOf>(
      context, loc_id, {.type_id = dest_ptr_type_id, .lvalue_id = ref_id});
}

// Returns whether `category` is a valid expression category to produce as a
// result of a conversion with kind `target_kind`.
static auto IsValidExprCategoryForConversionTarget(
    SemIR::ExprCategory category, ConversionTarget::Kind target_kind) -> bool {
  switch (target_kind) {
    case ConversionTarget::Value:
      return category == SemIR::ExprCategory::Value;
    case ConversionTarget::ValueOrRef:
      return category == SemIR::ExprCategory::Value ||
             category == SemIR::ExprCategory::DurableRef ||
             category == SemIR::ExprCategory::EphemeralRef;
    case ConversionTarget::Discarded:
      return category == SemIR::ExprCategory::Value ||
             category == SemIR::ExprCategory::DurableRef ||
             category == SemIR::ExprCategory::EphemeralRef ||
             category == SemIR::ExprCategory::ReprInitializing ||
             category == SemIR::ExprCategory::InPlaceInitializing;
    case ConversionTarget::RefParam:
    case ConversionTarget::UnmarkedRefParam:
      return category == SemIR::ExprCategory::DurableRef ||
             category == SemIR::ExprCategory::EphemeralRef;
    case ConversionTarget::DurableRef:
      return category == SemIR::ExprCategory::DurableRef;
    case ConversionTarget::CppThunkRef:
      return category == SemIR::ExprCategory::EphemeralRef;
    case ConversionTarget::NoOp:
    case ConversionTarget::ExplicitAs:
    case ConversionTarget::ExplicitUnsafeAs:
      return true;
    case ConversionTarget::InPlaceInitializing:
      return category == SemIR::ExprCategory::InPlaceInitializing;
    case ConversionTarget::Initializing:
      return category == SemIR::ExprCategory::ReprInitializing;
  }
}

// Determines whether the initialization representation of the type is a copy of
// the value representation.
static auto InitReprIsCopyOfValueRepr(const SemIR::File& sem_ir,
                                      SemIR::TypeId type_id) -> bool {
  // The initializing representation is a copy of the value representation if
  // they're both copies of the object representation.
  return SemIR::InitRepr::ForType(sem_ir, type_id).IsCopyOfObjectRepr() &&
         SemIR::ValueRepr::ForType(sem_ir, type_id)
             .IsCopyOfObjectRepr(sem_ir, type_id);
}

// Determines whether we can pull a value directly out of an initializing
// expression of type `type_id` to initialize a target of type `type_id` and
// kind `target_kind`.
static auto CanUseValueOfInitializer(const SemIR::File& sem_ir,
                                     SemIR::TypeId type_id,
                                     ConversionTarget::Kind target_kind)
    -> bool {
  if (!IsValidExprCategoryForConversionTarget(SemIR::ExprCategory::Value,
                                              target_kind)) {
    // We don't want a value expression.
    return false;
  }

  // We can pull a value out of an initializing expression if it holds one.
  return InitReprIsCopyOfValueRepr(sem_ir, type_id);
}

// Determine whether the given set of qualifiers can be added by a conversion
// of an expression of the given category.
static auto CanAddQualifiers(SemIR::TypeQualifiers quals,
                             SemIR::ExprCategory cat) -> bool {
  if (quals.HasAnyOf(SemIR::TypeQualifiers::MaybeUnformed) &&
      !SemIR::IsRefCategory(cat)) {
    // `MaybeUnformed(T)` may have a different value representation or
    // initializing representation from `T`, so only allow it to be added for a
    // reference expression.
    // TODO: We should allow converting an initializing expression of type `T`
    // to `MaybeUnformed(T)`. `PerformBuiltinConversion` will need to generate
    // an `InPlaceInit` instruction when needed.
    // NOLINTNEXTLINE(readability-simplify-boolean-expr)
    return false;
  }

  // `const` and `partial` can always be added.
  return true;
}

// Determine whether the given set of qualifiers can be removed by a conversion
// of an expression of the given category.
static auto CanRemoveQualifiers(SemIR::TypeQualifiers quals,
                                SemIR::ExprCategory cat,
                                ConversionTarget::Kind kind) -> bool {
  bool allow_unsafe = kind == ConversionTarget::ExplicitUnsafeAs;

  if (quals.HasAnyOf(SemIR::TypeQualifiers::Const) && !allow_unsafe &&
      SemIR::IsRefCategory(cat) &&
      IsValidExprCategoryForConversionTarget(cat, kind)) {
    // Removing `const` is an unsafe conversion for a reference expression. But
    // it's OK if we will be converting to a different category as part of this
    // overall conversion anyway.
    return false;
  }

  if (quals.HasAnyOf(SemIR::TypeQualifiers::Partial) && !allow_unsafe &&
      !SemIR::IsInitializerCategory(cat)) {
    // Removing `partial` is an unsafe conversion for a non-initializing
    // expression. But it's OK for an initializing expression because we will
    // initialize the vptr as part of the conversion.
    return false;
  }

  if (quals.HasAnyOf(SemIR::TypeQualifiers::MaybeUnformed) &&
      (!allow_unsafe || SemIR::IsInitializerCategory(cat))) {
    // As an unsafe conversion, `MaybeUnformed` can be removed from a value or
    // reference expression.
    return false;
  }

  return true;
}

static auto DiagnoseConversionFailureToConstraintValue(
    Context& context, SemIR::LocId loc_id, SemIR::InstId expr_id,
    SemIR::TypeId target_type_id) -> void {
  CARBON_CHECK(context.types().IsFacetType(target_type_id));

  // If the source type is/has a facet value (converted with `as type` or
  // otherwise), then we can include its `FacetType` in the diagnostic to help
  // explain what interfaces the source type implements.
  auto const_expr_id = GetCanonicalFacetOrTypeValue(context, expr_id);
  auto const_expr_type_id = context.insts().Get(const_expr_id).type_id();

  if (context.types().Is<SemIR::FacetType>(const_expr_type_id)) {
    CARBON_DIAGNOSTIC(ConversionFailureFacetToFacet, Error,
                      "cannot convert type {0} that implements {1} into type "
                      "implementing {2}",
                      InstIdAsType, SemIR::TypeId, SemIR::TypeId);
    context.emitter().Emit(loc_id, ConversionFailureFacetToFacet, expr_id,
                           const_expr_type_id, target_type_id);
  } else {
    CARBON_DIAGNOSTIC(ConversionFailureTypeToFacet, Error,
                      "cannot convert type {0} into type implementing {1}",
                      InstIdAsType, SemIR::TypeId);
    context.emitter().Emit(loc_id, ConversionFailureTypeToFacet, expr_id,
                           target_type_id);
  }
}

static auto PerformBuiltinConversion(Context& context, SemIR::LocId loc_id,
                                     SemIR::InstId value_id,
                                     ConversionTarget target) -> SemIR::InstId {
  auto& sem_ir = context.sem_ir();
  auto value = sem_ir.insts().Get(value_id);
  auto value_type_id = value.type_id();
  auto target_type_inst = sem_ir.types().GetAsInst(target.type_id);

  // Various forms of implicit conversion are supported as builtin conversions,
  // either in addition to or instead of `impl`s of `ImplicitAs` in the Carbon
  // prelude. There are a few reasons we need to perform some of these
  // conversions as builtins:
  //
  // 1) Conversions from struct and tuple *literals* have special rules that
  //    cannot be implemented by invoking `ImplicitAs`. Specifically, we must
  //    recurse into the elements of the literal before performing
  //    initialization in order to avoid unnecessary conversions between
  //    expression categories that would be performed by `ImplicitAs.Convert`.
  // 2) (Not implemented yet) Conversion of a facet to a facet type depends on
  //    the value of the facet, not only its type, and therefore cannot be
  //    modeled by `ImplicitAs`.
  // 3) Some of these conversions are used while checking the library
  //    definition of `ImplicitAs` itself or implementations of it.
  //
  // We also expect to see better performance by avoiding an `impl` lookup for
  // common conversions.
  //
  // TODO: We should provide a debugging flag to turn off as many of these
  // builtin conversions as we can so that we can test that they do the same
  // thing as the library implementations.
  //
  // The builtin conversions that correspond to `impl`s in the library all
  // correspond to `final impl`s, so we don't need to worry about `ImplicitAs`
  // being specialized in any of these cases.

  // If the value is already of the right kind and expression category, there's
  // nothing to do. Performing a conversion would decompose and rebuild tuples
  // and structs, so it's important that we bail out early in this case.
  if (value_type_id == target.type_id) {
    auto value_cat = SemIR::GetExprCategory(sem_ir, value_id);
    if (IsValidExprCategoryForConversionTarget(value_cat, target.kind)) {
      return value_id;
    }

    // If the source is an initializing expression, we may be able to pull a
    // value right out of it.
    if (value_cat == SemIR::ExprCategory::ReprInitializing &&
        CanUseValueOfInitializer(sem_ir, value_type_id, target.kind)) {
      return AddInst<SemIR::ValueOfInitializer>(
          context, loc_id, {.type_id = value_type_id, .init_id = value_id});
    }

    // Materialization is handled as part of the enclosing conversion.
    if (SemIR::IsInitializerCategory(value_cat) &&
        target.kind == ConversionTarget::ValueOrRef) {
      return value_id;
    }

    // Final destination store is handled as part of the enclosing conversion.
    if (value_cat == SemIR::ExprCategory::ReprInitializing &&
        target.kind == ConversionTarget::InPlaceInitializing) {
      return value_id;
    }

    // PerformBuiltinConversion converts each part of a tuple or struct, even
    // when the types are the same. This is not done for classes since they have
    // to define their conversions as part of their api.
    //
    // If a class adapts a tuple or struct, we convert each of its parts when
    // there's no other conversion going on (the source and target types are the
    // same). To do so, we have to insert a conversion of the value up to the
    // foundation and back down, and a conversion of the initializing object if
    // there is one.
    //
    // Implementation note: We do the conversion through a call to
    // PerformBuiltinConversion() call rather than a Convert() call to avoid
    // extraneous `converted` semir instructions on the adapted types, and as a
    // shortcut to doing the explicit calls to walk the parts of the
    // tuple/struct which happens inside PerformBuiltinConversion().
    if (auto foundation_type_id =
            context.types().GetTransitiveAdaptedType(value_type_id);
        foundation_type_id != value_type_id &&
        context.types().IsOneOf<SemIR::StructType, SemIR::TupleType>(
            foundation_type_id)) {
      auto foundation_value_id = AddInst<SemIR::AsCompatible>(
          context, loc_id,
          {.type_id = foundation_type_id, .source_id = value_id});

      auto foundation_init_id = target.storage_id;
      if (foundation_init_id != SemIR::InstId::None) {
        foundation_init_id =
            target.storage_access_block->AddInst<SemIR::AsCompatible>(
                loc_id, {.type_id = foundation_type_id,
                         .source_id = target.storage_id});
      }

      {
        // While the types are the same, the conversion can still fail if it
        // performs a copy while converting the value to another category, and
        // the type (or some part of it) is not copyable.
        Diagnostics::AnnotationScope annotate_diagnostics(
            &context.emitter(), [&](auto& builder) {
              CARBON_DIAGNOSTIC(InCopy, Note, "in copy of {0}", TypeOfInstId);
              builder.Note(value_id, InCopy, value_id);
            });

        foundation_value_id = PerformBuiltinConversion(
            context, loc_id, foundation_value_id,
            {.kind = target.kind,
             .type_id = foundation_type_id,
             .storage_id = foundation_init_id,
             .storage_access_block = target.storage_access_block,
             .diagnose = target.diagnose});
        if (foundation_value_id == SemIR::ErrorInst::InstId) {
          return SemIR::ErrorInst::InstId;
        }
      }

      return AddInst<SemIR::AsCompatible>(
          context, loc_id,
          {.type_id = target.type_id, .source_id = foundation_value_id});
    }
  }

  // T implicitly converts to U if T and U are the same ignoring qualifiers, and
  // we're allowed to remove / add any qualifiers that differ. Similarly, T
  // explicitly converts to U if T is compatible with U, and we're allowed to
  // remove / add any qualifiers that differ.
  if (target.type_id != value_type_id) {
    auto [target_foundation_id, target_quals] =
        target.is_explicit_as()
            ? context.types().GetTransitiveUnqualifiedAdaptedType(
                  target.type_id)
            : context.types().GetUnqualifiedTypeAndQualifiers(target.type_id);
    auto [value_foundation_id, value_quals] =
        target.is_explicit_as()
            ? context.types().GetTransitiveUnqualifiedAdaptedType(value_type_id)
            : context.types().GetUnqualifiedTypeAndQualifiers(value_type_id);
    if (target_foundation_id == value_foundation_id) {
      auto category = SemIR::GetExprCategory(context.sem_ir(), value_id);
      auto added_quals = target_quals & ~value_quals;
      auto removed_quals = value_quals & ~target_quals;
      if (CanAddQualifiers(added_quals, category) &&
          CanRemoveQualifiers(removed_quals, category, target.kind)) {
        // For a struct or tuple literal, perform a category conversion if
        // necessary.
        if (category == SemIR::ExprCategory::Mixed) {
          value_id = PerformBuiltinConversion(context, loc_id, value_id,
                                              {.kind = ConversionTarget::Value,
                                               .type_id = value_type_id,
                                               .diagnose = target.diagnose});
        }

        // `MaybeUnformed(T)` might have a pointer value representation when `T`
        // does not, so convert as needed when removing `MaybeUnformed`.
        bool need_value_binding = false;
        if ((removed_quals & SemIR::TypeQualifiers::MaybeUnformed) !=
                SemIR::TypeQualifiers::None &&
            category == SemIR::ExprCategory::Value) {
          auto value_rep =
              SemIR::ValueRepr::ForType(context.sem_ir(), value_type_id);
          auto unformed_value_rep =
              SemIR::ValueRepr::ForType(context.sem_ir(), target.type_id);
          if (value_rep.kind != unformed_value_rep.kind) {
            CARBON_CHECK(unformed_value_rep.kind == SemIR::ValueRepr::Pointer);
            value_id = AddInst<SemIR::ValueAsRef>(
                context, loc_id,
                {.type_id = value_type_id, .value_id = value_id});
            need_value_binding = true;
          }
        }

        if ((removed_quals & SemIR::TypeQualifiers::Partial) !=
                SemIR::TypeQualifiers::None &&
            SemIR::IsInitializerCategory(category)) {
          auto unqual_target_type_id =
              context.types().GetUnqualifiedType(target.type_id);
          if (auto target_class_type =
                  context.types().TryGetAs<SemIR::ClassType>(
                      unqual_target_type_id)) {
            value_id = ConvertPartialInitializerToNonPartial(
                context, target, *target_class_type, value_id);
          }
        }

        value_id = AddInst<SemIR::AsCompatible>(
            context, loc_id,
            {.type_id = target.type_id, .source_id = value_id});

        if (need_value_binding) {
          value_id = AddInst<SemIR::AcquireValue>(
              context, loc_id,
              {.type_id = target.type_id, .value_id = value_id});
        }
        return value_id;
      } else {
        // TODO: Produce a custom diagnostic explaining that we can't perform
        // this conversion due to the change in qualifiers and/or the expression
        // category.
      }
    }
  }

  // A tuple (T1, T2, ..., Tn) converts to (U1, U2, ..., Un) if each Ti
  // converts to Ui.
  if (auto target_tuple_type = target_type_inst.TryAs<SemIR::TupleType>()) {
    if (auto src_tuple_type =
            sem_ir.types().TryGetAs<SemIR::TupleType>(value_type_id)) {
      return ConvertTupleToTuple(context, *src_tuple_type, *target_tuple_type,
                                 value_id, target);
    }
  }

  // A struct {.f_1: T_1, .f_2: T_2, ..., .f_n: T_n} converts to
  // {.f_p(1): U_p(1), .f_p(2): U_p(2), ..., .f_p(n): U_p(n)} if
  // (p(1), ..., p(n)) is a permutation of (1, ..., n) and each Ti converts
  // to Ui.
  if (auto target_struct_type = target_type_inst.TryAs<SemIR::StructType>()) {
    if (auto src_struct_type =
            sem_ir.types().TryGetAs<SemIR::StructType>(value_type_id)) {
      return ConvertStructToStruct(context, *src_struct_type,
                                   *target_struct_type, value_id, target);
    }
  }

  // No other conversions apply when the source and destination types are the
  // same.
  if (value_type_id == target.type_id) {
    return value_id;
  }

  // A tuple (T1, T2, ..., Tn) converts to array(T, n) if each Ti converts to T.
  if (auto target_array_type = target_type_inst.TryAs<SemIR::ArrayType>()) {
    if (auto src_tuple_type =
            sem_ir.types().TryGetAs<SemIR::TupleType>(value_type_id)) {
      return ConvertTupleToArray(context, *src_tuple_type, *target_array_type,
                                 value_id, target);
    }
  }

  // Split the qualifiers off the target type.
  // TODO: Most conversions should probably be looking at the unqualified target
  // type.
  auto [target_unqual_type_id, target_quals] =
      context.types().GetUnqualifiedTypeAndQualifiers(target.type_id);
  auto target_unqual_type_inst =
      sem_ir.types().GetAsInst(target_unqual_type_id);

  // A struct {.f_1: T_1, .f_2: T_2, ..., .f_n: T_n} converts to a class type
  // if it converts to the struct type that is the class's representation type
  // (a struct with the same fields as the class, plus a base field where
  // relevant).
  if (auto target_class_type =
          target_unqual_type_inst.TryAs<SemIR::ClassType>()) {
    if (auto src_struct_type =
            sem_ir.types().TryGetAs<SemIR::StructType>(value_type_id)) {
      if (!context.classes()
               .Get(target_class_type->class_id)
               .adapt_id.has_value()) {
        return ConvertStructToClass(
            context, *src_struct_type, *target_class_type, value_id, target,
            target_quals.HasAnyOf(SemIR::TypeQualifiers::Partial));
      }
    }

    // An expression of type T converts to U if T is a class derived from U.
    //
    // TODO: Combine this with the qualifiers and adapter conversion logic above
    // to allow qualifiers and inheritance conversions to be performed together.
    if (auto path = ComputeInheritancePath(context, loc_id, value_type_id,
                                           target.type_id);
        path && !path->empty()) {
      return ConvertDerivedToBase(context, loc_id, value_id, *path);
    }
  }

  // A pointer T* converts to [qualified] U* if T is the same as U, or is a
  // class derived from U.
  if (auto target_pointer_type = target_type_inst.TryAs<SemIR::PointerType>()) {
    if (auto src_pointer_type =
            sem_ir.types().TryGetAs<SemIR::PointerType>(value_type_id)) {
      auto target_pointee_id = context.types().GetTypeIdForTypeInstId(
          target_pointer_type->pointee_id);
      auto src_pointee_id =
          context.types().GetTypeIdForTypeInstId(src_pointer_type->pointee_id);
      // Try to complete the pointee types so that we can walk through adapters
      // to their adapted types.
      TryToCompleteType(context, target_pointee_id, loc_id);
      TryToCompleteType(context, src_pointee_id, loc_id);
      auto [unqual_target_pointee_type_id, target_quals] =
          sem_ir.types().GetTransitiveUnqualifiedAdaptedType(target_pointee_id);
      auto [unqual_src_pointee_type_id, src_quals] =
          sem_ir.types().GetTransitiveUnqualifiedAdaptedType(src_pointee_id);

      // If the qualifiers are incompatible, we can't perform a conversion,
      // except with `unsafe as`.
      if ((src_quals & ~target_quals) != SemIR::TypeQualifiers::None &&
          target.kind != ConversionTarget::ExplicitUnsafeAs) {
        // TODO: Consider producing a custom diagnostic here for a cast that
        // discards constness.
        return value_id;
      }

      if (unqual_target_pointee_type_id != unqual_src_pointee_type_id) {
        // If there's an inheritance path from target to source, this is a
        // derived to base conversion.
        if (auto path = ComputeInheritancePath(context, loc_id,
                                               unqual_src_pointee_type_id,
                                               unqual_target_pointee_type_id);
            path && !path->empty()) {
          value_id = ConvertDerivedPointerToBasePointer(
              context, loc_id, *src_pointer_type, target.type_id, value_id,
              *path);
        } else {
          // No conversion was possible.
          return value_id;
        }
      }

      // Perform a compatible conversion to add any new qualifiers.
      if (src_quals != target_quals) {
        return AddInst<SemIR::AsCompatible>(
            context, loc_id,
            {.type_id = target.type_id, .source_id = value_id});
      }
      return value_id;
    }
  }

  if (sem_ir.types().IsFacetType(target.type_id)) {
    auto type_value_id = SemIR::TypeInstId::None;

    // A tuple of types converts to type `type`.
    if (sem_ir.types().Is<SemIR::TupleType>(value_type_id)) {
      type_value_id =
          ConvertTupleToType(context, loc_id, value_id, value_type_id, target);
    }

    // `{}` converts to `{} as type`.
    if (auto struct_type =
            sem_ir.types().TryGetAs<SemIR::StructType>(value_type_id)) {
      if (struct_type->fields_id == SemIR::StructTypeFieldsId::Empty) {
        type_value_id = sem_ir.types().GetTypeInstId(value_type_id);
      }
    }

    if (type_value_id != SemIR::InstId::None) {
      if (sem_ir.types().Is<SemIR::FacetType>(target.type_id)) {
        // Use the converted `TypeType` value for converting to a facet.
        value_id = type_value_id;
        value_type_id = SemIR::TypeType::TypeId;
      } else {
        // We wanted a `TypeType`, and we've done that.
        return type_value_id;
      }
    }
  }

  // FacetType converts to Type by wrapping the facet value in
  // FacetAccessType.
  if (target.type_id == SemIR::TypeType::TypeId &&
      sem_ir.types().Is<SemIR::FacetType>(value_type_id)) {
    return AddInst<SemIR::FacetAccessType>(
        context, loc_id,
        {.type_id = target.type_id, .facet_value_inst_id = value_id});
  }

  // Type values can convert to facet values, and facet values can convert to
  // other facet values, as long as they satisfy the required interfaces of the
  // target `FacetType`.
  if (sem_ir.types().Is<SemIR::FacetType>(target.type_id) &&
      sem_ir.types().IsOneOf<SemIR::TypeType, SemIR::FacetType>(
          value_type_id)) {
    // TODO: Runtime facet values should be allowed to convert based on their
    // FacetTypes, but we assume constant values for impl lookup at the moment.
    if (!context.constant_values().Get(value_id).is_constant()) {
      context.TODO(loc_id, "conversion of runtime facet value");
      return SemIR::ErrorInst::InstId;
    }

    // Get the canonical type for which we want to attach a new set of witnesses
    // to match the requirements of the target FacetType.
    auto type_inst_id = SemIR::TypeInstId::None;
    if (sem_ir.types().Is<SemIR::FacetType>(value_type_id)) {
      type_inst_id = AddTypeInst<SemIR::FacetAccessType>(
          context, loc_id,
          {.type_id = SemIR::TypeType::TypeId,
           .facet_value_inst_id = value_id});
    } else {
      type_inst_id = context.types().GetAsTypeInstId(value_id);

      // Shortcut for lossless round trips through a FacetAccessType when
      // converting back to the type of the original symbolic binding facet
      // value.
      //
      // In the case where the FacetAccessType wraps a SymbolicBinding with the
      // exact facet type that we are converting to, the resulting FacetValue
      // would evaluate back to the original SymbolicBinding as its canonical
      // form. We can skip past the whole impl lookup step then and do that
      // here.
      auto facet_value_inst_id =
          GetCanonicalFacetOrTypeValue(context, type_inst_id);
      if (sem_ir.insts().Get(facet_value_inst_id).type_id() == target.type_id) {
        return facet_value_inst_id;
      }
    }

    // Conversion from a facet value (which has type `FacetType`) or a type
    // value (which has type `TypeType`) to a facet value. We can do this if the
    // type satisfies the requirements of the target `FacetType`, as determined
    // by finding impl witnesses for the target FacetType.
    auto lookup_result = LookupImplWitness(
        context, loc_id, sem_ir.constant_values().Get(type_inst_id),
        sem_ir.types().GetConstantId(target.type_id), target.diagnose);
    if (lookup_result.has_value()) {
      if (lookup_result.has_error_value()) {
        return SemIR::ErrorInst::InstId;
      } else {
        // Note that `FacetValue`'s type is the same `FacetType` that was used
        // to construct the set of witnesses, ie. the query to
        // `LookupImplWitness()`. This ensures that the witnesses are in the
        // same order as the `required_impls()` in the `IdentifiedFacetType` of
        // the `FacetValue`'s type.
        return AddInst<SemIR::FacetValue>(
            context, loc_id,
            {.type_id = target.type_id,
             .type_inst_id = type_inst_id,
             .witnesses_block_id = lookup_result.inst_block_id()});
      }
    } else {
      // If impl lookup fails, don't keep looking for another way to convert.
      // See https://github.com/carbon-language/carbon-lang/issues/5122.
      // TODO: Pass this function into `LookupImplWitness` so it can construct
      // the error add notes explaining failure.
      if (target.diagnose) {
        DiagnoseConversionFailureToConstraintValue(context, loc_id, value_id,
                                                   target.type_id);
      }
      return SemIR::ErrorInst::InstId;
    }
  }

  // No builtin conversion applies.
  return value_id;
}

// Given a value expression, form a corresponding initializer that copies from
// that value to the specified target, if it is possible to do so.
static auto PerformCopy(Context& context, SemIR::InstId expr_id,
                        const ConversionTarget& target) -> SemIR::InstId {
  auto copy_id = BuildUnaryOperator(
      context, SemIR::LocId(expr_id), {.interface_name = CoreIdentifier::Copy},
      expr_id, target.diagnose, [&](auto& builder) {
        CARBON_DIAGNOSTIC(CopyOfUncopyableType, Context,
                          "cannot copy value of type {0}", TypeOfInstId);
        builder.Context(expr_id, CopyOfUncopyableType, expr_id);
      });
  return copy_id;
}

// Tries to form a `ValueAsRef` conversion that extracts the pointer value from
// a value expression with a pointer value representation. Returns the converted
// expression, or None if the conversion was not applicable.
static auto TryMakeValueAsRef(Context& context, SemIR::InstId expr_id)
    -> SemIR::InstId {
  auto expr = context.insts().Get(expr_id);

  // If the expression has a pointer value representation, extract that and use
  // it directly.
  if (SemIR::ValueRepr::ForType(context.sem_ir(), expr.type_id()).kind ==
      SemIR::ValueRepr::Pointer) {
    return AddInst<SemIR::ValueAsRef>(
        context, SemIR::LocId(expr_id),
        {.type_id = expr.type_id(), .value_id = expr_id});
  }

  return SemIR::InstId::None;
}

// Returns the Core interface name to use for a given kind of conversion.
static auto GetConversionInterfaceName(ConversionTarget::Kind kind)
    -> CoreIdentifier {
  switch (kind) {
    case ConversionTarget::ExplicitAs:
      return CoreIdentifier::As;
    case ConversionTarget::ExplicitUnsafeAs:
      return CoreIdentifier::UnsafeAs;
    default:
      return CoreIdentifier::ImplicitAs;
  }
}

auto PerformAction(Context& context, SemIR::LocId loc_id,
                   SemIR::ConvertToValueAction action) -> SemIR::InstId {
  return Convert(context, loc_id, action.inst_id,
                 {.kind = ConversionTarget::Value,
                  .type_id = context.types().GetTypeIdForTypeInstId(
                      action.target_type_inst_id)});
}

// State machine for performing category conversions.
class CategoryConverter {
 public:
  // Constructs a converter which converts an expression at the given location
  // to the given conversion target.
  CategoryConverter(Context& context, SemIR::LocId loc_id,
                    ConversionTarget& target)
      : context_(context),
        sem_ir_(context.sem_ir()),
        loc_id_(loc_id),
        target_(target) {}

  // Converts expr_id to the target specified in the constructor, and returns
  // the converted inst.
  auto Convert(SemIR::InstId expr_id) && -> SemIR::InstId {
    auto category = SemIR::GetExprCategory(sem_ir_, expr_id);
    while (true) {
      if (expr_id == SemIR::ErrorInst::InstId) {
        return expr_id;
      }
      CARBON_KIND_SWITCH(DoStep(expr_id, category)) {
        case CARBON_KIND(NextStep next_step): {
          CARBON_CHECK(next_step.expr_id != SemIR::InstId::None);
          expr_id = next_step.expr_id;
          category = next_step.category;
          break;
        }
        case CARBON_KIND(Done done): {
          return done.expr_id;
        }
      }
    }
  }

 private:
  // State that indicates there's more work to be done. As a convenience,
  // if expr_id is SemIR::ErrorInst::InstId, this is equivalent to
  // Done{SemIR::ErrorInst::InstId}.
  struct NextStep {
    // The inst to convert.
    SemIR::InstId expr_id;
    // The category of expr_id.
    SemIR::ExprCategory category;
  };

  // State that indicates we've finished category conversion.
  struct Done {
    // The result of the conversion.
    SemIR::InstId expr_id;
  };

  using State = std::variant<NextStep, Done>;

  // Performs the first step of converting `expr_id` with category `category`
  // to the target specified in the constructor, and returns the state after
  // that step.
  auto DoStep(SemIR::InstId expr_id, SemIR::ExprCategory category) const
      -> State;

  Context& context_;
  SemIR::File& sem_ir_;
  SemIR::LocId loc_id_;
  const ConversionTarget& target_;
};

auto CategoryConverter::DoStep(const SemIR::InstId expr_id,
                               const SemIR::ExprCategory category) const
    -> State {
  CARBON_DCHECK(SemIR::GetExprCategory(sem_ir_, expr_id) == category);
  switch (category) {
    case SemIR::ExprCategory::NotExpr:
    case SemIR::ExprCategory::Mixed:
    case SemIR::ExprCategory::Pattern:
      CARBON_FATAL("Unexpected expression {0} after builtin conversions",
                   sem_ir_.insts().Get(expr_id));

    case SemIR::ExprCategory::Error:
      return Done{SemIR::ErrorInst::InstId};

    case SemIR::ExprCategory::Dependent:
      context_.TODO(expr_id, "Support symbolic expression forms");
      return Done{SemIR::ErrorInst::InstId};

    case SemIR::ExprCategory::InPlaceInitializing:
    case SemIR::ExprCategory::ReprInitializing:
      if (target_.is_initializer()) {
        // Overwrite the initializer's storage argument with the inst currently
        // at target_.storage_id, if both are present and the storage argument
        // hasn't already been set. However, we skip this if the type is a C++
        // enum: in that case, we don't actually have an initializing
        // expression, we're just pretending we do.
        auto new_storage_id =
            OverwriteTemporaryStorageArg(sem_ir_, expr_id, target_);

        // If in-place initialization was requested, and it hasn't already
        // happened, ensure it happens now.
        if (target_.kind == ConversionTarget::InPlaceInitializing &&
            !IsInPlaceInitializing(context_, expr_id, category)) {
          target_.storage_access_block->InsertHere();
          CARBON_CHECK(new_storage_id.has_value());
          return Done{AddInst<SemIR::InPlaceInit>(context_, loc_id_,
                                                  {.type_id = target_.type_id,
                                                   .src_id = expr_id,
                                                   .dest_id = new_storage_id})};
        }
        return Done{expr_id};
      }

      if (target_.kind == ConversionTarget::Discarded) {
        DiscardInitializer(context_, expr_id);
        return Done{SemIR::InstId::None};
      } else if (IsValidExprCategoryForConversionTarget(category,
                                                        target_.kind)) {
        return Done{expr_id};
      } else {
        // Commit to using a temporary for this initializing expression.
        // TODO: Don't create a temporary if the initializing representation is
        // already a value representation.
        // TODO: If the target is DurableRef, materialize a VarStorage instead
        // of a TemporaryStorage to lifetime-extend.
        return NextStep{.expr_id = MaterializeTemporary(context_, expr_id),
                        .category = SemIR::ExprCategory::EphemeralRef};
      }

    case SemIR::ExprCategory::RefTagged: {
      auto tagged_expr_id =
          sem_ir_.insts().GetAs<SemIR::RefTagExpr>(expr_id).expr_id;
      auto tagged_expr_category =
          SemIR::GetExprCategory(sem_ir_, tagged_expr_id);
      if (target_.diagnose &&
          tagged_expr_category != SemIR::ExprCategory::DurableRef) {
        CARBON_DIAGNOSTIC(
            RefTagNotDurableRef, Error,
            "expression tagged with `ref` is not a durable reference");
        context_.emitter().Emit(tagged_expr_id, RefTagNotDurableRef);
      }

      if (target_.kind == ConversionTarget::RefParam) {
        return Done{expr_id};
      }

      // If the target isn't a reference parameter, ignore the `ref` tag.
      // Unnecessary `ref` tags are diagnosed earlier.
      return NextStep{.expr_id = tagged_expr_id,
                      .category = tagged_expr_category};
    }

    case SemIR::ExprCategory::DurableRef:
      if (target_.kind == ConversionTarget::DurableRef ||
          target_.kind == ConversionTarget::UnmarkedRefParam) {
        return Done{expr_id};
      }
      if (target_.kind == ConversionTarget::RefParam) {
        if (target_.diagnose) {
          CARBON_DIAGNOSTIC(
              RefParamNoRefTag, Error,
              "argument to `ref` parameter not marked with `ref`");
          context_.emitter().Emit(expr_id, RefParamNoRefTag);
        }
        return Done{expr_id};
      }
      [[fallthrough]];

    case SemIR::ExprCategory::EphemeralRef:
      // If a reference expression is an acceptable result, we're done.
      if (target_.kind == ConversionTarget::ValueOrRef ||
          target_.kind == ConversionTarget::Discarded ||
          target_.kind == ConversionTarget::CppThunkRef ||
          target_.kind == ConversionTarget::RefParam ||
          target_.kind == ConversionTarget::UnmarkedRefParam) {
        return Done{expr_id};
      }

      // If we have a reference and don't want one, form a value binding.
      // TODO: Support types with custom value representations.
      return NextStep{.expr_id = AddInst<SemIR::AcquireValue>(
                          context_, SemIR::LocId(expr_id),
                          {.type_id = target_.type_id, .value_id = expr_id}),
                      .category = SemIR::ExprCategory::Value};

    case SemIR::ExprCategory::Value:
      if (target_.kind == ConversionTarget::DurableRef) {
        if (target_.diagnose) {
          CARBON_DIAGNOSTIC(ConversionFailureNonRefToRef, Error,
                            "cannot bind durable reference to non-reference "
                            "value of type {0}",
                            SemIR::TypeId);
          context_.emitter().Emit(loc_id_, ConversionFailureNonRefToRef,
                                  target_.type_id);
        }
        return Done{SemIR::ErrorInst::InstId};
      }

      if (target_.kind == ConversionTarget::RefParam ||
          target_.kind == ConversionTarget::UnmarkedRefParam) {
        if (target_.diagnose) {
          CARBON_DIAGNOSTIC(ValueForRefParam, Error,
                            "value expression passed to reference parameter");
          context_.emitter().Emit(loc_id_, ValueForRefParam);
        }
        return Done{SemIR::ErrorInst::InstId};
      }

      // When initializing a C++ thunk parameter, try to pass a value "by
      // reference".
      if (target_.kind == ConversionTarget::CppThunkRef) {
        if (auto result_id = TryMakeValueAsRef(context_, expr_id);
            result_id.has_value()) {
          return Done{result_id};
        }
        // Otherwise, fall through to make a copy.
      }

      // When initializing from a value, perform a copy.
      if (target_.is_initializer() ||
          target_.kind == ConversionTarget::CppThunkRef) {
        auto copy_id = PerformCopy(context_, expr_id, target_);
        if (copy_id == SemIR::ErrorInst::InstId) {
          return Done{SemIR::ErrorInst::InstId};
        }
        return NextStep{.expr_id = copy_id,
                        .category = SemIR::GetExprCategory(sem_ir_, copy_id)};
      }

      return Done{expr_id};
  }
}

// Returns true if converting `expr_id` to `target` requires `target.type_id`
// to be complete.
static auto ConversionNeedsCompleteTarget(Context& context,
                                          SemIR::InstId expr_id,
                                          ConversionTarget target) -> bool {
  auto source_type_id = context.insts().Get(expr_id).type_id();

  // We allow conversion to incomplete facet types, since their representation
  // is fixed. This allows us to support using the `Self` of an interface inside
  // its definition.
  if (context.types().IsFacetType(target.type_id)) {
    return false;
  }

  // If the types are the same, we only have to worry about form conversions.
  if (source_type_id == target.type_id) {
    auto source_category = SemIR::GetExprCategory(context.sem_ir(), expr_id);

    // If there's no form conversion and no type conversion, the conversion is
    // a no-op, so we don't need a complete type.
    if (IsValidExprCategoryForConversionTarget(source_category, target.kind)) {
      return false;
    }
  }

  return true;
}

auto Convert(Context& context, SemIR::LocId loc_id, SemIR::InstId expr_id,
             ConversionTarget target) -> SemIR::InstId {
  auto& sem_ir = context.sem_ir();
  auto orig_expr_id = expr_id;

  // Start by making sure both sides are non-errors. If any part is an error,
  // the result is an error and we shouldn't diagnose.
  if (sem_ir.insts().Get(expr_id).type_id() == SemIR::ErrorInst::TypeId ||
      target.type_id == SemIR::ErrorInst::TypeId) {
    return SemIR::ErrorInst::InstId;
  }

  auto starting_category = SemIR::GetExprCategory(sem_ir, expr_id);
  if (starting_category == SemIR::ExprCategory::NotExpr) {
    // TODO: We currently encounter this for use of namespaces and functions.
    // We should provide a better diagnostic for inappropriate use of
    // namespace names, and allow use of functions as values.
    if (target.diagnose) {
      CARBON_DIAGNOSTIC(UseOfNonExprAsValue, Error,
                        "expression cannot be used as a value");
      context.emitter().Emit(expr_id, UseOfNonExprAsValue);
    }
    return SemIR::ErrorInst::InstId;
  }

  if (target.kind == ConversionTarget::NoOp) {
    CARBON_CHECK(target.type_id == sem_ir.insts().Get(expr_id).type_id());
    return expr_id;
  }

  // Diagnose unnecessary `ref` tags early, so that they're not obscured by
  // conversions.
  if (starting_category == SemIR::ExprCategory::RefTagged &&
      target.kind != ConversionTarget::RefParam && target.diagnose) {
    CARBON_DIAGNOSTIC(RefTagNoRefParam, Error,
                      "`ref` tag is not an argument to a `ref` parameter");
    context.emitter().Emit(expr_id, RefTagNoRefParam);
  }

  // TODO: Allow abstract but complete types if the conversion is just a
  // same-type value acqisition.
  // TODO: Push this check down to the points where we perform operations that
  // need the type to be complete.
  if (ConversionNeedsCompleteTarget(context, expr_id, target)) {
    if (target.diagnose) {
      if (!RequireConcreteType(
              context, target.type_id, loc_id,
              [&](auto& builder) {
                CARBON_CHECK(
                    !target.is_initializer(),
                    "Initialization of incomplete types is expected to be "
                    "caught elsewhere.");
                CARBON_DIAGNOSTIC(IncompleteTypeInValueConversion, Context,
                                  "forming value of incomplete type {0}",
                                  SemIR::TypeId);
                CARBON_DIAGNOSTIC(IncompleteTypeInConversion, Context,
                                  "invalid use of incomplete type {0}",
                                  SemIR::TypeId);
                builder.Context(loc_id,
                                target.kind == ConversionTarget::Value
                                    ? IncompleteTypeInValueConversion
                                    : IncompleteTypeInConversion,
                                target.type_id);
              },
              [&](auto& builder) {
                CARBON_DIAGNOSTIC(AbstractTypeInInit, Context,
                                  "initialization of abstract type {0}",
                                  SemIR::TypeId);
                builder.Context(loc_id, AbstractTypeInInit, target.type_id);
              })) {
        return SemIR::ErrorInst::InstId;
      }
    } else {
      if (!TryIsConcreteType(context, target.type_id, loc_id)) {
        return SemIR::ErrorInst::InstId;
      }
    }
  }

  // Clear storage_id in cases where it's clearly meaningless, to avoid misuse
  // and simplify the resulting SemIR.
  if (!target.is_initializer() ||
      (target.kind == ConversionTarget::Initializing &&
       SemIR::InitRepr::ForType(context.sem_ir(), target.type_id).kind ==
           SemIR::InitRepr::None)) {
    target.storage_id = SemIR::InstId::None;
  }

  // The source type doesn't need to be complete, but its completeness can
  // affect the result. For example, we don't know what type it adapts or
  // derives from unless it's complete.
  // TODO: Is there a risk of coherence problems if the source type is
  // incomplete, but a conversion would have been possible or would have behaved
  // differently if it were complete?
  TryToCompleteType(context, context.insts().Get(expr_id).type_id(), loc_id);

  // Check whether any builtin conversion applies.
  expr_id = PerformBuiltinConversion(context, loc_id, expr_id, target);
  if (expr_id == SemIR::ErrorInst::InstId) {
    return expr_id;
  }

  // Defer the action if it's dependent. We do this now rather than before
  // attempting any conversion so that we can still perform builtin conversions
  // on dependent arguments. This matters for things like converting a
  // `template T:! SomeInterface` to `type`, where it's important to form a
  // `FacetAccessType` when checking the template. But when running the action
  // later, we need to try builtin conversions again, because one may apply that
  // didn't apply in the template definition.
  // TODO: Support this for targets other than `Value`.
  if (sem_ir.insts().Get(expr_id).type_id() != target.type_id &&
      target.kind == ConversionTarget::Value) {
    auto target_type_inst_id = context.types().GetTypeInstId(target.type_id);
    SemIR::ConvertToValueAction convert_action = {
        .type_id = SemIR::InstType::TypeId,
        .inst_id = expr_id,
        .target_type_inst_id = target_type_inst_id};
    // We don't use `HandleAction` here because it would call `PerformAction`
    // inline if it's performable, which would lead to infinite recursion.
    if (!ActionIsPerformable(context, convert_action)) {
      return AddDependentActionSplice(context, loc_id, convert_action,
                                      target_type_inst_id);
    }
  }

  // If this is not a builtin conversion, try an `ImplicitAs` conversion.
  if (sem_ir.insts().Get(expr_id).type_id() != target.type_id) {
    SemIR::InstId interface_args[] = {
        context.types().GetTypeInstId(target.type_id)};
    Operator op = {
        .interface_name = GetConversionInterfaceName(target.kind),
        .interface_args_ref = interface_args,
        .op_name = CoreIdentifier::Convert,
    };
    expr_id = BuildUnaryOperator(
        context, loc_id, op, expr_id, target.diagnose, [&](auto& builder) {
          int target_kind_for_diag =
              target.kind == ConversionTarget::ExplicitAs         ? 1
              : target.kind == ConversionTarget::ExplicitUnsafeAs ? 2
                                                                  : 0;
          if (target.type_id == SemIR::TypeType::TypeId ||
              sem_ir.types().Is<SemIR::FacetType>(target.type_id)) {
            CARBON_DIAGNOSTIC(
                ConversionFailureNonTypeToFacet, Context,
                "cannot{0:=0: implicitly|:} convert non-type value of type {1} "
                "{2:to|into type implementing} {3}"
                "{0:=1: with `as`|=2: with `unsafe as`|:}",
                Diagnostics::IntAsSelect, TypeOfInstId,
                Diagnostics::BoolAsSelect, SemIR::TypeId);
            builder.Context(loc_id, ConversionFailureNonTypeToFacet,
                            target_kind_for_diag, expr_id,
                            target.type_id == SemIR::TypeType::TypeId,
                            target.type_id);
          } else {
            CARBON_DIAGNOSTIC(
                ConversionFailure, Context,
                "cannot{0:=0: implicitly|:} convert expression of type "
                "{1} to {2}{0:=1: with `as`|=2: with `unsafe as`|:}",
                Diagnostics::IntAsSelect, TypeOfInstId, SemIR::TypeId);
            builder.Context(loc_id, ConversionFailure, target_kind_for_diag,
                            expr_id, target.type_id);
          }
        });

    // Pull a value directly out of the initializer if possible and wanted.
    if (expr_id != SemIR::ErrorInst::InstId &&
        CanUseValueOfInitializer(sem_ir, target.type_id, target.kind)) {
      expr_id = AddInst<SemIR::ValueOfInitializer>(
          context, loc_id, {.type_id = target.type_id, .init_id = expr_id});
    }
  }

  // Track that we performed a type conversion, if we did so.
  if (orig_expr_id != expr_id) {
    expr_id = AddInst<SemIR::Converted>(context, loc_id,
                                        {.type_id = target.type_id,
                                         .original_id = orig_expr_id,
                                         .result_id = expr_id});
  }

  // For `as`, don't perform any value category conversions. In particular, an
  // identity conversion shouldn't change the expression category.
  if (target.is_explicit_as()) {
    return expr_id;
  }

  // Now perform any necessary value category conversions.
  expr_id = CategoryConverter(context, loc_id, target).Convert(expr_id);

  return expr_id;
}

auto InitializeExisting(Context& context, SemIR::LocId loc_id,
                        SemIR::InstId storage_id, SemIR::InstId value_id,
                        bool for_return) -> SemIR::InstId {
  auto type_id = context.insts().Get(storage_id).type_id();
  if (for_return &&
      !SemIR::InitRepr::ForType(context.sem_ir(), type_id).MightBeInPlace()) {
    // TODO: Is it safe to use storage_id when the init repr is dependent?
    storage_id = SemIR::InstId::None;
  }

  // TODO: This is only an approximation of a dominance check. Add a general
  // end-of-phase dominance check and remove the check here and the one in
  // `MergeReplacing`.
  CARBON_CHECK(!storage_id.has_value() ||
                   value_id == SemIR::ErrorInst::InstId ||
                   context.insts().GetRawIndex(storage_id) <=
                       context.insts().GetRawIndex(value_id),
               "Storage might not dominate initializer");
  PendingBlock target_block(&context);
  return Convert(context, loc_id, value_id,
                 {.kind = ConversionTarget::Initializing,
                  .type_id = type_id,
                  .storage_id = storage_id,
                  .storage_access_block = &target_block});
}

auto Initialize(Context& context, SemIR::LocId loc_id,
                SemIR::InstId&& storage_id, PendingBlock&& storage_access_block,
                SemIR::InstId value_id) -> InitializeResult {
  CARBON_CHECK(storage_id.has_value());
  auto type_id = context.insts().Get(storage_id).type_id();
  auto result_id = Convert(context, loc_id, value_id,
                           {.kind = ConversionTarget::Initializing,
                            .type_id = type_id,
                            .storage_id = storage_id,
                            .storage_access_block = &storage_access_block});

  // Insert the storage block now, in case it wasn't used by the initializer.
  storage_access_block.InsertHere();
  if (result_id == SemIR::ErrorInst::InstId) {
    return {.storage_id = SemIR::ErrorInst::InstId,
            .init_id = SemIR::ErrorInst::InstId};
  }

  // Find the storage argument. If the storage block was spliced or written over
  // an existing storage argument by `Convert`, the resulting expression will
  // have a storage argument that points to the possibly-rewritten storage
  // instruction, and we can use that. Otherwise, the storage access block will
  // have been inserted above, and we can use `storage_id` unchanged.
  auto storage_arg_id =
      SemIR::FindStorageArgForInitializer(context.sem_ir(), result_id);
  return {
      .storage_id = storage_arg_id.has_value() ? storage_arg_id : storage_id,
      .init_id = result_id};
}

auto ConvertToValueExpr(Context& context, SemIR::InstId expr_id)
    -> SemIR::InstId {
  return Convert(context, SemIR::LocId(expr_id), expr_id,
                 {.kind = ConversionTarget::Value,
                  .type_id = context.insts().Get(expr_id).type_id()});
}

auto ConvertToValueOrRefExpr(Context& context, SemIR::InstId expr_id)
    -> SemIR::InstId {
  return Convert(context, SemIR::LocId(expr_id), expr_id,
                 {.kind = ConversionTarget::ValueOrRef,
                  .type_id = context.insts().Get(expr_id).type_id()});
}

auto ConvertToValueOfType(Context& context, SemIR::LocId loc_id,
                          SemIR::InstId expr_id, SemIR::TypeId type_id,
                          bool diagnose) -> SemIR::InstId {
  return Convert(context, loc_id, expr_id,
                 {.kind = ConversionTarget::Value,
                  .type_id = type_id,
                  .diagnose = diagnose});
}

auto ConvertToValueOrRefOfType(Context& context, SemIR::LocId loc_id,
                               SemIR::InstId expr_id, SemIR::TypeId type_id)
    -> SemIR::InstId {
  return Convert(context, loc_id, expr_id,
                 {.kind = ConversionTarget::ValueOrRef, .type_id = type_id});
}

// Like ConvertToValueOfType but failure to convert does not result in
// diagnostics. An ErrorInst instruction is still returned on failure.
auto TryConvertToValueOfType(Context& context, SemIR::LocId loc_id,
                             SemIR::InstId expr_id, SemIR::TypeId type_id)
    -> SemIR::InstId {
  return Convert(
      context, loc_id, expr_id,
      {.kind = ConversionTarget::Value, .type_id = type_id, .diagnose = false});
}

auto ConvertToBoolValue(Context& context, SemIR::LocId loc_id,
                        SemIR::InstId value_id) -> SemIR::InstId {
  return ConvertToValueOfType(
      context, loc_id, value_id,
      GetSingletonType(context, SemIR::BoolType::TypeInstId));
}

auto ConvertForExplicitAs(Context& context, Parse::NodeId as_node,
                          SemIR::InstId value_id, SemIR::TypeId type_id,
                          bool unsafe) -> SemIR::InstId {
  return Convert(context, as_node, value_id,
                 {.kind = unsafe ? ConversionTarget::ExplicitUnsafeAs
                                 : ConversionTarget::ExplicitAs,
                  .type_id = type_id});
}

// TODO: Consider moving this to pattern_match.h.
auto ConvertCallArgs(Context& context, SemIR::LocId call_loc_id,
                     SemIR::InstId self_id,
                     llvm::ArrayRef<SemIR::InstId> arg_refs,
                     SemIR::InstId return_arg_id, const SemIR::Function& callee,
                     SemIR::SpecificId callee_specific_id,
                     bool is_operator_syntax) -> SemIR::InstBlockId {
  auto param_patterns =
      context.inst_blocks().GetOrEmpty(callee.param_patterns_id);
  auto return_pattern_id = callee.return_pattern_id;

  // The caller should have ensured this callee has the right arity.
  CARBON_CHECK(arg_refs.size() == param_patterns.size());

  if (callee.self_param_id.has_value() && !self_id.has_value()) {
    CARBON_DIAGNOSTIC(MissingObjectInMethodCall, Error,
                      "missing object argument in method call");
    CARBON_DIAGNOSTIC(InCallToFunction, Note, "calling function declared here");
    context.emitter()
        .Build(call_loc_id, MissingObjectInMethodCall)
        .Note(callee.latest_decl_id(), InCallToFunction)
        .Emit();
    self_id = SemIR::ErrorInst::InstId;
  }

  return CallerPatternMatch(context, callee_specific_id, callee.self_param_id,
                            callee.param_patterns_id, return_pattern_id,
                            self_id, arg_refs, return_arg_id,
                            is_operator_syntax);
}

auto TypeExpr::ForUnsugared(Context& context, SemIR::TypeId type_id)
    -> TypeExpr {
  return {.inst_id = context.types().GetTypeInstId(type_id),
          .type_id = type_id};
}

static auto DiagnoseTypeExprEvaluationFailure(Context& context,
                                              SemIR::LocId loc_id) -> void {
  CARBON_DIAGNOSTIC(TypeExprEvaluationFailure, Error,
                    "cannot evaluate type expression");
  context.emitter().Emit(loc_id, TypeExprEvaluationFailure);
}

auto ExprAsType(Context& context, SemIR::LocId loc_id, SemIR::InstId value_id,
                bool diagnose) -> TypeExpr {
  auto type_as_inst_id = ConvertToValueOfType(
      context, loc_id, value_id, SemIR::TypeType::TypeId, diagnose);
  if (type_as_inst_id == SemIR::ErrorInst::InstId) {
    return {.inst_id = SemIR::ErrorInst::TypeInstId,
            .type_id = SemIR::ErrorInst::TypeId};
  }

  auto type_as_const_id = context.constant_values().Get(type_as_inst_id);
  if (!type_as_const_id.is_constant()) {
    if (diagnose) {
      DiagnoseTypeExprEvaluationFailure(context, loc_id);
    }
    return {.inst_id = SemIR::ErrorInst::TypeInstId,
            .type_id = SemIR::ErrorInst::TypeId};
  }

  return {
      .inst_id = context.types().GetAsTypeInstId(type_as_inst_id),
      .type_id = context.types().GetTypeIdForTypeConstantId(type_as_const_id)};
}

auto FormExprAsForm(Context& context, SemIR::LocId loc_id,
                    SemIR::InstId value_id) -> Context::FormExpr {
  auto form_inst_id =
      ConvertToValueOfType(context, loc_id, value_id, SemIR::FormType::TypeId);
  if (form_inst_id == SemIR::ErrorInst::InstId) {
    return Context::FormExpr::Error;
  }

  form_inst_id = HandleAction<SemIR::RefineFormAction>(
      context, loc_id, SemIR::FormType::TypeInstId,
      {.type_id = SemIR::InstType::TypeId, .form_id = form_inst_id});

  auto form_const_id = context.constant_values().Get(form_inst_id);
  if (!form_const_id.is_constant()) {
    CARBON_DIAGNOSTIC(FormExprEvaluationFailure, Error,
                      "cannot evaluate form expression");
    context.emitter().Emit(loc_id, FormExprEvaluationFailure);
    return Context::FormExpr::Error;
  }

  auto type_id = GetTypeComponent(context, form_inst_id);
  auto type_inst_id = context.types().GetTypeInstId(type_id);
  return {.form_inst_id = form_inst_id,
          .type_component_inst_id = type_inst_id,
          .type_component_id = type_id};
}

auto ReturnExprAsForm(Context& context, SemIR::LocId loc_id,
                      SemIR::InstId value_id) -> Context::FormExpr {
  auto form_inst_id = SemIR::InstId::None;
  auto type_inst_id = SemIR::InstId::None;
  if (auto ref_tag = context.insts().TryGetAs<SemIR::RefTagExpr>(value_id)) {
    type_inst_id = ConvertToValueOfType(context, loc_id, ref_tag->expr_id,
                                        SemIR::TypeType::TypeId);
    if (type_inst_id == SemIR::ErrorInst::InstId) {
      return Context::FormExpr::Error;
    }
    if (!context.constant_values().Get(type_inst_id).is_constant()) {
      DiagnoseTypeExprEvaluationFailure(context,
                                        SemIR::LocId(ref_tag->expr_id));
      return Context::FormExpr::Error;
    }
    form_inst_id = AddInst(
        context,
        SemIR::LocIdAndInst::RuntimeVerified(
            context.sem_ir(), loc_id,
            SemIR::RefForm{.type_id = SemIR::FormType::TypeId,
                           .type_component_inst_id =
                               context.types().GetAsTypeInstId(type_inst_id)}));
  } else {
    type_inst_id = ConvertToValueOfType(context, loc_id, value_id,
                                        SemIR::TypeType::TypeId);
    if (type_inst_id == SemIR::ErrorInst::InstId) {
      return Context::FormExpr::Error;
    }
    if (!context.constant_values().Get(type_inst_id).is_constant()) {
      DiagnoseTypeExprEvaluationFailure(context, loc_id);
      return Context::FormExpr::Error;
    }
    form_inst_id = AddInst(
        context, SemIR::LocIdAndInst::RuntimeVerified(
                     context.sem_ir(), loc_id,
                     SemIR::InitForm{
                         .type_id = SemIR::FormType::TypeId,
                         .type_component_inst_id =
                             context.types().GetAsTypeInstId(type_inst_id)}));
  }

  auto type_const_id = context.constant_values().Get(type_inst_id);
  CARBON_CHECK(type_const_id.is_constant());

  return {
      .form_inst_id = form_inst_id,
      .type_component_inst_id = context.types().GetAsTypeInstId(type_inst_id),
      .type_component_id =
          context.types().GetTypeIdForTypeConstantId(type_const_id)};
}

auto DiscardExpr(Context& context, SemIR::InstId expr_id) -> void {
  // If we discard an initializing expression, convert it to a value or
  // reference so that it has something to initialize.
  auto expr = context.insts().Get(expr_id);
  Convert(context, SemIR::LocId(expr_id), expr_id,
          {.kind = ConversionTarget::Discarded, .type_id = expr.type_id()});

  // TODO: This will eventually need to do some "do not discard" analysis.
}

}  // namespace Carbon::Check

// NOLINTEND(misc-no-recursion)

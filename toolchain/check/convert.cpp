// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/convert.h"

#include <string>
#include <utility>

#include "common/check.h"
#include "common/map.h"
#include "llvm/ADT/STLExtras.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/impl_lookup.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/operator.h"
#include "toolchain/check/pattern_match.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/sem_ir/copy_on_write_block.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

// TODO: This contains a lot of recursion. Consider removing it in order to
// prevent accidents.
// NOLINTBEGIN(misc-no-recursion)

namespace Carbon::Check {

// Given an initializing expression, find its return slot argument. Returns
// `None` if there is no return slot, because the initialization is not
// performed in place.
static auto FindReturnSlotArgForInitializer(SemIR::File& sem_ir,
                                            SemIR::InstId init_id)
    -> SemIR::InstId {
  while (true) {
    SemIR::Inst init_untyped = sem_ir.insts().Get(init_id);
    CARBON_KIND_SWITCH(init_untyped) {
      case CARBON_KIND(SemIR::AsCompatible init): {
        init_id = init.source_id;
        continue;
      }
      case CARBON_KIND(SemIR::Converted init): {
        init_id = init.result_id;
        continue;
      }
      case CARBON_KIND(SemIR::ArrayInit init): {
        return init.dest_id;
      }
      case CARBON_KIND(SemIR::ClassInit init): {
        return init.dest_id;
      }
      case CARBON_KIND(SemIR::StructInit init): {
        return init.dest_id;
      }
      case CARBON_KIND(SemIR::TupleInit init): {
        return init.dest_id;
      }
      case CARBON_KIND(SemIR::InitializeFrom init): {
        return init.dest_id;
      }
      case CARBON_KIND(SemIR::Call call): {
        if (!SemIR::ReturnTypeInfo::ForType(sem_ir, call.type_id)
                 .has_return_slot()) {
          return SemIR::InstId::None;
        }
        if (!call.args_id.has_value()) {
          // Argument initialization failed, so we have no return slot.
          return SemIR::InstId::None;
        }
        return sem_ir.inst_blocks().Get(call.args_id).back();
      }
      default:
        CARBON_FATAL("Initialization from unexpected inst {0}", init_untyped);
    }
  }
}

// Marks the initializer `init_id` as initializing `target_id`.
static auto MarkInitializerFor(SemIR::File& sem_ir, SemIR::InstId init_id,
                               SemIR::InstId target_id,
                               PendingBlock& target_block) -> void {
  auto return_slot_arg_id = FindReturnSlotArgForInitializer(sem_ir, init_id);
  if (return_slot_arg_id.has_value()) {
    // Replace the temporary in the return slot with a reference to our target.
    CARBON_CHECK(sem_ir.insts().Get(return_slot_arg_id).kind() ==
                     SemIR::TemporaryStorage::Kind,
                 "Return slot for initializer does not contain a temporary; "
                 "initialized multiple times? Have {0}",
                 sem_ir.insts().Get(return_slot_arg_id));
    target_block.MergeReplacing(return_slot_arg_id, target_id);
  }
}

// Commits to using a temporary to store the result of the initializing
// expression described by `init_id`, and returns the location of the
// temporary. If `discarded` is `true`, the result is discarded, and no
// temporary will be created if possible; if no temporary is created, the
// return value will be `SemIR::InstId::None`.
static auto FinalizeTemporary(Context& context, SemIR::InstId init_id,
                              bool discarded) -> SemIR::InstId {
  auto& sem_ir = context.sem_ir();
  auto return_slot_arg_id = FindReturnSlotArgForInitializer(sem_ir, init_id);
  if (return_slot_arg_id.has_value()) {
    // The return slot should already have a materialized temporary in it.
    CARBON_CHECK(sem_ir.insts().Get(return_slot_arg_id).kind() ==
                     SemIR::TemporaryStorage::Kind,
                 "Return slot for initializer does not contain a temporary; "
                 "initialized multiple times? Have {0}",
                 sem_ir.insts().Get(return_slot_arg_id));
    auto init = sem_ir.insts().Get(init_id);
    return AddInst<SemIR::Temporary>(context, sem_ir.insts().GetLocId(init_id),
                                     {.type_id = init.type_id(),
                                      .storage_id = return_slot_arg_id,
                                      .init_id = init_id});
  }

  if (discarded) {
    // Don't invent a temporary that we're going to discard.
    return SemIR::InstId::None;
  }

  // The initializer has no return slot, but we want to produce a temporary
  // object. Materialize one now.
  // TODO: Consider using `None` to mean that we immediately materialize and
  // initialize a temporary, rather than two separate instructions.
  auto init = sem_ir.insts().Get(init_id);
  auto loc_id = sem_ir.insts().GetLocId(init_id);
  auto temporary_id = AddInst<SemIR::TemporaryStorage>(
      context, loc_id, {.type_id = init.type_id()});
  return AddInst<SemIR::Temporary>(context, loc_id,
                                   {.type_id = init.type_id(),
                                    .storage_id = temporary_id,
                                    .init_id = init_id});
}

// Materialize a temporary to hold the result of the given expression if it is
// an initializing expression.
static auto MaterializeIfInitializing(Context& context, SemIR::InstId expr_id)
    -> SemIR::InstId {
  if (GetExprCategory(context.sem_ir(), expr_id) ==
      SemIR::ExprCategory::Initializing) {
    return FinalizeTemporary(context, expr_id, /*discarded=*/false);
  }
  return expr_id;
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
  if constexpr (std::is_same_v<AccessInstT, SemIR::ArrayIndex>) {
    // TODO: Add a new instruction kind for indexing an array at a constant
    // index so that we don't need an integer literal instruction here, and
    // remove this special case.
    auto index_id = block.template AddInst<SemIR::IntValue>(
        loc_id, {.type_id = GetSingletonType(
                     context, SemIR::IntLiteralType::SingletonInstId),
                 .int_id = context.ints().Add(static_cast<int64_t>(i))});
    return AddInst<AccessInstT>(block, loc_id,
                                {elem_type_id, aggregate_id, index_id});
  } else {
    return AddInst<AccessInstT>(
        block, loc_id, {elem_type_id, aggregate_id, SemIR::ElementIndex(i)});
  }
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
    SemIR::TypeId src_elem_type,
    llvm::ArrayRef<SemIR::InstId> src_literal_elems,
    ConversionTarget::Kind kind, SemIR::InstId target_id,
    SemIR::TypeId target_elem_type, PendingBlock* target_block,
    size_t src_field_index, size_t target_field_index) -> SemIR::InstId {
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
  target.init_block = target_block;
  target.init_id = MakeElementAccessInst<TargetAccessInstT>(
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
  auto tuple_elem_types = sem_ir.type_blocks().Get(tuple_type.elements_id);

  auto value = sem_ir.insts().Get(value_id);
  auto value_loc_id = sem_ir.insts().GetLocId(value_id);

  // If we're initializing from a tuple literal, we will use its elements
  // directly. Otherwise, materialize a temporary if needed and index into the
  // result.
  llvm::ArrayRef<SemIR::InstId> literal_elems;
  if (auto tuple_literal = value.TryAs<SemIR::TupleLiteral>()) {
    literal_elems = sem_ir.inst_blocks().Get(tuple_literal->elements_id);
  } else {
    value_id = MaterializeIfInitializing(context, value_id);
  }

  // Check that the tuple is the right size.
  std::optional<uint64_t> array_bound =
      sem_ir.GetArrayBoundValue(array_type.bound_id);
  if (!array_bound) {
    // TODO: Should this fall back to using `ImplicitAs`?
    if (target.diagnose) {
      CARBON_DIAGNOSTIC(ArrayInitDependentBound, Error,
                        "cannot initialize array with dependent bound from a "
                        "list of initializers");
      context.emitter().Emit(value_loc_id, ArrayInitDependentBound);
    }
    return SemIR::ErrorInst::SingletonInstId;
  }
  if (tuple_elem_types.size() != array_bound) {
    if (target.diagnose) {
      CARBON_DIAGNOSTIC(ArrayInitFromLiteralArgCountMismatch, Error,
                        "cannot initialize array of {0} element{0:s} from {1} "
                        "initializer{1:s}",
                        IntAsSelect, IntAsSelect);
      CARBON_DIAGNOSTIC(
          ArrayInitFromExprArgCountMismatch, Error,
          "cannot initialize array of {0} element{0:s} from tuple "
          "with {1} element{1:s}",
          IntAsSelect, IntAsSelect);
      context.emitter().Emit(value_loc_id,
                             literal_elems.empty()
                                 ? ArrayInitFromExprArgCountMismatch
                                 : ArrayInitFromLiteralArgCountMismatch,
                             *array_bound, tuple_elem_types.size());
    }
    return SemIR::ErrorInst::SingletonInstId;
  }

  PendingBlock target_block_storage(context);
  PendingBlock* target_block =
      target.init_block ? target.init_block : &target_block_storage;

  // Arrays are always initialized in-place. Allocate a temporary as the
  // destination for the array initialization if we weren't given one.
  SemIR::InstId return_slot_arg_id = target.init_id;
  if (!target.init_id.has_value()) {
    return_slot_arg_id = target_block->AddInst<SemIR::TemporaryStorage>(
        value_loc_id, {.type_id = target.type_id});
  }

  // Initialize each element of the array from the corresponding element of the
  // tuple.
  // TODO: Annotate diagnostics coming from here with the array element index,
  // if initializing from a tuple literal.
  llvm::SmallVector<SemIR::InstId> inits;
  inits.reserve(*array_bound + 1);
  for (auto [i, src_type_id] : llvm::enumerate(tuple_elem_types)) {
    // TODO: This call recurses back into conversion. Switch to an iterative
    // approach.
    auto init_id =
        ConvertAggregateElement<SemIR::TupleAccess, SemIR::ArrayIndex>(
            context, value_loc_id, value_id, src_type_id, literal_elems,
            ConversionTarget::FullInitializer, return_slot_arg_id,
            array_type.element_type_id, target_block, i, i);
    if (init_id == SemIR::ErrorInst::SingletonInstId) {
      return SemIR::ErrorInst::SingletonInstId;
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
  auto src_elem_types = sem_ir.type_blocks().Get(src_type.elements_id);
  auto dest_elem_types = sem_ir.type_blocks().Get(dest_type.elements_id);

  auto value = sem_ir.insts().Get(value_id);
  auto value_loc_id = sem_ir.insts().GetLocId(value_id);

  // If we're initializing from a tuple literal, we will use its elements
  // directly. Otherwise, materialize a temporary if needed and index into the
  // result.
  llvm::ArrayRef<SemIR::InstId> literal_elems;
  auto literal_elems_id = SemIR::InstBlockId::None;
  if (auto tuple_literal = value.TryAs<SemIR::TupleLiteral>()) {
    literal_elems_id = tuple_literal->elements_id;
    literal_elems = sem_ir.inst_blocks().Get(literal_elems_id);
  } else {
    value_id = MaterializeIfInitializing(context, value_id);
  }

  // Check that the tuples are the same size.
  if (src_elem_types.size() != dest_elem_types.size()) {
    if (target.diagnose) {
      CARBON_DIAGNOSTIC(
          TupleInitElementCountMismatch, Error,
          "cannot initialize tuple of {0} element{0:s} from tuple "
          "with {1} element{1:s}",
          IntAsSelect, IntAsSelect);
      context.emitter().Emit(value_loc_id, TupleInitElementCountMismatch,
                             dest_elem_types.size(), src_elem_types.size());
    }
    return SemIR::ErrorInst::SingletonInstId;
  }

  // If we're forming an initializer, then we want an initializer for each
  // element. Otherwise, we want a value representation for each element.
  // Perform a final destination store if we're performing an in-place
  // initialization.
  bool is_init = target.is_initializer();
  ConversionTarget::Kind inner_kind =
      !is_init ? ConversionTarget::Value
      : SemIR::InitRepr::ForType(sem_ir, target.type_id).kind ==
              SemIR::InitRepr::InPlace
          ? ConversionTarget::FullInitializer
          : ConversionTarget::Initializer;

  // Initialize each element of the destination from the corresponding element
  // of the source.
  // TODO: Annotate diagnostics coming from here with the element index.
  auto new_block =
      literal_elems_id.has_value()
          ? SemIR::CopyOnWriteInstBlock(sem_ir, literal_elems_id)
          : SemIR::CopyOnWriteInstBlock(
                sem_ir, SemIR::CopyOnWriteInstBlock::UninitializedBlock{
                            src_elem_types.size()});
  for (auto [i, src_type_id, dest_type_id] :
       llvm::enumerate(src_elem_types, dest_elem_types)) {
    // TODO: This call recurses back into conversion. Switch to an iterative
    // approach.
    auto init_id =
        ConvertAggregateElement<SemIR::TupleAccess, SemIR::TupleAccess>(
            context, value_loc_id, value_id, src_type_id, literal_elems,
            inner_kind, target.init_id, dest_type_id, target.init_block, i, i);
    if (init_id == SemIR::ErrorInst::SingletonInstId) {
      return SemIR::ErrorInst::SingletonInstId;
    }
    new_block.Set(i, init_id);
  }

  if (is_init) {
    target.init_block->InsertHere();
    return AddInst<SemIR::TupleInit>(context, value_loc_id,
                                     {.type_id = target.type_id,
                                      .elements_id = new_block.id(),
                                      .dest_id = target.init_id});
  } else {
    return AddInst<SemIR::TupleValue>(
        context, value_loc_id,
        {.type_id = target.type_id, .elements_id = new_block.id()});
  }
}

// Common implementation for ConvertStructToStruct and ConvertStructToClass.
template <typename TargetAccessInstT>
static auto ConvertStructToStructOrClass(Context& context,
                                         SemIR::StructType src_type,
                                         SemIR::StructType dest_type,
                                         SemIR::InstId value_id,
                                         ConversionTarget target)
    -> SemIR::InstId {
  static_assert(std::is_same_v<SemIR::ClassElementAccess, TargetAccessInstT> ||
                std::is_same_v<SemIR::StructAccess, TargetAccessInstT>);
  constexpr bool ToClass =
      std::is_same_v<SemIR::ClassElementAccess, TargetAccessInstT>;

  auto& sem_ir = context.sem_ir();
  auto src_elem_fields = sem_ir.struct_type_fields().Get(src_type.fields_id);
  auto dest_elem_fields = sem_ir.struct_type_fields().Get(dest_type.fields_id);
  bool dest_has_vptr = !dest_elem_fields.empty() &&
                       dest_elem_fields.front().name_id == SemIR::NameId::Vptr;
  int dest_vptr_offset = (dest_has_vptr ? 1 : 0);
  auto dest_elem_fields_size = dest_elem_fields.size() - dest_vptr_offset;

  auto value = sem_ir.insts().Get(value_id);
  auto value_loc_id = sem_ir.insts().GetLocId(value_id);

  // If we're initializing from a struct literal, we will use its elements
  // directly. Otherwise, materialize a temporary if needed and index into the
  // result.
  llvm::ArrayRef<SemIR::InstId> literal_elems;
  auto literal_elems_id = SemIR::InstBlockId::None;
  if (auto struct_literal = value.TryAs<SemIR::StructLiteral>()) {
    literal_elems_id = struct_literal->elements_id;
    literal_elems = sem_ir.inst_blocks().Get(literal_elems_id);
  } else {
    value_id = MaterializeIfInitializing(context, value_id);
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
          BoolAsSelect, IntAsSelect, IntAsSelect);
      context.emitter().Emit(value_loc_id, StructInitElementCountMismatch,
                             ToClass, dest_elem_fields_size,
                             src_elem_fields.size());
    }
    return SemIR::ErrorInst::SingletonInstId;
  }

  // Prepare to look up fields in the source by index.
  Map<SemIR::NameId, int32_t> src_field_indexes;
  if (src_type.fields_id != dest_type.fields_id) {
    for (auto [i, field] : llvm::enumerate(src_elem_fields)) {
      auto result = src_field_indexes.Insert(field.name_id, i);
      CARBON_CHECK(result.is_inserted(), "Duplicate field in source structure");
    }
  }

  // If we're forming an initializer, then we want an initializer for each
  // element. Otherwise, we want a value representation for each element.
  // Perform a final destination store if we're performing an in-place
  // initialization.
  bool is_init = target.is_initializer();
  ConversionTarget::Kind inner_kind =
      !is_init ? ConversionTarget::Value
      : SemIR::InitRepr::ForType(sem_ir, target.type_id).kind ==
              SemIR::InitRepr::InPlace
          ? ConversionTarget::FullInitializer
          : ConversionTarget::Initializer;

  // Initialize each element of the destination from the corresponding element
  // of the source.
  // TODO: Annotate diagnostics coming from here with the element index.
  auto new_block =
      literal_elems_id.has_value() && !dest_has_vptr
          ? SemIR::CopyOnWriteInstBlock(sem_ir, literal_elems_id)
          : SemIR::CopyOnWriteInstBlock(
                sem_ir, SemIR::CopyOnWriteInstBlock::UninitializedBlock{
                            dest_elem_fields.size()});
  for (auto [i, dest_field] : llvm::enumerate(dest_elem_fields)) {
    if (dest_field.name_id == SemIR::NameId::Vptr) {
      // CARBON_CHECK(ToClass, "Only classes should have vptrs.");
      auto dest_id =
          AddInst<SemIR::ClassElementAccess>(context, value_loc_id,
                                             {.type_id = dest_field.type_id,
                                              .base_id = target.init_id,
                                              .index = SemIR::ElementIndex(i)});
      auto vtable_ptr_id = AddInst<SemIR::VtablePtr>(
          context, value_loc_id, {.type_id = dest_field.type_id});
      auto init_id =
          AddInst<SemIR::InitializeFrom>(context, value_loc_id,
                                         {.type_id = dest_field.type_id,
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
        return SemIR::ErrorInst::SingletonInstId;
      }
    }
    auto src_field = src_elem_fields[src_field_index];

    // TODO: This call recurses back into conversion. Switch to an iterative
    // approach.
    auto init_id =
        ConvertAggregateElement<SemIR::StructAccess, TargetAccessInstT>(
            context, value_loc_id, value_id, src_field.type_id, literal_elems,
            inner_kind, target.init_id, dest_field.type_id, target.init_block,
            src_field_index, src_field_index + dest_vptr_offset);
    if (init_id == SemIR::ErrorInst::SingletonInstId) {
      return SemIR::ErrorInst::SingletonInstId;
    }
    new_block.Set(i, init_id);
  }

  if (ToClass) {
    target.init_block->InsertHere();
    CARBON_CHECK(is_init,
                 "Converting directly to a class value is not supported");
    return AddInst<SemIR::ClassInit>(context, value_loc_id,
                                     {.type_id = target.type_id,
                                      .elements_id = new_block.id(),
                                      .dest_id = target.init_id});
  } else if (is_init) {
    target.init_block->InsertHere();
    return AddInst<SemIR::StructInit>(context, value_loc_id,
                                      {.type_id = target.type_id,
                                       .elements_id = new_block.id(),
                                       .dest_id = target.init_id});
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
                                 ConversionTarget target) -> SemIR::InstId {
  PendingBlock target_block(context);
  auto& dest_class_info = context.classes().Get(dest_type.class_id);
  CARBON_CHECK(dest_class_info.inheritance_kind != SemIR::Class::Abstract);
  auto object_repr_id =
      dest_class_info.GetObjectRepr(context.sem_ir(), dest_type.specific_id);
  if (object_repr_id == SemIR::ErrorInst::SingletonTypeId) {
    return SemIR::ErrorInst::SingletonInstId;
  }
  auto dest_struct_type =
      context.types().GetAs<SemIR::StructType>(object_repr_id);

  // If we're trying to create a class value, form a temporary for the value to
  // point to.
  bool need_temporary = !target.is_initializer();
  if (need_temporary) {
    target.kind = ConversionTarget::Initializer;
    target.init_block = &target_block;
    target.init_id = target_block.AddInst<SemIR::TemporaryStorage>(
        context.insts().GetLocId(value_id), {.type_id = target.type_id});
  }

  auto result_id = ConvertStructToStructOrClass<SemIR::ClassElementAccess>(
      context, src_type, dest_struct_type, value_id, target);

  if (need_temporary) {
    target_block.InsertHere();
    result_id =
        AddInst<SemIR::Temporary>(context, context.insts().GetLocId(value_id),
                                  {.type_id = target.type_id,
                                   .storage_id = target.init_id,
                                   .init_id = result_id});
  }
  return result_id;
}

// An inheritance path is a sequence of `BaseDecl`s and corresponding base types
// in order from derived to base.
using InheritancePath =
    llvm::SmallVector<std::pair<SemIR::InstId, SemIR::TypeId>>;

// Computes the inheritance path from class `derived_id` to class `base_id`.
// Returns nullopt if `derived_id` is not a class derived from `base_id`.
static auto ComputeInheritancePath(Context& context, SemIRLoc loc,
                                   SemIR::TypeId derived_id,
                                   SemIR::TypeId base_id)
    -> std::optional<InheritancePath> {
  // We intend for NRVO to be applied to `result`. All `return` statements in
  // this function should `return result;`.
  std::optional<InheritancePath> result(std::in_place);
  if (!TryToCompleteType(context, derived_id, loc)) {
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

  // Add a series of `.base` accesses.
  for (auto [base_id, base_type_id] : path) {
    auto base_decl = context.insts().GetAs<SemIR::BaseDecl>(base_id);
    value_id = AddInst<SemIR::ClassElementAccess>(context, loc_id,
                                                  {.type_id = base_type_id,
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
  // Form `*p`.
  ptr_id = ConvertToValueExpr(context, ptr_id);
  auto ref_id = AddInst<SemIR::Deref>(
      context, loc_id,
      {.type_id = src_ptr_type.pointee_id, .pointer_id = ptr_id});

  // Convert as a reference expression.
  ref_id = ConvertDerivedToBase(context, loc_id, ref_id, path);

  // Take the address.
  return AddInst<SemIR::AddrOf>(
      context, loc_id, {.type_id = dest_ptr_type_id, .lvalue_id = ref_id});
}

// Returns whether `category` is a valid expression category to produce as a
// result of a conversion with kind `target_kind`, or at most needs a temporary
// to be materialized.
static auto IsValidExprCategoryForConversionTarget(
    SemIR::ExprCategory category, ConversionTarget::Kind target_kind) -> bool {
  switch (target_kind) {
    case ConversionTarget::Value:
      return category == SemIR::ExprCategory::Value;
    case ConversionTarget::ValueOrRef:
    case ConversionTarget::Discarded:
      return category == SemIR::ExprCategory::Value ||
             category == SemIR::ExprCategory::DurableRef ||
             category == SemIR::ExprCategory::EphemeralRef ||
             category == SemIR::ExprCategory::Initializing;
    case ConversionTarget::ExplicitAs:
      return true;
    case ConversionTarget::Initializer:
    case ConversionTarget::FullInitializer:
      return category == SemIR::ExprCategory::Initializing;
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

// Returns the non-adapter type that is compatible with the specified type.
static auto GetTransitiveAdaptedType(Context& context, SemIR::TypeId type_id)
    -> SemIR::TypeId {
  // If the type is an adapter, its object representation type is its compatible
  // non-adapter type.
  while (auto class_type =
             context.types().TryGetAs<SemIR::ClassType>(type_id)) {
    auto& class_info = context.classes().Get(class_type->class_id);
    auto adapted_type_id =
        class_info.GetAdaptedType(context.sem_ir(), class_type->specific_id);
    if (!adapted_type_id.has_value()) {
      break;
    }
    type_id = adapted_type_id;
  }

  // Otherwise, the type itself is a non-adapter type.
  return type_id;
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
    if (value_cat == SemIR::ExprCategory::Initializing &&
        CanUseValueOfInitializer(sem_ir, value_type_id, target.kind)) {
      return AddInst<SemIR::ValueOfInitializer>(
          context, loc_id, {.type_id = value_type_id, .init_id = value_id});
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
            GetTransitiveAdaptedType(context, value_type_id);
        foundation_type_id != value_type_id &&
        (context.types().Is<SemIR::TupleType>(foundation_type_id) ||
         context.types().Is<SemIR::StructType>(foundation_type_id))) {
      auto foundation_value_id = AddInst<SemIR::AsCompatible>(
          context, loc_id,
          {.type_id = foundation_type_id, .source_id = value_id});

      auto foundation_init_id = target.init_id;
      if (foundation_init_id != SemIR::InstId::None) {
        foundation_init_id = target.init_block->AddInst<SemIR::AsCompatible>(
            loc_id,
            {.type_id = foundation_type_id, .source_id = target.init_id});
      }

      {
        // While the types are the same, the conversion can still fail if it
        // performs a copy while converting the value to another category, and
        // the type (or some part of it) is not copyable.
        DiagnosticAnnotationScope annotate_diagnostics(
            &context.emitter(), [&](auto& builder) {
              CARBON_DIAGNOSTIC(InCopy, Note, "in copy of {0}", TypeOfInstId);
              builder.Note(value_id, InCopy, value_id);
            });

        foundation_value_id =
            PerformBuiltinConversion(context, loc_id, foundation_value_id,
                                     {.kind = target.kind,
                                      .type_id = foundation_type_id,
                                      .init_id = foundation_init_id,
                                      .init_block = target.init_block,
                                      .diagnose = target.diagnose});
        if (foundation_value_id == SemIR::ErrorInst::SingletonInstId) {
          return SemIR::ErrorInst::SingletonInstId;
        }
      }

      return AddInst<SemIR::AsCompatible>(
          context, loc_id,
          {.type_id = target.type_id, .source_id = foundation_value_id});
    }
  }

  // T explicitly converts to U if T is compatible with U.
  if (target.kind == ConversionTarget::Kind::ExplicitAs &&
      target.type_id != value_type_id) {
    auto target_foundation_id =
        GetTransitiveAdaptedType(context, target.type_id);
    auto value_foundation_id = GetTransitiveAdaptedType(context, value_type_id);
    if (target_foundation_id == value_foundation_id) {
      // For a struct or tuple literal, perform a category conversion if
      // necessary.
      if (SemIR::GetExprCategory(context.sem_ir(), value_id) ==
          SemIR::ExprCategory::Mixed) {
        value_id = PerformBuiltinConversion(context, loc_id, value_id,
                                            {.kind = ConversionTarget::Value,
                                             .type_id = value_type_id,
                                             .diagnose = target.diagnose});
      }
      return AddInst<SemIR::AsCompatible>(
          context, loc_id, {.type_id = target.type_id, .source_id = value_id});
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

  // A tuple (T1, T2, ..., Tn) converts to array(T, n) if each Ti converts to T.
  if (auto target_array_type = target_type_inst.TryAs<SemIR::ArrayType>()) {
    if (auto src_tuple_type =
            sem_ir.types().TryGetAs<SemIR::TupleType>(value_type_id)) {
      return ConvertTupleToArray(context, *src_tuple_type, *target_array_type,
                                 value_id, target);
    }
  }

  // A struct {.f_1: T_1, .f_2: T_2, ..., .f_n: T_n} converts to a class type
  // if it converts to the struct type that is the class's representation type
  // (a struct with the same fields as the class, plus a base field where
  // relevant).
  if (auto target_class_type = target_type_inst.TryAs<SemIR::ClassType>()) {
    if (auto src_struct_type =
            sem_ir.types().TryGetAs<SemIR::StructType>(value_type_id)) {
      if (!context.classes()
               .Get(target_class_type->class_id)
               .adapt_id.has_value()) {
        return ConvertStructToClass(context, *src_struct_type,
                                    *target_class_type, value_id, target);
      }
    }

    // An expression of type T converts to U if T is a class derived from U.
    if (auto path = ComputeInheritancePath(context, loc_id, value_type_id,
                                           target.type_id);
        path && !path->empty()) {
      return ConvertDerivedToBase(context, loc_id, value_id, *path);
    }
  }

  // A pointer T* converts to U* if T is a class derived from U.
  if (auto target_pointer_type = target_type_inst.TryAs<SemIR::PointerType>()) {
    if (auto src_pointer_type =
            sem_ir.types().TryGetAs<SemIR::PointerType>(value_type_id)) {
      if (auto path = ComputeInheritancePath(context, loc_id,
                                             src_pointer_type->pointee_id,
                                             target_pointer_type->pointee_id);
          path && !path->empty()) {
        return ConvertDerivedPointerToBasePointer(
            context, loc_id, *src_pointer_type, target.type_id, value_id,
            *path);
      }
    }
  }

  if (target.type_id == SemIR::TypeType::SingletonTypeId) {
    // A tuple of types converts to type `type`.
    // TODO: This should apply even for non-literal tuples.
    if (auto tuple_literal = value.TryAs<SemIR::TupleLiteral>()) {
      llvm::SmallVector<SemIR::TypeId> type_ids;
      for (auto tuple_inst_id :
           sem_ir.inst_blocks().Get(tuple_literal->elements_id)) {
        // TODO: This call recurses back into conversion. Switch to an
        // iterative approach.
        type_ids.push_back(
            ExprAsType(context, loc_id, tuple_inst_id, target.diagnose)
                .type_id);
      }
      auto tuple_type_id = GetTupleType(context, type_ids);
      return sem_ir.types().GetInstId(tuple_type_id);
    }

    // `{}` converts to `{} as type`.
    // TODO: This conversion should also be performed for a non-literal value
    // of type `{}`.
    if (auto struct_literal = value.TryAs<SemIR::StructLiteral>();
        struct_literal &&
        struct_literal->elements_id == SemIR::InstBlockId::Empty) {
      value_id = sem_ir.types().GetInstId(value_type_id);
    }

    // Facet type conversions: a value T of facet type F1 can be implicitly
    // converted to facet type F2 if T satisfies the requirements of F2.
    //
    // TODO: Support this conversion in general. For now we only support it in
    // the case where F1 is a facet type and F2 is `type`.
    // TODO: Support converting tuple and struct values to facet types,
    // combining the above conversions and this one in a single conversion.
    if (sem_ir.types().Is<SemIR::FacetType>(value_type_id)) {
      return AddInst<SemIR::FacetAccessType>(
          context, loc_id,
          {.type_id = target.type_id, .facet_value_inst_id = value_id});
    }
  }

  if (sem_ir.types().Is<SemIR::FacetType>(target.type_id) &&
      (sem_ir.types().Is<SemIR::TypeType>(value_type_id) ||
       sem_ir.types().Is<SemIR::FacetType>(value_type_id))) {
    // The value is a type or facet value, so it has a constant value. We get
    // that to unwrap things like NameRef and get to the underlying type or
    // facet value instruction so that we can use `TryGetAs`.
    auto lookup_inst_id = sem_ir.constant_values().GetConstantInstId(value_id);

    if (auto facet_access_type_inst =
            sem_ir.insts().TryGetAs<SemIR::FacetAccessType>(lookup_inst_id)) {
      // Conversion from a `FacetAccessType` to a `FacetValue` of the target
      // `FacetType` if the instruction in the `FacetAccessType` is of a
      // `FacetType` that satisfies the requirements of the target `FacetType`.
      auto facet_value_inst_id = facet_access_type_inst->facet_value_inst_id;

      // If the `FacetType` exactly matches the target `FacetType` then we can
      // shortcut and use that value, and avoid impl lookup.
      if (sem_ir.insts().Get(facet_value_inst_id).type_id() == target.type_id) {
        return facet_value_inst_id;
      }

      auto lookup_result = LookupImplWitness(
          context, loc_id, context.constant_values().Get(facet_value_inst_id),
          context.types().GetConstantId(target.type_id));
      if (lookup_result.has_value()) {
        if (lookup_result.has_error_value()) {
          return SemIR::ErrorInst::SingletonInstId;
        } else {
          return AddInst<SemIR::FacetValue>(
              context, loc_id,
              {.type_id = target.type_id,
               .type_inst_id = lookup_inst_id,
               .witnesses_block_id = lookup_result.inst_block_id()});
        }
      }
    }

    if (sem_ir.types().Is<SemIR::FacetType>(
            sem_ir.insts().Get(lookup_inst_id).type_id())) {
      // Conversion from a facet value to a `FacetValue` of the target
      // `FacetType`. We move up to the higher typish level, and convert the
      // `FacetType` (which is of type TypeType) directly. If the `FacetType`
      // itself implements the target `FacetType` (`impl Iface1 as Iface2`),
      // then we will produce a `FacetValue` of the target `FacetType`.
      lookup_inst_id = context.types().GetInstId(
          sem_ir.insts().Get(lookup_inst_id).type_id());
    }

    if (sem_ir.types().Is<SemIR::TypeType>(
            sem_ir.insts().Get(lookup_inst_id).type_id())) {
      // Conversion from a type value (which has type `type`) to a facet value
      // (which has type `FacetType`), if the type satisfies the requirements of
      // the target `FacetType`, as determined by finding an impl witness. This
      // binds the value to the `FacetType` with a `FacetValue`.
      auto lookup_result = LookupImplWitness(
          context, loc_id, sem_ir.constant_values().Get(lookup_inst_id),
          sem_ir.types().GetConstantId(target.type_id));
      if (lookup_result.has_value()) {
        if (lookup_result.has_error_value()) {
          return SemIR::ErrorInst::SingletonInstId;
        } else {
          return AddInst<SemIR::FacetValue>(
              context, loc_id,
              {.type_id = target.type_id,
               .type_inst_id = lookup_inst_id,
               .witnesses_block_id = lookup_result.inst_block_id()});
        }
      }
    }
  }

  // No builtin conversion applies.
  return value_id;
}

// Given a value expression, form a corresponding initializer that copies from
// that value, if it is possible to do so.
static auto PerformCopy(Context& context, SemIR::InstId expr_id, bool diagnose)
    -> SemIR::InstId {
  auto expr = context.insts().Get(expr_id);
  auto type_id = expr.type_id();
  if (type_id == SemIR::ErrorInst::SingletonTypeId) {
    return SemIR::ErrorInst::SingletonInstId;
  }

  if (InitReprIsCopyOfValueRepr(context.sem_ir(), type_id)) {
    // For simple by-value types, no explicit action is required. Initializing
    // from a value expression is treated as copying the value.
    return expr_id;
  }

  // TODO: We don't yet have rules for whether and when a class type is
  // copyable, or how to perform the copy.
  if (diagnose) {
    CARBON_DIAGNOSTIC(CopyOfUncopyableType, Error,
                      "cannot copy value of type {0}", TypeOfInstId);
    context.emitter().Emit(expr_id, CopyOfUncopyableType, expr_id);
  }
  return SemIR::ErrorInst::SingletonInstId;
}

auto Convert(Context& context, SemIR::LocId loc_id, SemIR::InstId expr_id,
             ConversionTarget target) -> SemIR::InstId {
  auto& sem_ir = context.sem_ir();
  auto orig_expr_id = expr_id;

  // Start by making sure both sides are non-errors. If any part is an error,
  // the result is an error and we shouldn't diagnose.
  if (sem_ir.insts().Get(expr_id).type_id() ==
          SemIR::ErrorInst::SingletonTypeId ||
      target.type_id == SemIR::ErrorInst::SingletonTypeId) {
    return SemIR::ErrorInst::SingletonInstId;
  }

  if (SemIR::GetExprCategory(sem_ir, expr_id) == SemIR::ExprCategory::NotExpr) {
    // TODO: We currently encounter this for use of namespaces and functions.
    // We should provide a better diagnostic for inappropriate use of
    // namespace names, and allow use of functions as values.
    if (target.diagnose) {
      CARBON_DIAGNOSTIC(UseOfNonExprAsValue, Error,
                        "expression cannot be used as a value");
      context.emitter().Emit(expr_id, UseOfNonExprAsValue);
    }
    return SemIR::ErrorInst::SingletonInstId;
  }

  // We can only perform initialization for complete, non-abstract types. Note
  // that `RequireConcreteType` returns true for facet types, since their
  // representation is fixed. This allows us to support using the `Self` of an
  // interface inside its definition.
  if (!RequireConcreteType(
          context, target.type_id, loc_id,
          [&] {
            CARBON_CHECK(!target.is_initializer(),
                         "Initialization of incomplete types is expected to be "
                         "caught elsewhere.");
            if (!target.diagnose) {
              return context.emitter().BuildSuppressed();
            }
            CARBON_DIAGNOSTIC(IncompleteTypeInValueConversion, Error,
                              "forming value of incomplete type {0}",
                              SemIR::TypeId);
            CARBON_DIAGNOSTIC(IncompleteTypeInConversion, Error,
                              "invalid use of incomplete type {0}",
                              SemIR::TypeId);
            return context.emitter().Build(
                loc_id,
                target.kind == ConversionTarget::Value
                    ? IncompleteTypeInValueConversion
                    : IncompleteTypeInConversion,
                target.type_id);
          },
          [&] {
            if (!target.diagnose || !target.is_initializer()) {
              return context.emitter().BuildSuppressed();
            }
            CARBON_DIAGNOSTIC(AbstractTypeInInit, Error,
                              "initialization of abstract type {0}",
                              SemIR::TypeId);
            return context.emitter().Build(loc_id, AbstractTypeInInit,
                                           target.type_id);
          })) {
    return SemIR::ErrorInst::SingletonInstId;
  }

  // Check whether any builtin conversion applies.
  expr_id = PerformBuiltinConversion(context, loc_id, expr_id, target);
  if (expr_id == SemIR::ErrorInst::SingletonInstId) {
    return expr_id;
  }

  // If this is not a builtin conversion, try an `ImplicitAs` conversion.
  if (sem_ir.insts().Get(expr_id).type_id() != target.type_id) {
    SemIR::InstId interface_args[] = {
        context.types().GetInstId(target.type_id)};
    Operator op = {
        .interface_name = target.kind == ConversionTarget::ExplicitAs
                              ? llvm::StringLiteral("As")
                              : llvm::StringLiteral("ImplicitAs"),
        .interface_args_ref = interface_args,
        .op_name = "Convert",
    };
    expr_id = BuildUnaryOperator(context, loc_id, op, expr_id, [&] {
      if (!target.diagnose) {
        return context.emitter().BuildSuppressed();
      }
      // TODO: Should this message change to say "object of type" when
      // converting from a reference expression?
      CARBON_DIAGNOSTIC(ImplicitAsConversionFailure, Error,
                        "cannot implicitly convert value of type {0} to {1}",
                        TypeOfInstId, SemIR::TypeId);
      CARBON_DIAGNOSTIC(ExplicitAsConversionFailure, Error,
                        "cannot convert value of type {0} to {1} with `as`",
                        TypeOfInstId, SemIR::TypeId);
      return context.emitter().Build(loc_id,
                                     target.kind == ConversionTarget::ExplicitAs
                                         ? ExplicitAsConversionFailure
                                         : ImplicitAsConversionFailure,
                                     expr_id, target.type_id);
    });

    // Pull a value directly out of the initializer if possible and wanted.
    if (expr_id != SemIR::ErrorInst::SingletonInstId &&
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
  if (target.kind == ConversionTarget::ExplicitAs) {
    return expr_id;
  }

  // Now perform any necessary value category conversions.
  switch (SemIR::GetExprCategory(sem_ir, expr_id)) {
    case SemIR::ExprCategory::NotExpr:
    case SemIR::ExprCategory::Mixed:
      CARBON_FATAL("Unexpected expression {0} after builtin conversions",
                   sem_ir.insts().Get(expr_id));

    case SemIR::ExprCategory::Error:
      return SemIR::ErrorInst::SingletonInstId;

    case SemIR::ExprCategory::Initializing:
      if (target.is_initializer()) {
        if (orig_expr_id == expr_id) {
          // Don't fill in the return slot if we created the expression through
          // a conversion. In that case, we will have created it with the
          // target already set.
          // TODO: Find a better way to track whether we need to do this.
          MarkInitializerFor(sem_ir, expr_id, target.init_id,
                             *target.init_block);
        }
        break;
      }

      // Commit to using a temporary for this initializing expression.
      // TODO: Don't create a temporary if the initializing representation
      // is already a value representation.
      expr_id = FinalizeTemporary(context, expr_id,
                                  target.kind == ConversionTarget::Discarded);
      // We now have an ephemeral reference.
      [[fallthrough]];

    case SemIR::ExprCategory::DurableRef:
    case SemIR::ExprCategory::EphemeralRef:
      // If a reference expression is an acceptable result, we're done.
      if (target.kind == ConversionTarget::ValueOrRef ||
          target.kind == ConversionTarget::Discarded) {
        break;
      }

      // If we have a reference and don't want one, form a value binding.
      // TODO: Support types with custom value representations.
      expr_id = AddInst<SemIR::BindValue>(
          context, context.insts().GetLocId(expr_id),
          {.type_id = target.type_id, .value_id = expr_id});
      // We now have a value expression.
      [[fallthrough]];

    case SemIR::ExprCategory::Value:
      // When initializing from a value, perform a copy.
      if (target.is_initializer()) {
        expr_id = PerformCopy(context, expr_id, target.diagnose);
      }
      break;
  }

  // Perform a final destination store, if necessary.
  if (target.kind == ConversionTarget::FullInitializer) {
    if (auto init_rep = SemIR::InitRepr::ForType(sem_ir, target.type_id);
        init_rep.kind == SemIR::InitRepr::ByCopy) {
      target.init_block->InsertHere();
      expr_id = AddInst<SemIR::InitializeFrom>(context, loc_id,
                                               {.type_id = target.type_id,
                                                .src_id = expr_id,
                                                .dest_id = target.init_id});
    }
  }

  return expr_id;
}

auto Initialize(Context& context, SemIR::LocId loc_id, SemIR::InstId target_id,
                SemIR::InstId value_id) -> SemIR::InstId {
  PendingBlock target_block(context);
  return Convert(context, loc_id, value_id,
                 {.kind = ConversionTarget::Initializer,
                  .type_id = context.insts().Get(target_id).type_id(),
                  .init_id = target_id,
                  .init_block = &target_block});
}

auto ConvertToValueExpr(Context& context, SemIR::InstId expr_id)
    -> SemIR::InstId {
  return Convert(context, context.insts().GetLocId(expr_id), expr_id,
                 {.kind = ConversionTarget::Value,
                  .type_id = context.insts().Get(expr_id).type_id()});
}

auto ConvertToValueOrRefExpr(Context& context, SemIR::InstId expr_id)
    -> SemIR::InstId {
  return Convert(context, context.insts().GetLocId(expr_id), expr_id,
                 {.kind = ConversionTarget::ValueOrRef,
                  .type_id = context.insts().Get(expr_id).type_id()});
}

auto ConvertToValueOfType(Context& context, SemIR::LocId loc_id,
                          SemIR::InstId expr_id, SemIR::TypeId type_id)
    -> SemIR::InstId {
  return Convert(context, loc_id, expr_id,
                 {.kind = ConversionTarget::Value, .type_id = type_id});
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
      GetSingletonType(context, SemIR::BoolType::SingletonInstId));
}

auto ConvertForExplicitAs(Context& context, Parse::NodeId as_node,
                          SemIR::InstId value_id, SemIR::TypeId type_id)
    -> SemIR::InstId {
  return Convert(context, as_node, value_id,
                 {.kind = ConversionTarget::ExplicitAs, .type_id = type_id});
}

// TODO: Consider moving this to pattern_match.h.
auto ConvertCallArgs(Context& context, SemIR::LocId call_loc_id,
                     SemIR::InstId self_id,
                     llvm::ArrayRef<SemIR::InstId> arg_refs,
                     SemIR::InstId return_slot_arg_id,
                     const SemIR::Function& callee,
                     SemIR::SpecificId callee_specific_id)
    -> SemIR::InstBlockId {
  // The callee reference can be invalidated by conversions, so ensure all reads
  // from it are done before conversion calls.
  auto callee_decl_id = callee.latest_decl_id();
  auto param_patterns =
      context.inst_blocks().GetOrEmpty(callee.param_patterns_id);
  auto return_slot_pattern_id = callee.return_slot_pattern_id;

  // The caller should have ensured this callee has the right arity.
  CARBON_CHECK(arg_refs.size() == param_patterns.size());

  if (callee.self_param_id.has_value() && !self_id.has_value()) {
    CARBON_DIAGNOSTIC(MissingObjectInMethodCall, Error,
                      "missing object argument in method call");
    CARBON_DIAGNOSTIC(InCallToFunction, Note, "calling function declared here");
    context.emitter()
        .Build(call_loc_id, MissingObjectInMethodCall)
        .Note(callee_decl_id, InCallToFunction)
        .Emit();
    self_id = SemIR::ErrorInst::SingletonInstId;
  }

  return CallerPatternMatch(context, callee_specific_id, callee.self_param_id,
                            callee.param_patterns_id, return_slot_pattern_id,
                            self_id, arg_refs, return_slot_arg_id);
}

auto ExprAsType(Context& context, SemIR::LocId loc_id, SemIR::InstId value_id,
                bool diagnose) -> TypeExpr {
  auto type_inst_id = ConvertToValueOfType(context, loc_id, value_id,
                                           SemIR::TypeType::SingletonTypeId);
  if (type_inst_id == SemIR::ErrorInst::SingletonInstId) {
    return {.inst_id = type_inst_id,
            .type_id = SemIR::ErrorInst::SingletonTypeId};
  }

  auto type_const_id = context.constant_values().Get(type_inst_id);
  if (!type_const_id.is_constant()) {
    if (diagnose) {
      CARBON_DIAGNOSTIC(TypeExprEvaluationFailure, Error,
                        "cannot evaluate type expression");
      context.emitter().Emit(loc_id, TypeExprEvaluationFailure);
    }
    return {.inst_id = SemIR::ErrorInst::SingletonInstId,
            .type_id = SemIR::ErrorInst::SingletonTypeId};
  }

  return {.inst_id = type_inst_id,
          .type_id = context.types().GetTypeIdForTypeConstantId(type_const_id)};
}

}  // namespace Carbon::Check

// NOLINTEND(misc-no-recursion)

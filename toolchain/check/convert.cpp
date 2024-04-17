// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/convert.h"

#include <string>
#include <utility>

#include "common/check.h"
#include "llvm/ADT/STLExtras.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/sem_ir/copy_on_write_block.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Given an initializing expression, find its return slot. Returns `Invalid` if
// there is no return slot, because the initialization is not performed in
// place.
static auto FindReturnSlotForInitializer(SemIR::File& sem_ir,
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
        if (!SemIR::GetInitRepr(sem_ir, call.type_id).has_return_slot()) {
          return SemIR::InstId::Invalid;
        }
        if (!call.args_id.is_valid()) {
          // Argument initialization failed, so we have no return slot.
          return SemIR::InstId::Invalid;
        }
        return sem_ir.inst_blocks().Get(call.args_id).back();
      }
      default:
        CARBON_FATAL() << "Initialization from unexpected inst "
                       << init_untyped;
    }
  }
}

// Marks the initializer `init_id` as initializing `target_id`.
static auto MarkInitializerFor(SemIR::File& sem_ir, SemIR::InstId init_id,
                               SemIR::InstId target_id,
                               PendingBlock& target_block) -> void {
  auto return_slot_id = FindReturnSlotForInitializer(sem_ir, init_id);
  if (return_slot_id.is_valid()) {
    // Replace the temporary in the return slot with a reference to our target.
    CARBON_CHECK(sem_ir.insts().Get(return_slot_id).kind() ==
                 SemIR::TemporaryStorage::Kind)
        << "Return slot for initializer does not contain a temporary; "
        << "initialized multiple times? Have "
        << sem_ir.insts().Get(return_slot_id);
    target_block.MergeReplacing(return_slot_id, target_id);
  }
}

// Commits to using a temporary to store the result of the initializing
// expression described by `init_id`, and returns the location of the
// temporary. If `discarded` is `true`, the result is discarded, and no
// temporary will be created if possible; if no temporary is created, the
// return value will be `SemIR::InstId::Invalid`.
static auto FinalizeTemporary(Context& context, SemIR::InstId init_id,
                              bool discarded) -> SemIR::InstId {
  auto& sem_ir = context.sem_ir();
  auto return_slot_id = FindReturnSlotForInitializer(sem_ir, init_id);
  if (return_slot_id.is_valid()) {
    // The return slot should already have a materialized temporary in it.
    CARBON_CHECK(sem_ir.insts().Get(return_slot_id).kind() ==
                 SemIR::TemporaryStorage::Kind)
        << "Return slot for initializer does not contain a temporary; "
        << "initialized multiple times? Have "
        << sem_ir.insts().Get(return_slot_id);
    auto init = sem_ir.insts().Get(init_id);
    return context.AddInst(
        {sem_ir.insts().GetLocId(init_id),
         SemIR::Temporary{init.type_id(), return_slot_id, init_id}});
  }

  if (discarded) {
    // Don't invent a temporary that we're going to discard.
    return SemIR::InstId::Invalid;
  }

  // The initializer has no return slot, but we want to produce a temporary
  // object. Materialize one now.
  // TODO: Consider using an invalid ID to mean that we immediately
  // materialize and initialize a temporary, rather than two separate
  // instructions.
  auto init = sem_ir.insts().Get(init_id);
  auto loc_id = sem_ir.insts().GetLocId(init_id);
  auto temporary_id =
      context.AddInst({loc_id, SemIR::TemporaryStorage{init.type_id()}});
  return context.AddInst(
      {loc_id, SemIR::Temporary{init.type_id(), temporary_id, init_id}});
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

// Creates and adds an instruction to perform element access into an aggregate.
template <typename AccessInstT, typename InstBlockT>
static auto MakeElementAccessInst(Context& context, SemIR::LocId loc_id,
                                  SemIR::InstId aggregate_id,
                                  SemIR::TypeId elem_type_id, InstBlockT& block,
                                  std::size_t i) {
  if constexpr (std::is_same_v<AccessInstT, SemIR::ArrayIndex>) {
    // TODO: Add a new instruction kind for indexing an array at a constant
    // index so that we don't need an integer literal instruction here, and
    // remove this special case.
    auto index_id = block.AddInst(
        {loc_id,
         SemIR::IntLiteral{context.GetBuiltinType(SemIR::BuiltinKind::IntType),
                           context.ints().Add(llvm::APInt(32, i))}});
    return block.AddInst(
        {loc_id, AccessInstT{elem_type_id, aggregate_id, index_id}});
  } else {
    return block.AddInst({loc_id, AccessInstT{elem_type_id, aggregate_id,
                                              SemIR::ElementIndex(i)}});
  }
}

// Converts an element of one aggregate so that it can be used as an element of
// another aggregate.
//
// For the source: `src_id` is the source aggregate, `src_elem_type` is the
// element type, `i` is the index, and `SourceAccessInstT` is the kind of
// instruction used to access the source element.
//
// For the target: `kind` is the kind of conversion or initialization,
// `target_elem_type` is the element type. For initialization, `target_id` is
// the destination, `target_block` is a pending block for target location
// calculations that will be spliced as the return slot of the initializer if
// necessary, `i` is the index, and `TargetAccessInstT` is the kind of
// instruction used to access the destination element.
template <typename SourceAccessInstT, typename TargetAccessInstT>
static auto ConvertAggregateElement(
    Context& context, SemIR::LocId loc_id, SemIR::InstId src_id,
    SemIR::TypeId src_elem_type,
    llvm::ArrayRef<SemIR::InstId> src_literal_elems,
    ConversionTarget::Kind kind, SemIR::InstId target_id,
    SemIR::TypeId target_elem_type, PendingBlock* target_block, std::size_t i) {
  // Compute the location of the source element. This goes into the current code
  // block, not into the target block.
  // TODO: Ideally we would discard this instruction if it's unused.
  auto src_elem_id =
      !src_literal_elems.empty()
          ? src_literal_elems[i]
          : MakeElementAccessInst<SourceAccessInstT>(context, loc_id, src_id,
                                                     src_elem_type, context, i);

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
      context, loc_id, target_id, target_elem_type, *target_block, i);
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
  uint64_t array_bound = sem_ir.GetArrayBoundValue(array_type.bound_id);
  if (tuple_elem_types.size() != array_bound) {
    CARBON_DIAGNOSTIC(
        ArrayInitFromLiteralArgCountMismatch, Error,
        "Cannot initialize array of {0} element(s) from {1} initializer(s).",
        uint64_t, size_t);
    CARBON_DIAGNOSTIC(ArrayInitFromExprArgCountMismatch, Error,
                      "Cannot initialize array of {0} element(s) from tuple "
                      "with {1} element(s).",
                      uint64_t, size_t);
    context.emitter().Emit(value_loc_id,
                           literal_elems.empty()
                               ? ArrayInitFromExprArgCountMismatch
                               : ArrayInitFromLiteralArgCountMismatch,
                           array_bound, tuple_elem_types.size());
    return SemIR::InstId::BuiltinError;
  }

  PendingBlock target_block_storage(context);
  PendingBlock* target_block =
      target.init_block ? target.init_block : &target_block_storage;

  // Arrays are always initialized in-place. Allocate a temporary as the
  // destination for the array initialization if we weren't given one.
  SemIR::InstId return_slot_id = target.init_id;
  if (!target.init_id.is_valid()) {
    return_slot_id = target_block->AddInst(
        {value_loc_id, SemIR::TemporaryStorage{target.type_id}});
  }

  // Initialize each element of the array from the corresponding element of the
  // tuple.
  // TODO: Annotate diagnostics coming from here with the array element index,
  // if initializing from a tuple literal.
  llvm::SmallVector<SemIR::InstId> inits;
  inits.reserve(array_bound + 1);
  for (auto [i, src_type_id] : llvm::enumerate(tuple_elem_types)) {
    // TODO: This call recurses back into conversion. Switch to an iterative
    // approach.
    auto init_id =
        ConvertAggregateElement<SemIR::TupleAccess, SemIR::ArrayIndex>(
            context, value_loc_id, value_id, src_type_id, literal_elems,
            ConversionTarget::FullInitializer, return_slot_id,
            array_type.element_type_id, target_block, i);
    if (init_id == SemIR::InstId::BuiltinError) {
      return SemIR::InstId::BuiltinError;
    }
    inits.push_back(init_id);
  }

  // Flush the temporary here if we didn't insert it earlier, so we can add a
  // reference to the return slot.
  target_block->InsertHere();
  return context.AddInst(
      {value_loc_id,
       SemIR::ArrayInit{target.type_id, sem_ir.inst_blocks().Add(inits),
                        return_slot_id}});
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
  auto literal_elems_id = SemIR::InstBlockId::Invalid;
  if (auto tuple_literal = value.TryAs<SemIR::TupleLiteral>()) {
    literal_elems_id = tuple_literal->elements_id;
    literal_elems = sem_ir.inst_blocks().Get(literal_elems_id);
  } else {
    value_id = MaterializeIfInitializing(context, value_id);
  }

  // Check that the tuples are the same size.
  if (src_elem_types.size() != dest_elem_types.size()) {
    CARBON_DIAGNOSTIC(TupleInitElementCountMismatch, Error,
                      "Cannot initialize tuple of {0} element(s) from tuple "
                      "with {1} element(s).",
                      size_t, size_t);
    context.emitter().Emit(value_loc_id, TupleInitElementCountMismatch,
                           dest_elem_types.size(), src_elem_types.size());
    return SemIR::InstId::BuiltinError;
  }

  // If we're forming an initializer, then we want an initializer for each
  // element. Otherwise, we want a value representation for each element.
  // Perform a final destination store if we're performing an in-place
  // initialization.
  bool is_init = target.is_initializer();
  ConversionTarget::Kind inner_kind =
      !is_init ? ConversionTarget::Value
      : SemIR::GetInitRepr(sem_ir, target.type_id).kind ==
              SemIR::InitRepr::InPlace
          ? ConversionTarget::FullInitializer
          : ConversionTarget::Initializer;

  // Initialize each element of the destination from the corresponding element
  // of the source.
  // TODO: Annotate diagnostics coming from here with the element index.
  auto new_block =
      literal_elems_id.is_valid()
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
            inner_kind, target.init_id, dest_type_id, target.init_block, i);
    if (init_id == SemIR::InstId::BuiltinError) {
      return SemIR::InstId::BuiltinError;
    }
    new_block.Set(i, init_id);
  }

  if (is_init) {
    target.init_block->InsertHere();
    return context.AddInst(
        {value_loc_id,
         SemIR::TupleInit{target.type_id, new_block.id(), target.init_id}});
  } else {
    return context.AddInst(
        {value_loc_id, SemIR::TupleValue{target.type_id, new_block.id()}});
  }
}

// Common implementation for ConvertStructToStruct and ConvertStructToClass.
template <typename TargetAccessInstT>
static auto ConvertStructToStructOrClass(Context& context,
                                         SemIR::StructType src_type,
                                         SemIR::StructType dest_type,
                                         SemIR::InstId value_id,
                                         ConversionTarget target, bool is_class)
    -> SemIR::InstId {
  auto& sem_ir = context.sem_ir();
  auto src_elem_fields = sem_ir.inst_blocks().Get(src_type.fields_id);
  auto dest_elem_fields = sem_ir.inst_blocks().Get(dest_type.fields_id);

  auto value = sem_ir.insts().Get(value_id);
  auto value_loc_id = sem_ir.insts().GetLocId(value_id);

  // If we're initializing from a struct literal, we will use its elements
  // directly. Otherwise, materialize a temporary if needed and index into the
  // result.
  llvm::ArrayRef<SemIR::InstId> literal_elems;
  auto literal_elems_id = SemIR::InstBlockId::Invalid;
  if (auto struct_literal = value.TryAs<SemIR::StructLiteral>()) {
    literal_elems_id = struct_literal->elements_id;
    literal_elems = sem_ir.inst_blocks().Get(literal_elems_id);
  } else {
    value_id = MaterializeIfInitializing(context, value_id);
  }

  // Check that the structs are the same size.
  // TODO: If not, include the name of the first source field that doesn't
  // exist in the destination or vice versa in the diagnostic.
  if (src_elem_fields.size() != dest_elem_fields.size()) {
    CARBON_DIAGNOSTIC(StructInitElementCountMismatch, Error,
                      "Cannot initialize {0} with {1} field(s) from struct "
                      "with {2} field(s).",
                      llvm::StringLiteral, size_t, size_t);
    context.emitter().Emit(
        value_loc_id, StructInitElementCountMismatch,
        is_class ? llvm::StringLiteral("class") : llvm::StringLiteral("struct"),
        dest_elem_fields.size(), src_elem_fields.size());
    return SemIR::InstId::BuiltinError;
  }

  // Prepare to look up fields in the source by index.
  llvm::SmallDenseMap<SemIR::NameId, int32_t> src_field_indexes;
  if (src_type.fields_id != dest_type.fields_id) {
    for (auto [i, field_id] : llvm::enumerate(src_elem_fields)) {
      auto [it, added] = src_field_indexes.insert(
          {context.insts().GetAs<SemIR::StructTypeField>(field_id).name_id, i});
      CARBON_CHECK(added) << "Duplicate field in source structure";
    }
  }

  // If we're forming an initializer, then we want an initializer for each
  // element. Otherwise, we want a value representation for each element.
  // Perform a final destination store if we're performing an in-place
  // initialization.
  bool is_init = target.is_initializer();
  ConversionTarget::Kind inner_kind =
      !is_init ? ConversionTarget::Value
      : SemIR::GetInitRepr(sem_ir, target.type_id).kind ==
              SemIR::InitRepr::InPlace
          ? ConversionTarget::FullInitializer
          : ConversionTarget::Initializer;

  // Initialize each element of the destination from the corresponding element
  // of the source.
  // TODO: Annotate diagnostics coming from here with the element index.
  auto new_block =
      literal_elems_id.is_valid()
          ? SemIR::CopyOnWriteInstBlock(sem_ir, literal_elems_id)
          : SemIR::CopyOnWriteInstBlock(
                sem_ir, SemIR::CopyOnWriteInstBlock::UninitializedBlock{
                            src_elem_fields.size()});
  for (auto [i, dest_field_id] : llvm::enumerate(dest_elem_fields)) {
    auto dest_field =
        sem_ir.insts().GetAs<SemIR::StructTypeField>(dest_field_id);

    // Find the matching source field.
    auto src_field_index = i;
    if (src_type.fields_id != dest_type.fields_id) {
      auto src_field_it = src_field_indexes.find(dest_field.name_id);
      if (src_field_it == src_field_indexes.end()) {
        if (literal_elems_id.is_valid()) {
          CARBON_DIAGNOSTIC(
              StructInitMissingFieldInLiteral, Error,
              "Missing value for field `{0}` in struct initialization.",
              SemIR::NameId);
          context.emitter().Emit(value_loc_id, StructInitMissingFieldInLiteral,
                                 dest_field.name_id);
        } else {
          CARBON_DIAGNOSTIC(StructInitMissingFieldInConversion, Error,
                            "Cannot convert from struct type `{0}` to `{1}`: "
                            "missing field `{2}` in source type.",
                            SemIR::TypeId, SemIR::TypeId, SemIR::NameId);
          context.emitter().Emit(
              value_loc_id, StructInitMissingFieldInConversion, value.type_id(),
              target.type_id, dest_field.name_id);
        }
        return SemIR::InstId::BuiltinError;
      }
      src_field_index = src_field_it->second;
    }
    auto src_field = sem_ir.insts().GetAs<SemIR::StructTypeField>(
        src_elem_fields[src_field_index]);

    // TODO: This call recurses back into conversion. Switch to an iterative
    // approach.
    auto init_id =
        ConvertAggregateElement<SemIR::StructAccess, TargetAccessInstT>(
            context, value_loc_id, value_id, src_field.field_type_id,
            literal_elems, inner_kind, target.init_id, dest_field.field_type_id,
            target.init_block, src_field_index);
    if (init_id == SemIR::InstId::BuiltinError) {
      return SemIR::InstId::BuiltinError;
    }
    new_block.Set(i, init_id);
  }

  if (is_class) {
    target.init_block->InsertHere();
    CARBON_CHECK(is_init)
        << "Converting directly to a class value is not supported";
    return context.AddInst(
        {value_loc_id,
         SemIR::ClassInit{target.type_id, new_block.id(), target.init_id}});
  } else if (is_init) {
    target.init_block->InsertHere();
    return context.AddInst(
        {value_loc_id,
         SemIR::StructInit{target.type_id, new_block.id(), target.init_id}});
  } else {
    return context.AddInst(
        {value_loc_id, SemIR::StructValue{target.type_id, new_block.id()}});
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
      context, src_type, dest_type, value_id, target, /*is_class=*/false);
}

// Performs a conversion from a struct to a class type. This function only
// converts the type, and does not perform a final conversion to the requested
// expression category.
static auto ConvertStructToClass(Context& context, SemIR::StructType src_type,
                                 SemIR::ClassType dest_type,
                                 SemIR::InstId value_id,
                                 ConversionTarget target) -> SemIR::InstId {
  PendingBlock target_block(context);
  auto& class_info = context.classes().Get(dest_type.class_id);
  if (class_info.inheritance_kind == SemIR::Class::Abstract) {
    CARBON_DIAGNOSTIC(ConstructionOfAbstractClass, Error,
                      "Cannot construct instance of abstract class. "
                      "Consider using `partial {0}` instead.",
                      SemIR::TypeId);
    context.emitter().Emit(value_id, ConstructionOfAbstractClass,
                           target.type_id);
    return SemIR::InstId::BuiltinError;
  }
  if (class_info.object_repr_id == SemIR::TypeId::Error) {
    return SemIR::InstId::BuiltinError;
  }
  auto dest_struct_type =
      context.types().GetAs<SemIR::StructType>(class_info.object_repr_id);

  // If we're trying to create a class value, form a temporary for the value to
  // point to.
  bool need_temporary = !target.is_initializer();
  if (need_temporary) {
    target.kind = ConversionTarget::Initializer;
    target.init_block = &target_block;
    target.init_id =
        target_block.AddInst({context.insts().GetLocId(value_id),
                              SemIR::TemporaryStorage{target.type_id}});
  }

  auto result_id = ConvertStructToStructOrClass<SemIR::ClassElementAccess>(
      context, src_type, dest_struct_type, value_id, target, /*is_class=*/true);

  if (need_temporary) {
    target_block.InsertHere();
    result_id = context.AddInst(
        {context.insts().GetLocId(value_id),
         SemIR::Temporary{target.type_id, target.init_id, result_id}});
  }
  return result_id;
}

// An inheritance path is a sequence of `BaseDecl`s in order from derived to
// base.
using InheritancePath = llvm::SmallVector<SemIR::InstId>;

// Computes the inheritance path from class `derived_id` to class `base_id`.
// Returns nullopt if `derived_id` is not a class derived from `base_id`.
static auto ComputeInheritancePath(Context& context, SemIR::TypeId derived_id,
                                   SemIR::TypeId base_id)
    -> std::optional<InheritancePath> {
  // We intend for NRVO to be applied to `result`. All `return` statements in
  // this function should `return result;`.
  std::optional<InheritancePath> result(std::in_place);
  if (!context.TryToCompleteType(derived_id)) {
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
    if (!derived_class.base_id.is_valid()) {
      result = std::nullopt;
      break;
    }
    result->push_back(derived_class.base_id);
    derived_id = context.insts()
                     .GetAs<SemIR::BaseDecl>(derived_class.base_id)
                     .base_type_id;
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
  for (auto base_id : path) {
    auto base_decl = context.insts().GetAs<SemIR::BaseDecl>(base_id);
    value_id = context.AddInst(
        {loc_id, SemIR::ClassElementAccess{base_decl.base_type_id, value_id,
                                           base_decl.index}});
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
  auto ref_id =
      context.AddInst({loc_id, SemIR::Deref{src_ptr_type.pointee_id, ptr_id}});

  // Convert as a reference expression.
  ref_id = ConvertDerivedToBase(context, loc_id, ref_id, path);

  // Take the address.
  return context.AddInst({loc_id, SemIR::AddrOf{dest_ptr_type_id, ref_id}});
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

// Returns the non-adapter type that is compatible with the specified type.
static auto GetCompatibleBaseType(Context& context, SemIR::TypeId type_id)
    -> SemIR::TypeId {
  // If the type is an adapter, its object representation type is its compatible
  // non-adapter type.
  if (auto class_type = context.types().TryGetAs<SemIR::ClassType>(type_id)) {
    auto& class_info = context.classes().Get(class_type->class_id);
    if (class_info.adapt_id.is_valid()) {
      return class_info.object_repr_id;
    }
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
        IsValidExprCategoryForConversionTarget(SemIR::ExprCategory::Value,
                                               target.kind) &&
        SemIR::GetInitRepr(sem_ir, value_type_id).kind ==
            SemIR::InitRepr::ByCopy) {
      auto value_rep = SemIR::GetValueRepr(sem_ir, value_type_id);
      if (value_rep.kind == SemIR::ValueRepr::Copy &&
          value_rep.type_id == value_type_id) {
        // The initializer produces an object representation by copy, and the
        // value representation is a copy of the object representation, so we
        // already have a value of the right form.
        return context.AddInst(
            {loc_id, SemIR::ValueOfInitializer{value_type_id, value_id}});
      }
    }
  }

  // T explicitly converts to U if T is compatible with U.
  if (target.kind == ConversionTarget::Kind::ExplicitAs &&
      target.type_id != value_type_id) {
    auto target_base_id = GetCompatibleBaseType(context, target.type_id);
    auto value_base_id = GetCompatibleBaseType(context, value_type_id);
    if (target_base_id == value_base_id) {
      // For a struct or tuple literal, perform a category conversion if
      // necessary.
      if (SemIR::GetExprCategory(context.sem_ir(), value_id) ==
          SemIR::ExprCategory::Mixed) {
        value_id = PerformBuiltinConversion(
            context, loc_id, value_id,
            ConversionTarget{.kind = ConversionTarget::Value,
                             .type_id = value_type_id});
      }
      return context.AddInst(
          {loc_id, SemIR::AsCompatible{target.type_id, value_id}});
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

  // A tuple (T1, T2, ..., Tn) converts to [T; n] if each Ti converts to T.
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
               .adapt_id.is_valid()) {
        return ConvertStructToClass(context, *src_struct_type,
                                    *target_class_type, value_id, target);
      }
    }

    // An expression of type T converts to U if T is a class derived from U.
    if (auto path =
            ComputeInheritancePath(context, value_type_id, target.type_id);
        path && !path->empty()) {
      return ConvertDerivedToBase(context, loc_id, value_id, *path);
    }
  }

  // A pointer T* converts to U* if T is a class derived from U.
  if (auto target_pointer_type = target_type_inst.TryAs<SemIR::PointerType>()) {
    if (auto src_pointer_type =
            sem_ir.types().TryGetAs<SemIR::PointerType>(value_type_id)) {
      if (auto path =
              ComputeInheritancePath(context, src_pointer_type->pointee_id,
                                     target_pointer_type->pointee_id);
          path && !path->empty()) {
        return ConvertDerivedPointerToBasePointer(
            context, loc_id, *src_pointer_type, target.type_id, value_id,
            *path);
      }
    }
  }

  if (target.type_id == SemIR::TypeId::TypeType) {
    // A tuple of types converts to type `type`.
    // TODO: This should apply even for non-literal tuples.
    if (auto tuple_literal = value.TryAs<SemIR::TupleLiteral>()) {
      llvm::SmallVector<SemIR::TypeId> type_ids;
      for (auto tuple_inst_id :
           sem_ir.inst_blocks().Get(tuple_literal->elements_id)) {
        // TODO: This call recurses back into conversion. Switch to an
        // iterative approach.
        type_ids.push_back(ExprAsType(context, loc_id, tuple_inst_id));
      }
      auto tuple_type_id = context.GetTupleType(type_ids);
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
    // the case where F1 is an interface type and F2 is `type`.
    // TODO: Support converting tuple and struct values to facet types,
    // combining the above conversions and this one in a single conversion.
    if (sem_ir.types().Is<SemIR::InterfaceType>(value_type_id)) {
      return context.AddInst(
          {loc_id, SemIR::FacetTypeAccess{target.type_id, value_id}});
    }
  }

  // No builtin conversion applies.
  return value_id;
}

// Given a value expression, form a corresponding initializer that copies from
// that value, if it is possible to do so.
static auto PerformCopy(Context& context, SemIR::InstId expr_id)
    -> SemIR::InstId {
  auto expr = context.insts().Get(expr_id);
  auto type_id = expr.type_id();
  if (type_id == SemIR::TypeId::Error) {
    return SemIR::InstId::BuiltinError;
  }

  // TODO: Directly track on the value representation whether it's a copy of
  // the object representation.
  auto value_rep = SemIR::GetValueRepr(context.sem_ir(), type_id);
  if (value_rep.kind == SemIR::ValueRepr::Copy &&
      value_rep.aggregate_kind == SemIR::ValueRepr::NotAggregate &&
      value_rep.type_id == type_id) {
    // For by-value scalar types, no explicit action is required. Initializing
    // from a value expression is treated as copying the value.
    return expr_id;
  }

  // TODO: We don't yet have rules for whether and when a class type is
  // copyable, or how to perform the copy.
  CARBON_DIAGNOSTIC(CopyOfUncopyableType, Error,
                    "Cannot copy value of type `{0}`.", SemIR::TypeId);
  context.emitter().Emit(expr_id, CopyOfUncopyableType, type_id);
  return SemIR::InstId::BuiltinError;
}

auto Convert(Context& context, SemIR::LocId loc_id, SemIR::InstId expr_id,
             ConversionTarget target) -> SemIR::InstId {
  auto& sem_ir = context.sem_ir();
  auto orig_expr_id = expr_id;

  // Start by making sure both sides are valid. If any part is invalid, the
  // result is invalid and we shouldn't error.
  if (sem_ir.insts().Get(expr_id).type_id() == SemIR::TypeId::Error ||
      target.type_id == SemIR::TypeId::Error) {
    return SemIR::InstId::BuiltinError;
  }

  if (SemIR::GetExprCategory(sem_ir, expr_id) == SemIR::ExprCategory::NotExpr) {
    // TODO: We currently encounter this for use of namespaces and functions.
    // We should provide a better diagnostic for inappropriate use of
    // namespace names, and allow use of functions as values.
    CARBON_DIAGNOSTIC(UseOfNonExprAsValue, Error,
                      "Expression cannot be used as a value.");
    context.emitter().Emit(expr_id, UseOfNonExprAsValue);
    return SemIR::InstId::BuiltinError;
  }

  // We can only perform initialization for complete types.
  if (!context.TryToCompleteType(target.type_id, [&] {
        CARBON_DIAGNOSTIC(IncompleteTypeInInit, Error,
                          "Initialization of incomplete type `{0}`.",
                          SemIR::TypeId);
        CARBON_DIAGNOSTIC(IncompleteTypeInValueConversion, Error,
                          "Forming value of incomplete type `{0}`.",
                          SemIR::TypeId);
        CARBON_DIAGNOSTIC(IncompleteTypeInConversion, Error,
                          "Invalid use of incomplete type `{0}`.",
                          SemIR::TypeId);
        return context.emitter().Build(loc_id,
                                       target.is_initializer()
                                           ? IncompleteTypeInInit
                                       : target.kind == ConversionTarget::Value
                                           ? IncompleteTypeInValueConversion
                                           : IncompleteTypeInConversion,
                                       target.type_id);
      })) {
    return SemIR::InstId::BuiltinError;
  }

  // Check whether any builtin conversion applies.
  expr_id = PerformBuiltinConversion(context, loc_id, expr_id, target);
  if (expr_id == SemIR::InstId::BuiltinError) {
    return expr_id;
  }

  // If the types don't match at this point, we can't perform the conversion.
  // TODO: Look for an `ImplicitAs` impl, or an `As` impl in the case where
  // `target.kind == ConversionTarget::ExplicitAs`.
  SemIR::Inst expr = sem_ir.insts().Get(expr_id);
  if (expr.type_id() != target.type_id) {
    CARBON_DIAGNOSTIC(ImplicitAsConversionFailure, Error,
                      "Cannot implicitly convert from `{0}` to `{1}`.",
                      SemIR::TypeId, SemIR::TypeId);
    CARBON_DIAGNOSTIC(ExplicitAsConversionFailure, Error,
                      "Cannot convert from `{0}` to `{1}` with `as`.",
                      SemIR::TypeId, SemIR::TypeId);
    context.emitter()
        .Build(loc_id,
               target.kind == ConversionTarget::ExplicitAs
                   ? ExplicitAsConversionFailure
                   : ImplicitAsConversionFailure,
               expr.type_id(), target.type_id)
        .Emit();
    return SemIR::InstId::BuiltinError;
  }

  // Track that we performed a type conversion, if we did so.
  if (orig_expr_id != expr_id) {
    expr_id = context.AddInst(
        {loc_id, SemIR::Converted{target.type_id, orig_expr_id, expr_id}});
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
      CARBON_FATAL() << "Unexpected expression " << expr
                     << " after builtin conversions";

    case SemIR::ExprCategory::Error:
      return SemIR::InstId::BuiltinError;

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
      expr_id = context.AddInst({context.insts().GetLocId(expr_id),
                                 SemIR::BindValue{expr.type_id(), expr_id}});
      // We now have a value expression.
      [[fallthrough]];

    case SemIR::ExprCategory::Value:
      // When initializing from a value, perform a copy.
      if (target.is_initializer()) {
        expr_id = PerformCopy(context, expr_id);
      }
      break;
  }

  // Perform a final destination store, if necessary.
  if (target.kind == ConversionTarget::FullInitializer) {
    if (auto init_rep = SemIR::GetInitRepr(sem_ir, target.type_id);
        init_rep.kind == SemIR::InitRepr::ByCopy) {
      target.init_block->InsertHere();
      expr_id = context.AddInst(
          {loc_id,
           SemIR::InitializeFrom{target.type_id, expr_id, target.init_id}});
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

auto ConvertToBoolValue(Context& context, SemIR::LocId loc_id,
                        SemIR::InstId value_id) -> SemIR::InstId {
  return ConvertToValueOfType(
      context, loc_id, value_id,
      context.GetBuiltinType(SemIR::BuiltinKind::BoolType));
}

auto ConvertForExplicitAs(Context& context, Parse::NodeId as_node,
                          SemIR::InstId value_id, SemIR::TypeId type_id)
    -> SemIR::InstId {
  return Convert(context, as_node, value_id,
                 {.kind = ConversionTarget::ExplicitAs, .type_id = type_id});
}

CARBON_DIAGNOSTIC(InCallToFunction, Note, "Calling function declared here.");

// Convert the object argument in a method call to match the `self` parameter.
static auto ConvertSelf(Context& context, SemIR::LocId call_loc_id,
                        SemIR::InstId callee_id,
                        std::optional<SemIR::AddrPattern> addr_pattern,
                        SemIR::InstId self_param_id, SemIR::Param self_param,
                        SemIR::InstId self_id) -> SemIR::InstId {
  if (!self_id.is_valid()) {
    CARBON_DIAGNOSTIC(MissingObjectInMethodCall, Error,
                      "Missing object argument in method call.");
    context.emitter()
        .Build(call_loc_id, MissingObjectInMethodCall)
        .Note(callee_id, InCallToFunction)
        .Emit();
    return SemIR::InstId::BuiltinError;
  }

  DiagnosticAnnotationScope annotate_diagnostics(
      &context.emitter(), [&](auto& builder) {
        CARBON_DIAGNOSTIC(
            InCallToFunctionSelf, Note,
            "Initializing `{0}` parameter of method declared here.",
            llvm::StringLiteral);
        builder.Note(self_param_id, InCallToFunctionSelf,
                     addr_pattern ? llvm::StringLiteral("addr self")
                                  : llvm::StringLiteral("self"));
      });

  // For `addr self`, take the address of the object argument.
  auto self_or_addr_id = self_id;
  if (addr_pattern) {
    self_or_addr_id = ConvertToValueOrRefExpr(context, self_or_addr_id);
    auto self = context.insts().Get(self_or_addr_id);
    switch (SemIR::GetExprCategory(context.sem_ir(), self_id)) {
      case SemIR::ExprCategory::Error:
      case SemIR::ExprCategory::DurableRef:
      case SemIR::ExprCategory::EphemeralRef:
        break;
      default:
        CARBON_DIAGNOSTIC(AddrSelfIsNonRef, Error,
                          "`addr self` method cannot be invoked on a value.");
        context.emitter().Emit(TokenOnly(call_loc_id), AddrSelfIsNonRef);
        return SemIR::InstId::BuiltinError;
    }
    auto loc_id = context.insts().GetLocId(self_or_addr_id);
    self_or_addr_id = context.AddInst(
        {loc_id, SemIR::AddrOf{context.GetPointerType(self.type_id()),
                               self_or_addr_id}});
  }

  return ConvertToValueOfType(context, call_loc_id, self_or_addr_id,
                              self_param.type_id);
}

auto ConvertCallArgs(Context& context, SemIR::LocId call_loc_id,
                     SemIR::InstId self_id,
                     llvm::ArrayRef<SemIR::InstId> arg_refs,
                     SemIR::InstId return_storage_id, SemIR::InstId callee_id,
                     SemIR::InstBlockId implicit_param_refs_id,
                     SemIR::InstBlockId param_refs_id) -> SemIR::InstBlockId {
  auto implicit_param_refs = context.inst_blocks().Get(implicit_param_refs_id);
  auto param_refs = context.inst_blocks().Get(param_refs_id);

  // If sizes mismatch, fail early.
  if (arg_refs.size() != param_refs.size()) {
    CARBON_DIAGNOSTIC(CallArgCountMismatch, Error,
                      "{0} argument(s) passed to function expecting "
                      "{1} argument(s).",
                      int, int);
    context.emitter()
        .Build(call_loc_id, CallArgCountMismatch, arg_refs.size(),
               param_refs.size())
        .Note(callee_id, InCallToFunction)
        .Emit();
    return SemIR::InstBlockId::Invalid;
  }

  // Start building a block to hold the converted arguments.
  llvm::SmallVector<SemIR::InstId> args;
  args.reserve(implicit_param_refs.size() + param_refs.size() +
               return_storage_id.is_valid());

  // Check implicit parameters.
  for (auto implicit_param_id : implicit_param_refs) {
    auto addr_pattern =
        context.insts().TryGetAs<SemIR::AddrPattern>(implicit_param_id);
    auto [param_id, param] = SemIR::Function::GetParamFromParamRefId(
        context.sem_ir(), implicit_param_id);
    if (param.name_id == SemIR::NameId::SelfValue) {
      auto converted_self_id =
          ConvertSelf(context, call_loc_id, callee_id, addr_pattern, param_id,
                      param, self_id);
      if (converted_self_id == SemIR::InstId::BuiltinError) {
        return SemIR::InstBlockId::Invalid;
      }
      args.push_back(converted_self_id);
    } else {
      // TODO: Form argument values for implicit parameters.
      context.TODO(call_loc_id, "Call with implicit parameters");
      return SemIR::InstBlockId::Invalid;
    }
  }

  int diag_param_index;
  DiagnosticAnnotationScope annotate_diagnostics(
      &context.emitter(), [&](auto& builder) {
        CARBON_DIAGNOSTIC(
            InCallToFunctionParam, Note,
            "Initializing parameter {0} of function declared here.", int);
        builder.Note(callee_id, InCallToFunctionParam, diag_param_index + 1);
      });

  // Check type conversions per-element.
  for (auto [i, arg_id, param_id] : llvm::enumerate(arg_refs, param_refs)) {
    diag_param_index = i;

    auto param_type_id = context.insts().Get(param_id).type_id();
    // TODO: Convert to the proper expression category. For now, we assume
    // parameters are all `let` bindings.
    auto converted_arg_id =
        ConvertToValueOfType(context, call_loc_id, arg_id, param_type_id);
    if (converted_arg_id == SemIR::InstId::BuiltinError) {
      return SemIR::InstBlockId::Invalid;
    }

    args.push_back(converted_arg_id);
  }

  // Track the return storage, if present.
  if (return_storage_id.is_valid()) {
    args.push_back(return_storage_id);
  }

  return context.inst_blocks().Add(args);
}

auto ExprAsType(Context& context, SemIR::LocId loc_id, SemIR::InstId value_id)
    -> SemIR::TypeId {
  auto type_inst_id =
      ConvertToValueOfType(context, loc_id, value_id, SemIR::TypeId::TypeType);
  if (type_inst_id == SemIR::InstId::BuiltinError) {
    return SemIR::TypeId::Error;
  }

  auto type_const_id = context.constant_values().Get(type_inst_id);
  if (!type_const_id.is_constant()) {
    CARBON_DIAGNOSTIC(TypeExprEvaluationFailure, Error,
                      "Cannot evaluate type expression.");
    context.emitter().Emit(loc_id, TypeExprEvaluationFailure);
    return SemIR::TypeId::Error;
  }

  return context.GetTypeIdForTypeConstant(type_const_id);
}

}  // namespace Carbon::Check

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/convert.h"

#include <string>
#include <utility>

#include "common/check.h"
#include "llvm/ADT/STLExtras.h"
#include "toolchain/check/context.h"
#include "toolchain/diagnostics/diagnostic_kind.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/inst_kind.h"

namespace Carbon::Check {

// Given an initializing expression, find its return slot. Returns `Invalid` if
// there is no return slot, because the initialization is not performed in
// place.
static auto FindReturnSlotForInitializer(SemIR::File& sem_ir,
                                         SemIR::InstId init_id)
    -> SemIR::InstId {
  SemIR::Node init = sem_ir.insts().Get(init_id);
  switch (init.kind()) {
    default:
      CARBON_FATAL() << "Initialization from unexpected node " << init;

    case SemIR::StructInit::Kind:
    case SemIR::TupleInit::Kind:
      // TODO: Track a return slot for these initializers.
      CARBON_FATAL() << init
                     << " should be created with its return slot already "
                        "filled in properly";

    case SemIR::InitializeFrom::Kind: {
      return init.As<SemIR::InitializeFrom>().dest_id;
    }

    case SemIR::Call::Kind: {
      auto call = init.As<SemIR::Call>();
      if (!SemIR::GetInitializingRepresentation(sem_ir, call.type_id)
               .has_return_slot()) {
        return SemIR::InstId::Invalid;
      }
      return sem_ir.inst_blocks().Get(call.args_id).back();
    }

    case SemIR::ArrayInit::Kind: {
      return sem_ir.inst_blocks()
          .Get(init.As<SemIR::ArrayInit>().inits_and_return_slot_id)
          .back();
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
    return context.AddNode(SemIR::Temporary{init.parse_node(), init.type_id(),
                                            return_slot_id, init_id});
  }

  if (discarded) {
    // Don't invent a temporary that we're going to discard.
    return SemIR::InstId::Invalid;
  }

  // The initializer has no return slot, but we want to produce a temporary
  // object. Materialize one now.
  // TODO: Consider using an invalid ID to mean that we immediately
  // materialize and initialize a temporary, rather than two separate
  // nodes.
  auto init = sem_ir.insts().Get(init_id);
  auto temporary_id = context.AddNode(
      SemIR::TemporaryStorage{init.parse_node(), init.type_id()});
  return context.AddNode(SemIR::Temporary{init.parse_node(), init.type_id(),
                                          temporary_id, init_id});
}

// Materialize a temporary to hold the result of the given expression if it is
// an initializing expression.
static auto MaterializeIfInitializing(Context& context, SemIR::InstId expr_id)
    -> SemIR::InstId {
  if (GetExpressionCategory(context.sem_ir(), expr_id) ==
      SemIR::ExpressionCategory::Initializing) {
    return FinalizeTemporary(context, expr_id, /*discarded=*/false);
  }
  return expr_id;
}

// Creates and adds a node to perform element access into an aggregate.
template <typename AccessNodeT, typename InstBlockT>
static auto MakeElemAccessNode(Context& context, Parse::Node parse_node,
                               SemIR::InstId aggregate_id,
                               SemIR::TypeId elem_type_id, InstBlockT& block,
                               std::size_t i) {
  if constexpr (std::is_same_v<AccessNodeT, SemIR::ArrayIndex>) {
    // TODO: Add a new node kind for indexing an array at a constant index
    // so that we don't need an integer literal node here, and remove this
    // special case.
    auto index_id = block.AddNode(SemIR::IntegerLiteral{
        parse_node, context.GetBuiltinType(SemIR::BuiltinKind::IntegerType),
        context.sem_ir().integers().Add(llvm::APInt(32, i))});
    return block.AddNode(
        AccessNodeT{parse_node, elem_type_id, aggregate_id, index_id});
  } else {
    return block.AddNode(AccessNodeT{parse_node, elem_type_id, aggregate_id,
                                     SemIR::MemberIndex(i)});
  }
}

// Converts an element of one aggregate so that it can be used as an element of
// another aggregate.
//
// For the source: `src_id` is the source aggregate, `src_elem_type` is the
// element type, `i` is the index, and `SourceAccessNodeT` is the kind of node
// used to access the source element.
//
// For the target: `kind` is the kind of conversion or initialization,
// `target_elem_type` is the element type. For initialization, `target_id` is
// the destination, `target_block` is a pending block for target location
// calculations that will be spliced as the return slot of the initializer if
// necessary, `i` is the index, and `TargetAccessNodeT` is the kind of node
// used to access the destination element.
template <typename SourceAccessNodeT, typename TargetAccessNodeT>
static auto ConvertAggregateElement(
    Context& context, Parse::Node parse_node, SemIR::InstId src_id,
    SemIR::TypeId src_elem_type,
    llvm::ArrayRef<SemIR::InstId> src_literal_elems,
    ConversionTarget::Kind kind, SemIR::InstId target_id,
    SemIR::TypeId target_elem_type, PendingBlock* target_block, std::size_t i) {
  // Compute the location of the source element. This goes into the current code
  // block, not into the target block.
  // TODO: Ideally we would discard this node if it's unused.
  auto src_elem_id =
      !src_literal_elems.empty()
          ? src_literal_elems[i]
          : MakeElemAccessNode<SourceAccessNodeT>(context, parse_node, src_id,
                                                  src_elem_type, context, i);

  // If we're performing a conversion rather than an initialization, we won't
  // have or need a target.
  ConversionTarget target = {.kind = kind, .type_id = target_elem_type};
  if (!target.is_initializer()) {
    return Convert(context, parse_node, src_elem_id, target);
  }

  // Compute the location of the target element and initialize it.
  PendingBlock::DiscardUnusedNodesScope scope(target_block);
  target.init_block = target_block;
  target.init_id = MakeElemAccessNode<TargetAccessNodeT>(
      context, parse_node, target_id, target_elem_type, *target_block, i);
  return Convert(context, parse_node, src_elem_id, target);
}

namespace {
// A handle to a new block that may be modified, with copy-on-write semantics.
//
// The constructor is given the ID of an existing block that provides the
// initial contents of the new block. The new block is lazily allocated; if no
// modifications have been made, the `id()` function will return the original
// block ID.
//
// This is intended to avoid an unnecessary block allocation in the case where
// the new block ends up being exactly the same as the original block.
class CopyOnWriteBlock {
 public:
  // Constructs the block. If `source_id` is valid, it is used as the initial
  // value of the block. Otherwise, uninitialized storage for `size` elements
  // is allocated.
  CopyOnWriteBlock(SemIR::File& file, SemIR::InstBlockId source_id, size_t size)
      : file_(file), source_id_(source_id) {
    if (!source_id_.is_valid()) {
      id_ = file_.inst_blocks().AddUninitialized(size);
    }
  }

  auto id() -> SemIR::InstBlockId const { return id_; }

  auto Set(int i, SemIR::InstId value) -> void {
    if (source_id_.is_valid() && file_.inst_blocks().Get(id_)[i] == value) {
      return;
    }
    if (id_ == source_id_) {
      id_ = file_.inst_blocks().Add(file_.inst_blocks().Get(source_id_));
    }
    file_.inst_blocks().Get(id_)[i] = value;
  }

 private:
  SemIR::File& file_;
  SemIR::InstBlockId source_id_;
  SemIR::InstBlockId id_ = source_id_;
};
}  // namespace

// Performs a conversion from a tuple to an array type. Does not perform a
// final conversion to the requested expression category.
static auto ConvertTupleToArray(Context& context, SemIR::TupleType tuple_type,
                                SemIR::ArrayType array_type,
                                SemIR::InstId value_id, ConversionTarget target)
    -> SemIR::InstId {
  auto& sem_ir = context.sem_ir();
  auto tuple_elem_types = sem_ir.type_blocks().Get(tuple_type.elements_id);

  auto value = sem_ir.insts().Get(value_id);

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
    CARBON_DIAGNOSTIC(ArrayInitFromExpressionArgCountMismatch, Error,
                      "Cannot initialize array of {0} element(s) from tuple "
                      "with {1} element(s).",
                      uint64_t, size_t);
    context.emitter().Emit(value.parse_node(),
                           literal_elems.empty()
                               ? ArrayInitFromExpressionArgCountMismatch
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
    return_slot_id = target_block->AddNode(
        SemIR::TemporaryStorage{value.parse_node(), target.type_id});
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
            context, value.parse_node(), value_id, src_type_id, literal_elems,
            ConversionTarget::FullInitializer, return_slot_id,
            array_type.element_type_id, target_block, i);
    if (init_id == SemIR::InstId::BuiltinError) {
      return SemIR::InstId::BuiltinError;
    }
    inits.push_back(init_id);
  }

  // The last element of the refs block contains the return slot for the array
  // initialization. Flush the temporary here if we didn't insert it earlier.
  target_block->InsertHere();
  inits.push_back(return_slot_id);

  return context.AddNode(SemIR::ArrayInit{value.parse_node(), target.type_id,
                                          value_id,
                                          sem_ir.inst_blocks().Add(inits)});
}

// Performs a conversion from a tuple to a tuple type. Does not perform a
// final conversion to the requested expression category.
static auto ConvertTupleToTuple(Context& context, SemIR::TupleType src_type,
                                SemIR::TupleType dest_type,
                                SemIR::InstId value_id, ConversionTarget target)
    -> SemIR::InstId {
  auto& sem_ir = context.sem_ir();
  auto src_elem_types = sem_ir.type_blocks().Get(src_type.elements_id);
  auto dest_elem_types = sem_ir.type_blocks().Get(dest_type.elements_id);

  auto value = sem_ir.insts().Get(value_id);

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
    context.emitter().Emit(value.parse_node(), TupleInitElementCountMismatch,
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
      : SemIR::GetInitializingRepresentation(sem_ir, target.type_id).kind ==
              SemIR::InitializingRepresentation::InPlace
          ? ConversionTarget::FullInitializer
          : ConversionTarget::Initializer;

  // Initialize each element of the destination from the corresponding element
  // of the source.
  // TODO: Annotate diagnostics coming from here with the element index.
  CopyOnWriteBlock new_block(sem_ir, literal_elems_id, src_elem_types.size());
  for (auto [i, src_type_id, dest_type_id] :
       llvm::enumerate(src_elem_types, dest_elem_types)) {
    // TODO: This call recurses back into conversion. Switch to an iterative
    // approach.
    auto init_id =
        ConvertAggregateElement<SemIR::TupleAccess, SemIR::TupleAccess>(
            context, value.parse_node(), value_id, src_type_id, literal_elems,
            inner_kind, target.init_id, dest_type_id, target.init_block, i);
    if (init_id == SemIR::InstId::BuiltinError) {
      return SemIR::InstId::BuiltinError;
    }
    new_block.Set(i, init_id);
  }

  return is_init ? context.AddNode(SemIR::TupleInit{value.parse_node(),
                                                    target.type_id, value_id,
                                                    new_block.id()})
                 : context.AddNode(SemIR::TupleValue{value.parse_node(),
                                                     target.type_id, value_id,
                                                     new_block.id()});
}

// Performs a conversion from a struct to a struct type. Does not perform a
// final conversion to the requested expression category.
static auto ConvertStructToStruct(Context& context, SemIR::StructType src_type,
                                  SemIR::StructType dest_type,
                                  SemIR::InstId value_id,
                                  ConversionTarget target) -> SemIR::InstId {
  auto& sem_ir = context.sem_ir();
  auto src_elem_fields = sem_ir.inst_blocks().Get(src_type.fields_id);
  auto dest_elem_fields = sem_ir.inst_blocks().Get(dest_type.fields_id);

  auto value = sem_ir.insts().Get(value_id);

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
                      "Cannot initialize struct of {0} element(s) from struct "
                      "with {1} element(s).",
                      size_t, size_t);
    context.emitter().Emit(value.parse_node(), StructInitElementCountMismatch,
                           dest_elem_fields.size(), src_elem_fields.size());
    return SemIR::InstId::BuiltinError;
  }

  // Prepare to look up fields in the source by index.
  llvm::SmallDenseMap<StringId, int32_t> src_field_indexes;
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
      : SemIR::GetInitializingRepresentation(sem_ir, target.type_id).kind ==
              SemIR::InitializingRepresentation::InPlace
          ? ConversionTarget::FullInitializer
          : ConversionTarget::Initializer;

  // Initialize each element of the destination from the corresponding element
  // of the source.
  // TODO: Annotate diagnostics coming from here with the element index.
  CopyOnWriteBlock new_block(sem_ir, literal_elems_id, src_elem_fields.size());
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
              llvm::StringRef);
          context.emitter().Emit(value.parse_node(),
                                 StructInitMissingFieldInLiteral,
                                 sem_ir.strings().Get(dest_field.name_id));
        } else {
          CARBON_DIAGNOSTIC(StructInitMissingFieldInConversion, Error,
                            "Cannot convert from struct type `{0}` to `{1}`: "
                            "missing field `{2}` in source type.",
                            std::string, std::string, llvm::StringRef);
          context.emitter().Emit(value.parse_node(),
                                 StructInitMissingFieldInConversion,
                                 sem_ir.StringifyType(value.type_id()),
                                 sem_ir.StringifyType(target.type_id),
                                 sem_ir.strings().Get(dest_field.name_id));
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
        ConvertAggregateElement<SemIR::StructAccess, SemIR::StructAccess>(
            context, value.parse_node(), value_id, src_field.field_type_id,
            literal_elems, inner_kind, target.init_id, dest_field.field_type_id,
            target.init_block, src_field_index);
    if (init_id == SemIR::InstId::BuiltinError) {
      return SemIR::InstId::BuiltinError;
    }
    new_block.Set(i, init_id);
  }

  return is_init ? context.AddNode(SemIR::StructInit{value.parse_node(),
                                                     target.type_id, value_id,
                                                     new_block.id()})
                 : context.AddNode(SemIR::StructValue{value.parse_node(),
                                                      target.type_id, value_id,
                                                      new_block.id()});
}

// Returns whether `category` is a valid expression category to produce as a
// result of a conversion with kind `target_kind`, or at most needs a temporary
// to be materialized.
static bool IsValidExpressionCategoryForConversionTarget(
    SemIR::ExpressionCategory category, ConversionTarget::Kind target_kind) {
  switch (target_kind) {
    case ConversionTarget::Value:
      return category == SemIR::ExpressionCategory::Value;
    case ConversionTarget::ValueOrReference:
    case ConversionTarget::Discarded:
      return category == SemIR::ExpressionCategory::Value ||
             category == SemIR::ExpressionCategory::DurableReference ||
             category == SemIR::ExpressionCategory::EphemeralReference ||
             category == SemIR::ExpressionCategory::Initializing;
    case ConversionTarget::ExplicitAs:
      return true;
    case ConversionTarget::Initializer:
    case ConversionTarget::FullInitializer:
      return category == SemIR::ExpressionCategory::Initializing;
  }
}

static auto PerformBuiltinConversion(Context& context, Parse::Node parse_node,
                                     SemIR::InstId value_id,
                                     ConversionTarget target) -> SemIR::InstId {
  auto& sem_ir = context.sem_ir();
  auto value = sem_ir.insts().Get(value_id);
  auto value_type_id = value.type_id();
  auto target_type_node =
      sem_ir.insts().Get(sem_ir.GetTypeAllowBuiltinTypes(target.type_id));

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
    auto value_cat = SemIR::GetExpressionCategory(sem_ir, value_id);
    if (IsValidExpressionCategoryForConversionTarget(value_cat, target.kind)) {
      return value_id;
    }

    // If the source is an initializing expression, we may be able to pull a
    // value right out of it.
    if (value_cat == SemIR::ExpressionCategory::Initializing &&
        IsValidExpressionCategoryForConversionTarget(
            SemIR::ExpressionCategory::Value, target.kind) &&
        SemIR::GetInitializingRepresentation(sem_ir, value_type_id).kind ==
            SemIR::InitializingRepresentation::ByCopy) {
      auto value_rep = SemIR::GetValueRepresentation(sem_ir, value_type_id);
      if (value_rep.kind == SemIR::ValueRepresentation::Copy &&
          value_rep.type_id == value_type_id) {
        // The initializer produces an object representation by copy, and the
        // value representation is a copy of the object representation, so we
        // already have a value of the right form.
        return context.AddNode(
            SemIR::ValueOfInitializer{parse_node, value_type_id, value_id});
      }
    }
  }

  // A tuple (T1, T2, ..., Tn) converts to (U1, U2, ..., Un) if each Ti
  // converts to Ui.
  if (auto target_tuple_type = target_type_node.TryAs<SemIR::TupleType>()) {
    auto value_type_node =
        sem_ir.insts().Get(sem_ir.GetTypeAllowBuiltinTypes(value_type_id));
    if (auto src_tuple_type = value_type_node.TryAs<SemIR::TupleType>()) {
      return ConvertTupleToTuple(context, *src_tuple_type, *target_tuple_type,
                                 value_id, target);
    }
  }

  // A struct {.f_1: T_1, .f_2: T_2, ..., .f_n: T_n} converts to
  // {.f_p(1): U_p(1), .f_p(2): U_p(2), ..., .f_p(n): U_p(n)} if
  // (p(1), ..., p(n)) is a permutation of (1, ..., n) and each Ti converts
  // to Ui.
  if (auto target_struct_type = target_type_node.TryAs<SemIR::StructType>()) {
    auto value_type_node =
        sem_ir.insts().Get(sem_ir.GetTypeAllowBuiltinTypes(value_type_id));
    if (auto src_struct_type = value_type_node.TryAs<SemIR::StructType>()) {
      return ConvertStructToStruct(context, *src_struct_type,
                                   *target_struct_type, value_id, target);
    }
  }

  // A tuple (T1, T2, ..., Tn) converts to [T; n] if each Ti converts to T.
  if (auto target_array_type = target_type_node.TryAs<SemIR::ArrayType>()) {
    auto value_type_node =
        sem_ir.insts().Get(sem_ir.GetTypeAllowBuiltinTypes(value_type_id));
    if (auto src_tuple_type = value_type_node.TryAs<SemIR::TupleType>()) {
      return ConvertTupleToArray(context, *src_tuple_type, *target_array_type,
                                 value_id, target);
    }
  }

  if (target.type_id == SemIR::TypeId::TypeType) {
    // A tuple of types converts to type `type`.
    // TODO: This should apply even for non-literal tuples.
    if (auto tuple_literal = value.TryAs<SemIR::TupleLiteral>()) {
      llvm::SmallVector<SemIR::TypeId> type_ids;
      for (SemIR::InstId tuple_inst_id :
           sem_ir.inst_blocks().Get(tuple_literal->elements_id)) {
        // TODO: This call recurses back into conversion. Switch to an
        // iterative approach.
        type_ids.push_back(
            ExpressionAsType(context, parse_node, tuple_inst_id));
      }
      auto tuple_type_id =
          context.CanonicalizeTupleType(parse_node, std::move(type_ids));
      return sem_ir.GetTypeAllowBuiltinTypes(tuple_type_id);
    }

    // `{}` converts to `{} as type`.
    // TODO: This conversion should also be performed for a non-literal value
    // of type `{}`.
    if (auto struct_literal = value.TryAs<SemIR::StructLiteral>();
        struct_literal &&
        struct_literal->elements_id == SemIR::InstBlockId::Empty) {
      value_id = sem_ir.GetTypeAllowBuiltinTypes(value_type_id);
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
  auto value_rep = SemIR::GetValueRepresentation(context.sem_ir(), type_id);
  if (value_rep.kind == SemIR::ValueRepresentation::Copy &&
      value_rep.aggregate_kind == SemIR::ValueRepresentation::NotAggregate &&
      value_rep.type_id == type_id) {
    // For by-value scalar types, no explicit action is required. Initializing
    // from a value expression is treated as copying the value.
    return expr_id;
  }

  // TODO: We don't yet have rules for whether and when a class type is
  // copyable, or how to perform the copy.
  CARBON_DIAGNOSTIC(CopyOfUncopyableType, Error,
                    "Cannot copy value of type `{0}`.", std::string);
  context.emitter().Emit(expr.parse_node(), CopyOfUncopyableType,
                         context.sem_ir().StringifyType(type_id));
  return SemIR::InstId::BuiltinError;
}

auto Convert(Context& context, Parse::Node parse_node, SemIR::InstId expr_id,
             ConversionTarget target) -> SemIR::InstId {
  auto& sem_ir = context.sem_ir();
  auto orig_expr_id = expr_id;

  // Start by making sure both sides are valid. If any part is invalid, the
  // result is invalid and we shouldn't error.
  if (sem_ir.insts().Get(expr_id).type_id() == SemIR::TypeId::Error ||
      target.type_id == SemIR::TypeId::Error) {
    return SemIR::InstId::BuiltinError;
  }

  if (SemIR::GetExpressionCategory(sem_ir, expr_id) ==
      SemIR::ExpressionCategory::NotExpression) {
    // TODO: We currently encounter this for use of namespaces and functions.
    // We should provide a better diagnostic for inappropriate use of
    // namespace names, and allow use of functions as values.
    CARBON_DIAGNOSTIC(UseOfNonExpressionAsValue, Error,
                      "Expression cannot be used as a value.");
    context.emitter().Emit(sem_ir.insts().Get(expr_id).parse_node(),
                           UseOfNonExpressionAsValue);
    return SemIR::InstId::BuiltinError;
  }

  // We can only perform initialization for complete types.
  if (!context.TryToCompleteType(target.type_id, [&] {
        CARBON_DIAGNOSTIC(IncompleteTypeInInitialization, Error,
                          "Initialization of incomplete type `{0}`.",
                          std::string);
        CARBON_DIAGNOSTIC(IncompleteTypeInValueConversion, Error,
                          "Forming value of incomplete type `{0}`.",
                          std::string);
        CARBON_DIAGNOSTIC(IncompleteTypeInConversion, Error,
                          "Invalid use of incomplete type `{0}`.", std::string);
        return context.emitter().Build(
            parse_node,
            target.is_initializer() ? IncompleteTypeInInitialization
            : target.kind == ConversionTarget::Value
                ? IncompleteTypeInValueConversion
                : IncompleteTypeInConversion,
            context.sem_ir().StringifyType(target.type_id, true));
      })) {
    return SemIR::InstId::BuiltinError;
  }

  // Check whether any builtin conversion applies.
  expr_id = PerformBuiltinConversion(context, parse_node, expr_id, target);
  if (expr_id == SemIR::InstId::BuiltinError) {
    return expr_id;
  }

  // If the types don't match at this point, we can't perform the conversion.
  // TODO: Look for an `ImplicitAs` impl, or an `As` impl in the case where
  // `target.kind == ConversionTarget::ExplicitAs`.
  SemIR::Node expr = sem_ir.insts().Get(expr_id);
  if (expr.type_id() != target.type_id) {
    CARBON_DIAGNOSTIC(ImplicitAsConversionFailure, Error,
                      "Cannot implicitly convert from `{0}` to `{1}`.",
                      std::string, std::string);
    CARBON_DIAGNOSTIC(ExplicitAsConversionFailure, Error,
                      "Cannot convert from `{0}` to `{1}` with `as`.",
                      std::string, std::string);
    context.emitter()
        .Build(parse_node,
               target.kind == ConversionTarget::ExplicitAs
                   ? ExplicitAsConversionFailure
                   : ImplicitAsConversionFailure,
               sem_ir.StringifyType(expr.type_id()),
               sem_ir.StringifyType(target.type_id))
        .Emit();
    return SemIR::InstId::BuiltinError;
  }

  // For `as`, don't perform any value category conversions. In particular, an
  // identity conversion shouldn't change the expression category.
  if (target.kind == ConversionTarget::ExplicitAs) {
    return expr_id;
  }

  // Now perform any necessary value category conversions.
  switch (SemIR::GetExpressionCategory(sem_ir, expr_id)) {
    case SemIR::ExpressionCategory::NotExpression:
    case SemIR::ExpressionCategory::Mixed:
      CARBON_FATAL() << "Unexpected expression " << expr
                     << " after builtin conversions";

    case SemIR::ExpressionCategory::Error:
      return SemIR::InstId::BuiltinError;

    case SemIR::ExpressionCategory::Initializing:
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

    case SemIR::ExpressionCategory::DurableReference:
    case SemIR::ExpressionCategory::EphemeralReference:
      // If a reference expression is an acceptable result, we're done.
      if (target.kind == ConversionTarget::ValueOrReference ||
          target.kind == ConversionTarget::Discarded) {
        break;
      }

      // If we have a reference and don't want one, form a value binding.
      // TODO: Support types with custom value representations.
      expr_id = context.AddNode(
          SemIR::BindValue{expr.parse_node(), expr.type_id(), expr_id});
      // We now have a value expression.
      [[fallthrough]];

    case SemIR::ExpressionCategory::Value:
      // When initializing from a value, perform a copy.
      if (target.is_initializer()) {
        expr_id = PerformCopy(context, expr_id);
      }
      break;
  }

  // Perform a final destination store, if necessary.
  if (target.kind == ConversionTarget::FullInitializer) {
    if (auto init_rep =
            SemIR::GetInitializingRepresentation(sem_ir, target.type_id);
        init_rep.kind == SemIR::InitializingRepresentation::ByCopy) {
      target.init_block->InsertHere();
      expr_id = context.AddNode(SemIR::InitializeFrom{
          parse_node, target.type_id, expr_id, target.init_id});
    }
  }

  return expr_id;
}

auto Initialize(Context& context, Parse::Node parse_node,
                SemIR::InstId target_id, SemIR::InstId value_id)
    -> SemIR::InstId {
  PendingBlock target_block(context);
  return Convert(context, parse_node, value_id,
                 {.kind = ConversionTarget::Initializer,
                  .type_id = context.sem_ir().insts().Get(target_id).type_id(),
                  .init_id = target_id,
                  .init_block = &target_block});
}

auto ConvertToValueExpression(Context& context, SemIR::InstId expr_id)
    -> SemIR::InstId {
  auto expr = context.sem_ir().insts().Get(expr_id);
  return Convert(context, expr.parse_node(), expr_id,
                 {.kind = ConversionTarget::Value, .type_id = expr.type_id()});
}

auto ConvertToValueOrReferenceExpression(Context& context,
                                         SemIR::InstId expr_id)
    -> SemIR::InstId {
  auto expr = context.sem_ir().insts().Get(expr_id);
  return Convert(
      context, expr.parse_node(), expr_id,
      {.kind = ConversionTarget::ValueOrReference, .type_id = expr.type_id()});
}

auto ConvertToValueOfType(Context& context, Parse::Node parse_node,
                          SemIR::InstId value_id, SemIR::TypeId type_id)
    -> SemIR::InstId {
  return Convert(context, parse_node, value_id,
                 {.kind = ConversionTarget::Value, .type_id = type_id});
}

auto ConvertToBoolValue(Context& context, Parse::Node parse_node,
                        SemIR::InstId value_id) -> SemIR::InstId {
  return ConvertToValueOfType(
      context, parse_node, value_id,
      context.GetBuiltinType(SemIR::BuiltinKind::BoolType));
}

auto ConvertForExplicitAs(Context& context, Parse::Node as_node,
                          SemIR::InstId value_id, SemIR::TypeId type_id)
    -> SemIR::InstId {
  return Convert(context, as_node, value_id,
                 {.kind = ConversionTarget::ExplicitAs, .type_id = type_id});
}

CARBON_DIAGNOSTIC(InCallToFunction, Note, "Calling function declared here.");

// Convert the object argument in a method call to match the `self` parameter.
static auto ConvertSelf(Context& context, Parse::Node call_parse_node,
                        Parse::Node callee_parse_node,
                        SemIR::SelfParameter self_param, SemIR::InstId self_id)
    -> SemIR::InstId {
  if (!self_id.is_valid()) {
    CARBON_DIAGNOSTIC(MissingObjectInMethodCall, Error,
                      "Missing object argument in method call.");
    context.emitter()
        .Build(call_parse_node, MissingObjectInMethodCall)
        .Note(callee_parse_node, InCallToFunction)
        .Emit();
    return SemIR::InstId::BuiltinError;
  }

  DiagnosticAnnotationScope annotate_diagnostics(
      &context.emitter(), [&](auto& builder) {
        CARBON_DIAGNOSTIC(
            InCallToFunctionSelf, Note,
            "Initializing `{0}` parameter of method declared here.",
            llvm::StringLiteral);
        builder.Note(self_param.parse_node, InCallToFunctionSelf,
                     self_param.is_addr_self.index
                         ? llvm::StringLiteral("addr self")
                         : llvm::StringLiteral("self"));
      });

  // For `addr self`, take the address of the object argument.
  auto self_or_addr_id = self_id;
  if (self_param.is_addr_self.index) {
    self_or_addr_id =
        ConvertToValueOrReferenceExpression(context, self_or_addr_id);
    auto self = context.insts().Get(self_or_addr_id);
    switch (SemIR::GetExpressionCategory(context.sem_ir(), self_id)) {
      case SemIR::ExpressionCategory::Error:
      case SemIR::ExpressionCategory::DurableReference:
      case SemIR::ExpressionCategory::EphemeralReference:
        break;
      default:
        CARBON_DIAGNOSTIC(AddrSelfIsNonReference, Error,
                          "`addr self` method cannot be invoked on a value.");
        context.emitter().Emit(call_parse_node, AddrSelfIsNonReference);
        return SemIR::InstId::BuiltinError;
    }
    self_or_addr_id = context.AddNode(SemIR::AddressOf{
        self.parse_node(),
        context.GetPointerType(self.parse_node(), self.type_id()),
        self_or_addr_id});
  }

  return ConvertToValueOfType(context, call_parse_node, self_or_addr_id,
                              self_param.type_id);
}

auto ConvertCallArgs(Context& context, Parse::Node call_parse_node,
                     SemIR::InstId self_id,
                     llvm::ArrayRef<SemIR::InstId> arg_refs,
                     SemIR::InstId return_storage_id,
                     Parse::Node callee_parse_node,
                     SemIR::InstBlockId implicit_param_refs_id,
                     SemIR::InstBlockId param_refs_id) -> SemIR::InstBlockId {
  auto implicit_param_refs =
      context.sem_ir().inst_blocks().Get(implicit_param_refs_id);
  auto param_refs = context.sem_ir().inst_blocks().Get(param_refs_id);

  // If sizes mismatch, fail early.
  if (arg_refs.size() != param_refs.size()) {
    CARBON_DIAGNOSTIC(CallArgCountMismatch, Error,
                      "{0} argument(s) passed to function expecting "
                      "{1} argument(s).",
                      int, int);
    context.emitter()
        .Build(call_parse_node, CallArgCountMismatch, arg_refs.size(),
               param_refs.size())
        .Note(callee_parse_node, InCallToFunction)
        .Emit();
    return SemIR::InstBlockId::Invalid;
  }

  // Start building a block to hold the converted arguments.
  llvm::SmallVector<SemIR::InstId> args;
  args.reserve(implicit_param_refs.size() + param_refs.size() +
               return_storage_id.is_valid());

  // Check implicit parameters.
  for (auto implicit_param_id : implicit_param_refs) {
    auto param = context.insts().Get(implicit_param_id);
    if (auto self_param = param.TryAs<SemIR::SelfParameter>()) {
      auto converted_self_id = ConvertSelf(
          context, call_parse_node, callee_parse_node, *self_param, self_id);
      if (converted_self_id == SemIR::InstId::BuiltinError) {
        return SemIR::InstBlockId::Invalid;
      }
      args.push_back(converted_self_id);
    } else {
      // TODO: Form argument values for implicit parameters.
      context.TODO(call_parse_node, "Call with implicit parameters");
      return SemIR::InstBlockId::Invalid;
    }
  }

  int diag_param_index;
  DiagnosticAnnotationScope annotate_diagnostics(
      &context.emitter(), [&](auto& builder) {
        CARBON_DIAGNOSTIC(
            InCallToFunctionParam, Note,
            "Initializing parameter {0} of function declared here.", int);
        builder.Note(callee_parse_node, InCallToFunctionParam,
                     diag_param_index + 1);
      });

  // Check type conversions per-element.
  for (auto [i, arg_id, param_id] : llvm::enumerate(arg_refs, param_refs)) {
    diag_param_index = i;

    auto param_type_id = context.sem_ir().insts().Get(param_id).type_id();
    // TODO: Convert to the proper expression category. For now, we assume
    // parameters are all `let` bindings.
    auto converted_arg_id =
        ConvertToValueOfType(context, call_parse_node, arg_id, param_type_id);
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

auto ExpressionAsType(Context& context, Parse::Node parse_node,
                      SemIR::InstId value_id) -> SemIR::TypeId {
  return context.CanonicalizeType(ConvertToValueOfType(
      context, parse_node, value_id, SemIR::TypeId::TypeType));
}

}  // namespace Carbon::Check

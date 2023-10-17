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
#include "toolchain/sem_ir/node.h"
#include "toolchain/sem_ir/node_kind.h"

namespace Carbon::Check {

// Given an initializing expression, find its return slot. Returns `Invalid` if
// there is no return slot, because the initialization is not performed in
// place.
static auto FindReturnSlotForInitializer(SemIR::File& semantics_ir,
                                         SemIR::NodeId init_id)
    -> SemIR::NodeId {
  SemIR::Node init = semantics_ir.GetNode(init_id);
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
      if (!SemIR::GetInitializingRepresentation(semantics_ir, call.type_id)
               .has_return_slot()) {
        return SemIR::NodeId::Invalid;
      }
      return semantics_ir.GetNodeBlock(call.args_id).back();
    }

    case SemIR::ArrayInit::Kind: {
      return semantics_ir
          .GetNodeBlock(init.As<SemIR::ArrayInit>().inits_and_return_slot_id)
          .back();
    }
  }
}

// Marks the initializer `init_id` as initializing `target_id`.
static auto MarkInitializerFor(SemIR::File& semantics_ir, SemIR::NodeId init_id,
                               SemIR::NodeId target_id,
                               PendingBlock& target_block) -> void {
  auto return_slot_id = FindReturnSlotForInitializer(semantics_ir, init_id);
  if (return_slot_id.is_valid()) {
    // Replace the temporary in the return slot with a reference to our target.
    CARBON_CHECK(semantics_ir.GetNode(return_slot_id).kind() ==
                 SemIR::TemporaryStorage::Kind)
        << "Return slot for initializer does not contain a temporary; "
        << "initialized multiple times? Have "
        << semantics_ir.GetNode(return_slot_id);
    target_block.MergeReplacing(return_slot_id, target_id);
  }
}

// Commits to using a temporary to store the result of the initializing
// expression described by `init_id`, and returns the location of the
// temporary. If `discarded` is `true`, the result is discarded, and no
// temporary will be created if possible; if no temporary is created, the
// return value will be `SemIR::NodeId::Invalid`.
static auto FinalizeTemporary(Context& context, SemIR::NodeId init_id,
                              bool discarded) -> SemIR::NodeId {
  auto& semantics_ir = context.semantics_ir();
  auto return_slot_id = FindReturnSlotForInitializer(semantics_ir, init_id);
  if (return_slot_id.is_valid()) {
    // The return slot should already have a materialized temporary in it.
    CARBON_CHECK(semantics_ir.GetNode(return_slot_id).kind() ==
                 SemIR::TemporaryStorage::Kind)
        << "Return slot for initializer does not contain a temporary; "
        << "initialized multiple times? Have "
        << semantics_ir.GetNode(return_slot_id);
    auto init = semantics_ir.GetNode(init_id);
    return context.AddNode(SemIR::Temporary(init.parse_node(), init.type_id(),
                                            return_slot_id, init_id));
  }

  if (discarded) {
    // Don't invent a temporary that we're going to discard.
    return SemIR::NodeId::Invalid;
  }

  // The initializer has no return slot, but we want to produce a temporary
  // object. Materialize one now.
  // TODO: Consider using an invalid ID to mean that we immediately
  // materialize and initialize a temporary, rather than two separate
  // nodes.
  auto init = semantics_ir.GetNode(init_id);
  auto temporary_id = context.AddNode(
      SemIR::TemporaryStorage(init.parse_node(), init.type_id()));
  return context.AddNode(SemIR::Temporary(init.parse_node(), init.type_id(),
                                          temporary_id, init_id));
}

// Materialize a temporary to hold the result of the given expression if it is
// an initializing expression.
static auto MaterializeIfInitializing(Context& context, SemIR::NodeId expr_id)
    -> SemIR::NodeId {
  if (GetExpressionCategory(context.semantics_ir(), expr_id) ==
      SemIR::ExpressionCategory::Initializing) {
    return FinalizeTemporary(context, expr_id, /*discarded=*/false);
  }
  return expr_id;
}

// Creates and adds a node to perform element access into an aggregate.
template <typename AccessNodeT, typename NodeBlockT>
static auto MakeElemAccessNode(Context& context, Parse::Node parse_node,
                               SemIR::NodeId aggregate_id,
                               SemIR::TypeId elem_type_id, NodeBlockT& block,
                               std::size_t i) {
  if constexpr (std::is_same_v<AccessNodeT, SemIR::ArrayIndex>) {
    // TODO: Add a new node kind for indexing an array at a constant index
    // so that we don't need an integer literal node here, and remove this
    // special case.
    auto index_id = block.AddNode(SemIR::IntegerLiteral(
        parse_node, context.GetBuiltinType(SemIR::BuiltinKind::IntegerType),
        context.semantics_ir().AddInteger(llvm::APInt(32, i))));
    return block.AddNode(
        AccessNodeT(parse_node, elem_type_id, aggregate_id, index_id));
  } else {
    return block.AddNode(AccessNodeT(parse_node, elem_type_id, aggregate_id,
                                     SemIR::MemberIndex(i)));
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
    Context& context, Parse::Node parse_node, SemIR::NodeId src_id,
    SemIR::TypeId src_elem_type,
    llvm::ArrayRef<SemIR::NodeId> src_literal_elems,
    ConversionTarget::Kind kind, SemIR::NodeId target_id,
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
  CopyOnWriteBlock(SemIR::File& file, SemIR::NodeBlockId source_id, size_t size)
      : file_(file), source_id_(source_id) {
    if (!source_id_.is_valid()) {
      id_ = file_.AddUninitializedNodeBlock(size);
    }
  }

  auto id() -> SemIR::NodeBlockId const { return id_; }

  auto Set(int i, SemIR::NodeId value) -> void {
    if (source_id_.is_valid() && file_.GetNodeBlock(id_)[i] == value) {
      return;
    }
    if (id_ == source_id_) {
      id_ = file_.AddNodeBlock(file_.GetNodeBlock(source_id_));
    }
    file_.GetNodeBlock(id_)[i] = value;
  }

 private:
  SemIR::File& file_;
  SemIR::NodeBlockId source_id_;
  SemIR::NodeBlockId id_ = source_id_;
};
}  // namespace

// Performs a conversion from a tuple to an array type. Does not perform a
// final conversion to the requested expression category.
static auto ConvertTupleToArray(Context& context,
                                SemIR::TupleType::Data tuple_type,
                                SemIR::ArrayType::Data array_type,
                                SemIR::NodeId value_id, ConversionTarget target)
    -> SemIR::NodeId {
  auto& semantics_ir = context.semantics_ir();
  auto tuple_elem_types = semantics_ir.GetTypeBlock(tuple_type.elements_id);

  auto value = semantics_ir.GetNode(value_id);

  // If we're initializing from a tuple literal, we will use its elements
  // directly. Otherwise, materialize a temporary if needed and index into the
  // result.
  llvm::ArrayRef<SemIR::NodeId> literal_elems;
  if (auto tuple_literal = value.TryAs<SemIR::TupleLiteral>()) {
    literal_elems = semantics_ir.GetNodeBlock(tuple_literal->elements_id);
  } else {
    value_id = MaterializeIfInitializing(context, value_id);
  }

  // Check that the tuple is the right size.
  uint64_t array_bound = semantics_ir.GetArrayBoundValue(array_type.bound_id);
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
    return SemIR::NodeId::BuiltinError;
  }

  PendingBlock target_block_storage(context);
  PendingBlock* target_block =
      target.init_block ? target.init_block : &target_block_storage;

  // Arrays are always initialized in-place. Allocate a temporary as the
  // destination for the array initialization if we weren't given one.
  SemIR::NodeId return_slot_id = target.init_id;
  if (!target.init_id.is_valid()) {
    return_slot_id = target_block->AddNode(
        SemIR::TemporaryStorage(value.parse_node(), target.type_id));
  }

  // Initialize each element of the array from the corresponding element of the
  // tuple.
  // TODO: Annotate diagnostics coming from here with the array element index,
  // if initializing from a tuple literal.
  llvm::SmallVector<SemIR::NodeId> inits;
  inits.reserve(array_bound + 1);
  for (auto [i, src_type_id] : llvm::enumerate(tuple_elem_types)) {
    // TODO: This call recurses back into conversion. Switch to an iterative
    // approach.
    auto init_id =
        ConvertAggregateElement<SemIR::TupleAccess, SemIR::ArrayIndex>(
            context, value.parse_node(), value_id, src_type_id, literal_elems,
            ConversionTarget::FullInitializer, return_slot_id,
            array_type.element_type_id, target_block, i);
    if (init_id == SemIR::NodeId::BuiltinError) {
      return SemIR::NodeId::BuiltinError;
    }
    inits.push_back(init_id);
  }

  // The last element of the refs block contains the return slot for the array
  // initialization. Flush the temporary here if we didn't insert it earlier.
  target_block->InsertHere();
  inits.push_back(return_slot_id);

  return context.AddNode(SemIR::ArrayInit(value.parse_node(), target.type_id,
                                          value_id,
                                          semantics_ir.AddNodeBlock(inits)));
}

// Performs a conversion from a tuple to a tuple type. Does not perform a
// final conversion to the requested expression category.
static auto ConvertTupleToTuple(Context& context,
                                SemIR::TupleType::Data src_type,
                                SemIR::TupleType::Data dest_type,
                                SemIR::NodeId value_id, ConversionTarget target)
    -> SemIR::NodeId {
  auto& semantics_ir = context.semantics_ir();
  auto src_elem_types = semantics_ir.GetTypeBlock(src_type.elements_id);
  auto dest_elem_types = semantics_ir.GetTypeBlock(dest_type.elements_id);

  auto value = semantics_ir.GetNode(value_id);

  // If we're initializing from a tuple literal, we will use its elements
  // directly. Otherwise, materialize a temporary if needed and index into the
  // result.
  llvm::ArrayRef<SemIR::NodeId> literal_elems;
  auto literal_elems_id = SemIR::NodeBlockId::Invalid;
  if (auto tuple_literal = value.TryAs<SemIR::TupleLiteral>()) {
    literal_elems_id = tuple_literal->elements_id;
    literal_elems = semantics_ir.GetNodeBlock(literal_elems_id);
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
    return SemIR::NodeId::BuiltinError;
  }

  // If we're forming an initializer, then we want an initializer for each
  // element. Otherwise, we want a value representation for each element.
  // Perform a final destination store if we're performing an in-place
  // initialization.
  bool is_init = target.is_initializer();
  ConversionTarget::Kind inner_kind =
      !is_init ? ConversionTarget::Value
      : SemIR::GetInitializingRepresentation(semantics_ir, target.type_id)
                  .kind == SemIR::InitializingRepresentation::InPlace
          ? ConversionTarget::FullInitializer
          : ConversionTarget::Initializer;

  // Initialize each element of the destination from the corresponding element
  // of the source.
  // TODO: Annotate diagnostics coming from here with the element index.
  CopyOnWriteBlock new_block(semantics_ir, literal_elems_id,
                             src_elem_types.size());
  for (auto [i, src_type_id, dest_type_id] :
       llvm::enumerate(src_elem_types, dest_elem_types)) {
    // TODO: This call recurses back into conversion. Switch to an iterative
    // approach.
    auto init_id =
        ConvertAggregateElement<SemIR::TupleAccess, SemIR::TupleAccess>(
            context, value.parse_node(), value_id, src_type_id, literal_elems,
            inner_kind, target.init_id, dest_type_id, target.init_block, i);
    if (init_id == SemIR::NodeId::BuiltinError) {
      return SemIR::NodeId::BuiltinError;
    }
    new_block.Set(i, init_id);
  }

  return is_init ? context.AddNode(SemIR::TupleInit(value.parse_node(),
                                                    target.type_id, value_id,
                                                    new_block.id()))
                 : context.AddNode(SemIR::TupleValue(value.parse_node(),
                                                     target.type_id, value_id,
                                                     new_block.id()));
}

// Performs a conversion from a struct to a struct type. Does not perform a
// final conversion to the requested expression category.
static auto ConvertStructToStruct(Context& context,
                                  SemIR::StructType::Data src_type,
                                  SemIR::StructType::Data dest_type,
                                  SemIR::NodeId value_id,
                                  ConversionTarget target) -> SemIR::NodeId {
  auto& semantics_ir = context.semantics_ir();
  auto src_elem_fields = semantics_ir.GetNodeBlock(src_type.fields_id);
  auto dest_elem_fields = semantics_ir.GetNodeBlock(dest_type.fields_id);

  auto value = semantics_ir.GetNode(value_id);

  // If we're initializing from a struct literal, we will use its elements
  // directly. Otherwise, materialize a temporary if needed and index into the
  // result.
  llvm::ArrayRef<SemIR::NodeId> literal_elems;
  auto literal_elems_id = SemIR::NodeBlockId::Invalid;
  if (auto struct_literal = value.TryAs<SemIR::StructLiteral>()) {
    literal_elems_id = struct_literal->elements_id;
    literal_elems = semantics_ir.GetNodeBlock(literal_elems_id);
  } else {
    value_id = MaterializeIfInitializing(context, value_id);
  }

  // Check that the structs are the same size.
  // TODO: Check the field names are the same up to permutation, compute the
  // permutation, and use it below.
  if (src_elem_fields.size() != dest_elem_fields.size()) {
    CARBON_DIAGNOSTIC(StructInitElementCountMismatch, Error,
                      "Cannot initialize struct of {0} element(s) from struct "
                      "with {1} element(s).",
                      size_t, size_t);
    context.emitter().Emit(value.parse_node(), StructInitElementCountMismatch,
                           dest_elem_fields.size(), src_elem_fields.size());
    return SemIR::NodeId::BuiltinError;
  }

  // If we're forming an initializer, then we want an initializer for each
  // element. Otherwise, we want a value representation for each element.
  // Perform a final destination store if we're performing an in-place
  // initialization.
  bool is_init = target.is_initializer();
  ConversionTarget::Kind inner_kind =
      !is_init ? ConversionTarget::Value
      : SemIR::GetInitializingRepresentation(semantics_ir, target.type_id)
                  .kind == SemIR::InitializingRepresentation::InPlace
          ? ConversionTarget::FullInitializer
          : ConversionTarget::Initializer;

  // Initialize each element of the destination from the corresponding element
  // of the source.
  // TODO: Annotate diagnostics coming from here with the element index.
  CopyOnWriteBlock new_block(semantics_ir, literal_elems_id,
                             src_elem_fields.size());
  for (auto [i, src_field_id, dest_field_id] :
       llvm::enumerate(src_elem_fields, dest_elem_fields)) {
    auto src_field =
        semantics_ir.GetNodeAs<SemIR::StructTypeField>(src_field_id);
    auto dest_field =
        semantics_ir.GetNodeAs<SemIR::StructTypeField>(dest_field_id);
    if (src_field.name_id != dest_field.name_id) {
      CARBON_DIAGNOSTIC(
          StructInitFieldNameMismatch, Error,
          "Mismatched names for field {0} in struct initialization: "
          "source has field name `{1}`, destination has field name `{2}`.",
          size_t, llvm::StringRef, llvm::StringRef);
      context.emitter().Emit(value.parse_node(), StructInitFieldNameMismatch,
                             i + 1, semantics_ir.GetString(src_field.name_id),
                             semantics_ir.GetString(dest_field.name_id));
      return SemIR::NodeId::BuiltinError;
    }

    // TODO: This call recurses back into conversion. Switch to an iterative
    // approach.
    auto init_id =
        ConvertAggregateElement<SemIR::StructAccess, SemIR::StructAccess>(
            context, value.parse_node(), value_id, src_field.type_id,
            literal_elems, inner_kind, target.init_id, dest_field.type_id,
            target.init_block, i);
    if (init_id == SemIR::NodeId::BuiltinError) {
      return SemIR::NodeId::BuiltinError;
    }
    new_block.Set(i, init_id);
  }

  return is_init ? context.AddNode(SemIR::StructInit(value.parse_node(),
                                                     target.type_id, value_id,
                                                     new_block.id()))
                 : context.AddNode(SemIR::StructValue(value.parse_node(),
                                                      target.type_id, value_id,
                                                      new_block.id()));
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
    case ConversionTarget::Initializer:
    case ConversionTarget::FullInitializer:
      return category == SemIR::ExpressionCategory::Initializing;
  }
}

static auto PerformBuiltinConversion(Context& context, Parse::Node parse_node,
                                     SemIR::NodeId value_id,
                                     ConversionTarget target) -> SemIR::NodeId {
  auto& semantics_ir = context.semantics_ir();
  auto value = semantics_ir.GetNode(value_id);
  auto value_type_id = value.type_id();
  auto target_type_node = semantics_ir.GetNode(
      semantics_ir.GetTypeAllowBuiltinTypes(target.type_id));

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
  if (value_type_id == target.type_id &&
      IsValidExpressionCategoryForConversionTarget(
          SemIR::GetExpressionCategory(semantics_ir, value_id), target.kind)) {
    return value_id;
  }

  // A tuple (T1, T2, ..., Tn) converts to (U1, U2, ..., Un) if each Ti
  // converts to Ui.
  if (auto target_tuple_type = target_type_node.TryAs<SemIR::TupleType>()) {
    auto value_type_node = semantics_ir.GetNode(
        semantics_ir.GetTypeAllowBuiltinTypes(value_type_id));
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
    auto value_type_node = semantics_ir.GetNode(
        semantics_ir.GetTypeAllowBuiltinTypes(value_type_id));
    if (auto src_struct_type = value_type_node.TryAs<SemIR::StructType>()) {
      return ConvertStructToStruct(context, *src_struct_type,
                                   *target_struct_type, value_id, target);
    }
  }

  // A tuple (T1, T2, ..., Tn) converts to [T; n] if each Ti converts to T.
  if (auto target_array_type = target_type_node.TryAs<SemIR::ArrayType>()) {
    auto value_type_node = semantics_ir.GetNode(
        semantics_ir.GetTypeAllowBuiltinTypes(value_type_id));
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
      for (auto tuple_node_id :
           semantics_ir.GetNodeBlock(tuple_literal->elements_id)) {
        // TODO: This call recurses back into conversion. Switch to an
        // iterative approach.
        type_ids.push_back(
            ExpressionAsType(context, parse_node, tuple_node_id));
      }
      auto tuple_type_id =
          context.CanonicalizeTupleType(parse_node, std::move(type_ids));
      return semantics_ir.GetTypeAllowBuiltinTypes(tuple_type_id);
    }

    // `{}` converts to `{} as type`.
    // TODO: This conversion should also be performed for a non-literal value
    // of type `{}`.
    if (auto struct_literal = value.TryAs<SemIR::StructLiteral>();
        struct_literal &&
        struct_literal->elements_id == SemIR::NodeBlockId::Empty) {
      value_id = semantics_ir.GetTypeAllowBuiltinTypes(value_type_id);
    }
  }

  // No builtin conversion applies.
  return value_id;
}

auto Convert(Context& context, Parse::Node parse_node, SemIR::NodeId expr_id,
             ConversionTarget target) -> SemIR::NodeId {
  auto& semantics_ir = context.semantics_ir();
  auto orig_expr_id = expr_id;

  // Start by making sure both sides are valid. If any part is invalid, the
  // result is invalid and we shouldn't error.
  if (semantics_ir.GetNode(expr_id).type_id() == SemIR::TypeId::Error ||
      target.type_id == SemIR::TypeId::Error) {
    return SemIR::NodeId::BuiltinError;
  }

  if (SemIR::GetExpressionCategory(semantics_ir, expr_id) ==
      SemIR::ExpressionCategory::NotExpression) {
    // TODO: We currently encounter this for use of namespaces and functions.
    // We should provide a better diagnostic for inappropriate use of
    // namespace names, and allow use of functions as values.
    CARBON_DIAGNOSTIC(UseOfNonExpressionAsValue, Error,
                      "Expression cannot be used as a value.");
    context.emitter().Emit(semantics_ir.GetNode(expr_id).parse_node(),
                           UseOfNonExpressionAsValue);
    return SemIR::NodeId::BuiltinError;
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
            context.semantics_ir().StringifyType(target.type_id, true));
      })) {
    return SemIR::NodeId::BuiltinError;
  }

  // Check whether any builtin conversion applies.
  expr_id = PerformBuiltinConversion(context, parse_node, expr_id, target);
  if (expr_id == SemIR::NodeId::BuiltinError) {
    return expr_id;
  }

  // If the types don't match at this point, we can't perform the conversion.
  // TODO: Look for an ImplicitAs impl.
  SemIR::Node expr = semantics_ir.GetNode(expr_id);
  if (expr.type_id() != target.type_id) {
    CARBON_DIAGNOSTIC(ImplicitAsConversionFailure, Error,
                      "Cannot implicitly convert from `{0}` to `{1}`.",
                      std::string, std::string);
    context.emitter()
        .Build(parse_node, ImplicitAsConversionFailure,
               semantics_ir.StringifyType(expr.type_id()),
               semantics_ir.StringifyType(target.type_id))
        .Emit();
    return SemIR::NodeId::BuiltinError;
  }

  // Now perform any necessary value category conversions.
  switch (SemIR::GetExpressionCategory(semantics_ir, expr_id)) {
    case SemIR::ExpressionCategory::NotExpression:
    case SemIR::ExpressionCategory::Mixed:
      CARBON_FATAL() << "Unexpected expression " << expr
                     << " after builtin conversions";

    case SemIR::ExpressionCategory::Error:
      return SemIR::NodeId::BuiltinError;

    case SemIR::ExpressionCategory::Initializing:
      if (target.is_initializer()) {
        if (orig_expr_id == expr_id) {
          // Don't fill in the return slot if we created the expression through
          // a conversion. In that case, we will have created it with the
          // target already set.
          // TODO: Find a better way to track whether we need to do this.
          MarkInitializerFor(semantics_ir, expr_id, target.init_id,
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
    case SemIR::ExpressionCategory::EphemeralReference: {
      // If we have a reference and don't want one, form a value binding.
      if (target.kind != ConversionTarget::ValueOrReference &&
          target.kind != ConversionTarget::Discarded) {
        // TODO: Support types with custom value representations.
        expr_id = context.AddNode(
            SemIR::BindValue(expr.parse_node(), expr.type_id(), expr_id));
      }
      break;
    }

    case SemIR::ExpressionCategory::Value:
      break;
  }

  // Perform a final destination store, if necessary.
  if (target.kind == ConversionTarget::FullInitializer) {
    if (auto init_rep =
            SemIR::GetInitializingRepresentation(semantics_ir, target.type_id);
        init_rep.kind == SemIR::InitializingRepresentation::ByCopy) {
      target.init_block->InsertHere();
      expr_id = context.AddNode(SemIR::InitializeFrom(
          parse_node, target.type_id, expr_id, target.init_id));
    }
  }

  return expr_id;
}

auto Initialize(Context& context, Parse::Node parse_node,
                SemIR::NodeId target_id, SemIR::NodeId value_id)
    -> SemIR::NodeId {
  PendingBlock target_block(context);
  return Convert(
      context, parse_node, value_id,
      {.kind = ConversionTarget::Initializer,
       .type_id = context.semantics_ir().GetNode(target_id).type_id(),
       .init_id = target_id,
       .init_block = &target_block});
}

auto ConvertToValueExpression(Context& context, SemIR::NodeId expr_id)
    -> SemIR::NodeId {
  auto expr = context.semantics_ir().GetNode(expr_id);
  return Convert(context, expr.parse_node(), expr_id,
                 {.kind = ConversionTarget::Value, .type_id = expr.type_id()});
}

auto ConvertToValueOrReferenceExpression(Context& context,
                                         SemIR::NodeId expr_id)
    -> SemIR::NodeId {
  auto expr = context.semantics_ir().GetNode(expr_id);
  return Convert(
      context, expr.parse_node(), expr_id,
      {.kind = ConversionTarget::ValueOrReference, .type_id = expr.type_id()});
}

auto ConvertToValueOfType(Context& context, Parse::Node parse_node,
                          SemIR::NodeId value_id, SemIR::TypeId type_id)
    -> SemIR::NodeId {
  return Convert(context, parse_node, value_id,
                 {.kind = ConversionTarget::Value, .type_id = type_id});
}

auto ConvertToBoolValue(Context& context, Parse::Node parse_node,
                        SemIR::NodeId value_id) -> SemIR::NodeId {
  return ConvertToValueOfType(
      context, parse_node, value_id,
      context.GetBuiltinType(SemIR::BuiltinKind::BoolType));
}

auto ConvertCallArgs(Context& context, Parse::Node call_parse_node,
                     SemIR::NodeBlockId arg_refs_id,
                     Parse::Node param_parse_node,
                     SemIR::NodeBlockId param_refs_id, bool has_return_slot)
    -> bool {
  // If both arguments and parameters are empty, return quickly. Otherwise,
  // we'll fetch both so that errors are consistent.
  if (arg_refs_id == SemIR::NodeBlockId::Empty &&
      param_refs_id == SemIR::NodeBlockId::Empty) {
    return true;
  }

  auto arg_refs = context.semantics_ir().GetNodeBlock(arg_refs_id);
  auto param_refs = context.semantics_ir().GetNodeBlock(param_refs_id);

  if (has_return_slot) {
    // There's no entry in the parameter block for the return slot, so ignore
    // the corresponding entry in the argument block.
    // TODO: Consider adding the return slot to the parameter list.
    CARBON_CHECK(!arg_refs.empty()) << "missing return slot";
    arg_refs = arg_refs.drop_back();
  }

  // If sizes mismatch, fail early.
  if (arg_refs.size() != param_refs.size()) {
    CARBON_DIAGNOSTIC(CallArgCountMismatch, Error,
                      "{0} argument(s) passed to function expecting "
                      "{1} argument(s).",
                      int, int);
    CARBON_DIAGNOSTIC(InCallToFunction, Note,
                      "Calling function declared here.");
    context.emitter()
        .Build(call_parse_node, CallArgCountMismatch, arg_refs.size(),
               param_refs.size())
        .Note(param_parse_node, InCallToFunction)
        .Emit();
    return false;
  }

  if (param_refs.empty()) {
    return true;
  }

  int diag_param_index;
  DiagnosticAnnotationScope annotate_diagnostics(
      &context.emitter(), [&](auto& builder) {
        CARBON_DIAGNOSTIC(
            InCallToFunctionParam, Note,
            "Initializing parameter {0} of function declared here.", int);
        builder.Note(param_parse_node, InCallToFunctionParam,
                     diag_param_index + 1);
      });

  // Check type conversions per-element.
  for (auto [i, value_id, param_ref] : llvm::enumerate(arg_refs, param_refs)) {
    diag_param_index = i;

    auto as_type_id = context.semantics_ir().GetNode(param_ref).type_id();
    // TODO: Convert to the proper expression category. For now, we assume
    // parameters are all `let` bindings.
    value_id =
        ConvertToValueOfType(context, call_parse_node, value_id, as_type_id);
    if (value_id == SemIR::NodeId::BuiltinError) {
      return false;
    }
    arg_refs[i] = value_id;
  }

  return true;
}

auto ExpressionAsType(Context& context, Parse::Node parse_node,
                      SemIR::NodeId value_id) -> SemIR::TypeId {
  return context.CanonicalizeType(ConvertToValueOfType(
      context, parse_node, value_id, SemIR::TypeId::TypeType));
}

}  // namespace Carbon::Check

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/convert.h"

#include <utility>

#include "common/check.h"
#include "common/vlog.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "toolchain/check/context.h"
#include "toolchain/check/declaration_name_stack.h"
#include "toolchain/check/initializing_expression.h"
#include "toolchain/check/node_block_stack.h"
#include "toolchain/diagnostics/diagnostic_kind.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/node.h"
#include "toolchain/sem_ir/node_kind.h"

namespace Carbon::Check {

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
static auto ConvertTupleToArray(Context& context, SemIR::Node tuple_type,
                                SemIR::Node array_type, SemIR::NodeId value_id,
                                ConversionTarget target) -> SemIR::NodeId {
  auto [array_bound_id, element_type_id] = array_type.GetAsArrayType();
  auto tuple_elem_types_id = tuple_type.GetAsTupleType();
  const auto& tuple_elem_types =
      context.semantics_ir().GetTypeBlock(tuple_elem_types_id);

  auto value = context.semantics_ir().GetNode(value_id);

  // If we're initializing from a tuple literal, we will use its elements
  // directly. Otherwise, materialize a temporary if needed and index into the
  // result.
  llvm::ArrayRef<SemIR::NodeId> literal_elems;
  if (value.kind() == SemIR::NodeKind::TupleLiteral) {
    literal_elems =
        context.semantics_ir().GetNodeBlock(value.GetAsTupleLiteral());
  } else {
    value_id = MaterializeIfInitializing(context, value_id);
  }

  // Check that the tuple is the right size.
  uint64_t array_bound =
      context.semantics_ir().GetArrayBoundValue(array_bound_id);
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
    return_slot_id = target_block->AddNode(SemIR::Node::TemporaryStorage::Make(
        value.parse_node(), target.type_id));
  }

  // Initialize each element of the array from the corresponding element of the
  // tuple.
  // TODO: Annotate diagnostics coming from here with the array element index,
  // if initializing from a tuple literal.
  llvm::SmallVector<SemIR::NodeId> inits;
  inits.reserve(array_bound + 1);
  for (auto [i, src_type_id] : llvm::enumerate(tuple_elem_types)) {
    PendingBlock::DiscardUnusedNodesScope scope(target_block);
    // TODO: Add a new node kind for indexing an array at a constant index
    // so that we don't need an integer literal node here.
    auto index_id = target_block->AddNode(SemIR::Node::IntegerLiteral::Make(
        value.parse_node(),
        context.CanonicalizeType(SemIR::NodeId::BuiltinIntegerType),
        context.semantics_ir().AddIntegerLiteral(llvm::APInt(32, i))));
    auto target_id = target_block->AddNode(SemIR::Node::ArrayIndex::Make(
        value.parse_node(), element_type_id, return_slot_id, index_id));
    // Note, this is computing the source location not the destination, so it
    // goes into the current code block, not into the target block.
    // TODO: Ideally we would also discard this node if it's unused.
    auto src_id = !literal_elems.empty()
                      ? literal_elems[i]
                      : context.AddNode(SemIR::Node::TupleAccess::Make(
                            value.parse_node(), src_type_id, value_id,
                            SemIR::MemberIndex(i)));
    // TODO: This call recurses back into conversion. Switch to an iterative
    // approach.
    auto init_id = Convert(context, value.parse_node(), src_id,
                           {.kind = ConversionTarget::FullInitializer,
                            .type_id = element_type_id,
                            .init_id = target_id,
                            .init_block = target_block});
    if (init_id == SemIR::NodeId::BuiltinError) {
      return SemIR::NodeId::BuiltinError;
    }
    inits.push_back(init_id);
  }

  // The last element of the refs block contains the return slot for the array
  // initialization. Flush the temporary here if we didn't insert it earlier.
  target_block->InsertHere();
  inits.push_back(return_slot_id);

  return context.AddNode(
      SemIR::Node::ArrayInit::Make(value.parse_node(), target.type_id, value_id,
                                   context.semantics_ir().AddNodeBlock(inits)));
}

// Performs a conversion from a tuple to a tuple type. Does not perform a
// final conversion to the requested expression category.
static auto ConvertTupleToTuple(Context& context, SemIR::Node src_type,
                                SemIR::Node dest_type, SemIR::NodeId value_id,
                                ConversionTarget target) -> SemIR::NodeId {
  auto src_elem_types =
      context.semantics_ir().GetTypeBlock(src_type.GetAsTupleType());
  auto dest_elem_types =
      context.semantics_ir().GetTypeBlock(dest_type.GetAsTupleType());

  auto value = context.semantics_ir().GetNode(value_id);

  // If we're initializing from a tuple literal, we will use its elements
  // directly. Otherwise, materialize a temporary if needed and index into the
  // result.
  llvm::ArrayRef<SemIR::NodeId> literal_elems;
  if (value.kind() == SemIR::NodeKind::TupleLiteral) {
    literal_elems =
        context.semantics_ir().GetNodeBlock(value.GetAsTupleLiteral());
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
      : SemIR::GetInitializingRepresentation(context.semantics_ir(),
                                             target.type_id)
                  .kind == SemIR::InitializingRepresentation::InPlace
          ? ConversionTarget::FullInitializer
          : ConversionTarget::Initializer;

  // Initialize each element of the destination from the corresponding element
  // of the source.
  // TODO: Annotate diagnostics coming from here with the element index.
  CopyOnWriteBlock new_block(context.semantics_ir(),
                             value.kind() == SemIR::NodeKind::TupleLiteral
                                 ? value.GetAsTupleLiteral()
                                 : SemIR::NodeBlockId::Invalid,
                             src_elem_types.size());
  for (auto [i, src_type_id, dest_type_id] :
       llvm::enumerate(src_elem_types, dest_elem_types)) {
    PendingBlock::DiscardUnusedNodesScope scope(target.init_block);
    auto target_id =
        is_init ? target.init_block->AddNode(SemIR::Node::TupleAccess::Make(
                      value.parse_node(), dest_type_id, target.init_id,
                      SemIR::MemberIndex(i)))
                : SemIR::NodeId::Invalid;
    // Note, this is computing the source location not the destination, so it
    // goes into the current code block, not into the target block.
    // TODO: Ideally we would also discard this node if it's unused.
    auto src_id = !literal_elems.empty()
                      ? literal_elems[i]
                      : context.AddNode(SemIR::Node::TupleAccess::Make(
                            value.parse_node(), src_type_id, value_id,
                            SemIR::MemberIndex(i)));
    // TODO: This call recurses back into conversion. Switch to an iterative
    // approach.
    auto init_id = Convert(context, value.parse_node(), src_id,
                           {.kind = inner_kind,
                            .type_id = dest_type_id,
                            .init_id = target_id,
                            .init_block = target.init_block});
    if (init_id == SemIR::NodeId::BuiltinError) {
      return SemIR::NodeId::BuiltinError;
    }
    new_block.Set(i, init_id);
  }

  return context.AddNode(
      is_init
          ? SemIR::Node::TupleInit::Make(value.parse_node(), target.type_id,
                                         value_id, new_block.id())
          : SemIR::Node::TupleValue::Make(value.parse_node(), target.type_id,
                                          value_id, new_block.id()));
}

// Performs a conversion from a struct to a struct type. Does not perform a
// final conversion to the requested expression category.
static auto ConvertStructToStruct(Context& context, SemIR::Node src_type,
                                  SemIR::Node dest_type, SemIR::NodeId value_id,
                                  ConversionTarget target) -> SemIR::NodeId {
  auto src_elem_fields =
      context.semantics_ir().GetNodeBlock(src_type.GetAsStructType());
  auto dest_elem_fields =
      context.semantics_ir().GetNodeBlock(dest_type.GetAsStructType());

  auto value = context.semantics_ir().GetNode(value_id);

  // If we're initializing from a struct literal, we will use its elements
  // directly. Otherwise, materialize a temporary if needed and index into the
  // result.
  llvm::ArrayRef<SemIR::NodeId> literal_elems;
  if (value.kind() == SemIR::NodeKind::StructLiteral) {
    literal_elems =
        context.semantics_ir().GetNodeBlock(value.GetAsStructLiteral());
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
      : SemIR::GetInitializingRepresentation(context.semantics_ir(),
                                             target.type_id)
                  .kind == SemIR::InitializingRepresentation::InPlace
          ? ConversionTarget::FullInitializer
          : ConversionTarget::Initializer;

  // Initialize each element of the destination from the corresponding element
  // of the source.
  // TODO: Annotate diagnostics coming from here with the element index.
  CopyOnWriteBlock new_block(context.semantics_ir(),
                             value.kind() == SemIR::NodeKind::StructLiteral
                                 ? value.GetAsStructLiteral()
                                 : SemIR::NodeBlockId::Invalid,
                             src_elem_fields.size());
  for (auto [i, src_field_id, dest_field_id] :
       llvm::enumerate(src_elem_fields, dest_elem_fields)) {
    auto [src_name_id, src_type_id] =
        context.semantics_ir().GetNode(src_field_id).GetAsStructTypeField();
    auto [dest_name_id, dest_type_id] =
        context.semantics_ir().GetNode(dest_field_id).GetAsStructTypeField();
    if (src_name_id != dest_name_id) {
      CARBON_DIAGNOSTIC(
          StructInitFieldNameMismatch, Error,
          "Mismatched names for field {0} in struct initialization: "
          "source has field name `{1}`, destination has field name `{2}`.",
          size_t, llvm::StringRef, llvm::StringRef);
      context.emitter().Emit(value.parse_node(), StructInitFieldNameMismatch,
                             i + 1,
                             context.semantics_ir().GetString(src_name_id),
                             context.semantics_ir().GetString(dest_name_id));
      return SemIR::NodeId::BuiltinError;
    }
    PendingBlock::DiscardUnusedNodesScope scope(target.init_block);
    auto target_id =
        is_init ? target.init_block->AddNode(SemIR::Node::StructAccess::Make(
                      value.parse_node(), dest_type_id, target.init_id,
                      SemIR::MemberIndex(i)))
                : SemIR::NodeId::Invalid;
    // Note, this is computing the source location not the destination, so it
    // goes into the current code block, not into the target block.
    // TODO: Ideally we would also discard this node if it's unused.
    auto src_id = !literal_elems.empty()
                      ? literal_elems[i]
                      : context.AddNode(SemIR::Node::StructAccess::Make(
                            value.parse_node(), src_type_id, value_id,
                            SemIR::MemberIndex(i)));
    // TODO: This call recurses back into conversion. Switch to an iterative
    // approach.
    auto init_id = Convert(context, value.parse_node(), src_id,
                           {.kind = inner_kind,
                            .type_id = dest_type_id,
                            .init_id = target_id,
                            .init_block = target.init_block});
    if (init_id == SemIR::NodeId::BuiltinError) {
      return SemIR::NodeId::BuiltinError;
    }
    new_block.Set(i, init_id);
  }

  return context.AddNode(
      is_init
          ? SemIR::Node::StructInit::Make(value.parse_node(), target.type_id,
                                          value_id, new_block.id())
          : SemIR::Node::StructValue::Make(value.parse_node(), target.type_id,
                                           value_id, new_block.id()));
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
  auto value = context.semantics_ir().GetNode(value_id);
  auto value_type_id = value.type_id();
  auto target_type_node = context.semantics_ir().GetNode(
      context.semantics_ir().GetTypeAllowBuiltinTypes(target.type_id));

  // Various forms of implicit conversion are supported as builtin conversions
  // rather than being implemented purely in the Carbon prelude as `impl`s of
  // `ImplicitAs`. There are a few reasons we need to perform some of these
  // conversions as builtins:
  //
  // 1) Conversions from struct and tuple *literals* have special rules that
  //    cannot be implemented by invoking `ImplicitAs`. Specifically, we must
  //    recurse into the elements of the literal before performing
  //    initialization in order to avoid unnecessary conversions between
  //    expression categories.
  // 2) (Not implemented yet) Conversion of a facet to a facet type depends on
  //    the value of the facet, not only its type.
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
  // These builtin conversions all correspond to `final impl`s in the library,
  // so we don't need to worry about `ImplicitAs` being specialized in any of
  // these cases.

  // If the value is already of the right kind and expression category, there's
  // nothing to do. Performing a conversion would decompose and rebuild tuples
  // and structs, so it's important that we bail out early in this case.
  if (value_type_id == target.type_id &&
      IsValidExpressionCategoryForConversionTarget(
          SemIR::GetExpressionCategory(context.semantics_ir(), value_id),
          target.kind)) {
    return value_id;
  }

  // A tuple (T1, T2, ..., Tn) converts to (U1, U2, ..., Un) if each Ti
  // converts to Ui.
  if (target_type_node.kind() == SemIR::NodeKind::TupleType) {
    auto value_type_node = context.semantics_ir().GetNode(
        context.semantics_ir().GetTypeAllowBuiltinTypes(value_type_id));
    if (value_type_node.kind() == SemIR::NodeKind::TupleType) {
      return ConvertTupleToTuple(context, value_type_node, target_type_node,
                                 value_id, target);
    }
  }

  // A struct {.f_1: T_1, .f_2: T_2, ..., .f_n: T_n} converts to
  // {.f_p(1): U_p(1), .f_p(2): U_p(2), ..., .f_p(n): U_p(n)} if
  // (p(1), ..., p(n)) is a permutation of (1, ..., n) and each Ti converts
  // to Ui.
  if (target_type_node.kind() == SemIR::NodeKind::StructType) {
    auto value_type_node = context.semantics_ir().GetNode(
        context.semantics_ir().GetTypeAllowBuiltinTypes(value_type_id));
    if (value_type_node.kind() == SemIR::NodeKind::StructType) {
      return ConvertStructToStruct(context, value_type_node, target_type_node,
                                   value_id, target);
    }
  }

  // A tuple (T1, T2, ..., Tn) converts to [T; n] if each Ti converts to T.
  if (target_type_node.kind() == SemIR::NodeKind::ArrayType) {
    auto value_type_node = context.semantics_ir().GetNode(
        context.semantics_ir().GetTypeAllowBuiltinTypes(value_type_id));
    if (value_type_node.kind() == SemIR::NodeKind::TupleType) {
      return ConvertTupleToArray(context, value_type_node, target_type_node,
                                 value_id, target);
    }
  }

  if (target.type_id == SemIR::TypeId::TypeType) {
    // A tuple of types converts to type `type`.
    // TODO: This should apply even for non-literal tuples.
    if (value.kind() == SemIR::NodeKind::TupleLiteral) {
      auto tuple_block_id = value.GetAsTupleLiteral();
      llvm::SmallVector<SemIR::TypeId> type_ids;
      // If it is empty tuple type, we don't fetch anything.
      if (tuple_block_id != SemIR::NodeBlockId::Empty) {
        const auto& tuple_block =
            context.semantics_ir().GetNodeBlock(tuple_block_id);
        for (auto tuple_node_id : tuple_block) {
          // TODO: This call recurses back into conversion. Switch to an
          // iterative approach.
          type_ids.push_back(
              ExpressionAsType(context, parse_node, tuple_node_id));
        }
      }
      auto tuple_type_id =
          context.CanonicalizeTupleType(parse_node, std::move(type_ids));
      return context.semantics_ir().GetTypeAllowBuiltinTypes(tuple_type_id);
    }

    // `{}` converts to `{} as type`.
    // TODO: This conversion should also be performed for a non-literal value
    // of type `{}`.
    if (value.kind() == SemIR::NodeKind::StructLiteral &&
        value.GetAsStructLiteral() == SemIR::NodeBlockId::Empty) {
      value_id = context.semantics_ir().GetTypeAllowBuiltinTypes(value_type_id);
    }
  }

  // No builtin conversion applies.
  return value_id;
}

auto Convert(Context& context, Parse::Node parse_node, SemIR::NodeId expr_id,
             ConversionTarget target) -> SemIR::NodeId {
  auto orig_expr_id = expr_id;

  // Start by making sure both sides are valid. If any part is invalid, the
  // result is invalid and we shouldn't error.
  if (expr_id == SemIR::NodeId::BuiltinError) {
    return expr_id;
  }
  if (context.semantics_ir().GetNode(expr_id).type_id() ==
          SemIR::TypeId::Error ||
      target.type_id == SemIR::TypeId::Error) {
    return SemIR::NodeId::BuiltinError;
  }

  if (SemIR::GetExpressionCategory(context.semantics_ir(), expr_id) ==
      SemIR::ExpressionCategory::NotExpression) {
    // TODO: We currently encounter this for use of namespaces and functions.
    // We should provide a better diagnostic for inappropriate use of
    // namespace names, and allow use of functions as values.
    CARBON_DIAGNOSTIC(UseOfNonExpressionAsValue, Error,
                      "Expression cannot be used as a value.");
    context.emitter().Emit(context.semantics_ir().GetNode(expr_id).parse_node(),
                           UseOfNonExpressionAsValue);
    return SemIR::NodeId::BuiltinError;
  }

  // Check whether any builtin conversion applies.
  expr_id = PerformBuiltinConversion(context, parse_node, expr_id, target);
  if (expr_id == SemIR::NodeId::BuiltinError) {
    return expr_id;
  }

  // If the types don't match at this point, we can't perform the conversion.
  // TODO: Look for an ImplicitAs impl.
  SemIR::Node expr = context.semantics_ir().GetNode(expr_id);
  if (expr.type_id() != target.type_id) {
    CARBON_DIAGNOSTIC(ImplicitAsConversionFailure, Error,
                      "Cannot implicitly convert from `{0}` to `{1}`.",
                      std::string, std::string);
    context.emitter()
        .Build(parse_node, ImplicitAsConversionFailure,
               context.semantics_ir().StringifyType(expr.type_id()),
               context.semantics_ir().StringifyType(target.type_id))
        .Emit();
    return SemIR::NodeId::BuiltinError;
  }

  // Now perform any necessary value category conversions.
  switch (SemIR::GetExpressionCategory(context.semantics_ir(), expr_id)) {
    case SemIR::ExpressionCategory::NotExpression:
    case SemIR::ExpressionCategory::Mixed:
      CARBON_FATAL() << "Unexpected expression " << expr
                     << " after builtin conversions";

    case SemIR::ExpressionCategory::Initializing:
      if (target.is_initializer()) {
        if (orig_expr_id == expr_id) {
          // Don't fill in the return slot if we created the expression through
          // a conversion. In that case, we will have created it with the
          // target already set.
          // TODO: Find a better way to track whether we need to do this.
          MarkInitializerFor(context, expr_id, target.init_id,
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
        expr_id = context.AddNode(SemIR::Node::BindValue::Make(
            expr.parse_node(), expr.type_id(), expr_id));
      }
      break;
    }

    case SemIR::ExpressionCategory::Value:
      break;
  }

  // Perform a final destination store, if necessary.
  if (target.kind == ConversionTarget::FullInitializer) {
    if (auto init_rep = SemIR::GetInitializingRepresentation(
            context.semantics_ir(), target.type_id);
        init_rep.kind == SemIR::InitializingRepresentation::ByCopy) {
      target.init_block->InsertHere();
      expr_id = context.AddNode(SemIR::Node::InitializeFrom::Make(
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
      context.CanonicalizeType(SemIR::NodeId::BuiltinBoolType));
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

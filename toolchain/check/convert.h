// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CONVERT_H_
#define CARBON_TOOLCHAIN_CHECK_CONVERT_H_

#include "toolchain/check/context.h"
#include "toolchain/check/pending_block.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Description of the target of a conversion.
struct ConversionTarget {
  enum Kind : int8_t {
    // Convert to a value of type `type_id`.
    Value,
    // Convert to either a value or a reference of type `type_id`.
    ValueOrRef,
    // Convert to a durable reference of type `type_id`.
    DurableRef,
    // Convert to a reference, suitable for binding to a reference parameter.
    // This allows both durable and ephemeral references. The restriction that
    // only a `ref self` parameter can bind to an ephemeral reference is
    // enforced separately when handling `ref` tags on call arguments.
    RefParam,
    // Equivalent to RefParam, except that the source expression is not required
    // to be marked with a `ref` tag, such as an argument to a `ref self`
    // parameter or an operator operand.
    UnmarkedRefParam,
    // Convert to a reference of type `type_id`, for use as the argument to a
    // C++ thunk.
    CppThunkRef,
    // Convert for an explicit `as` cast. This allows any expression category
    // as the result, and uses the `As` interface instead of the `ImplicitAs`
    // interface.
    ExplicitAs,
    // Convert for an explicit `unsafe as` cast. This allows any expression
    // category as the result, and uses the `UnsafeAs` interface instead of the
    // `As` or `ImplicitAs` interface.
    ExplicitUnsafeAs,
    // The result of the conversion is discarded. It can't be an initializing
    // expression, but can be anything else.
    Discarded,
    // Convert to an initializing expression, which a subsequent operation (such
    // as `InitializeFrom` or `Temporary`) can use to initialize `storage_id`.
    // `storage_id` is only used if `type_id` has an in-place initializing
    // representation; otherwise, `storage_id` can be `None`, and the resulting
    // initializing expression can be used to initialize any object of the
    // appropriate type.
    Initializer,
    // Convert to an initializing expression, and use it to initialize
    // `storage_id` (which must not be `None`).
    FullInitializer,
    Last = FullInitializer
  };
  // The kind of the target for this conversion.
  Kind kind;
  // The target type for the conversion.
  SemIR::TypeId type_id;
  // The storage being initialized, if any.
  SemIR::InstId storage_id = SemIR::InstId::None;
  // For an initializer, a block of pending instructions that `storage_id`
  // depends on, and that can be discarded if `storage_id` is not accessed.
  PendingBlock* storage_access_block = nullptr;
  // Whether failure of conversion is an error and is diagnosed to the user.
  // When looking for a possible conversion but with graceful fallback, diagnose
  // should be false.
  bool diagnose = true;

  // Are we converting this value into an initializer for an object?
  auto is_initializer() const -> bool {
    return kind == Initializer || kind == FullInitializer;
  }
  // Is this some kind of explicit `as` conversion?
  auto is_explicit_as() const -> bool {
    return kind == ExplicitAs || kind == ExplicitUnsafeAs;
  }
};

// Convert a value to another type and expression category.
// TODO: The `vtable_id` parameter is too much of a special case here, and
// should be removed - once partial classes are implemented, the vtable pointer
// initialization will be done not in this conversion, but during initialization
// of the object of non-partial class type from the object of partial class
// type.
auto Convert(Context& context, SemIR::LocId loc_id, SemIR::InstId expr_id,
             ConversionTarget target,
             SemIR::ClassType* vtable_class_type = nullptr) -> SemIR::InstId;

// Converts `value_id` to an initializing expression of the type of
// `storage_id`, and returns the possibly-converted initializing expression. If
// initialization is in-place, `storage_id` is used as the in-place storage;
// otherwise it is used only to determine the target type. The caller is
// responsible for assigning the returned initializing expression to the target
// using a suitable node for the kind of initialization.
//
// TODO: Consider making the target type a separate parameter, and making
// storage_id optional.
auto Initialize(Context& context, SemIR::LocId loc_id, SemIR::InstId storage_id,
                SemIR::InstId value_id) -> SemIR::InstId;

// Convert the given expression to a value expression of the same type.
auto ConvertToValueExpr(Context& context, SemIR::InstId expr_id)
    -> SemIR::InstId;

// Convert the given expression to a value or reference expression of the same
// type.
auto ConvertToValueOrRefExpr(Context& context, SemIR::InstId expr_id)
    -> SemIR::InstId;

// Converts `expr_id` to a value expression of type `type_id`.
auto ConvertToValueOfType(Context& context, SemIR::LocId loc_id,
                          SemIR::InstId expr_id, SemIR::TypeId type_id)
    -> SemIR::InstId;

// Convert the given expression to a value or reference expression of the given
// type.
auto ConvertToValueOrRefOfType(Context& context, SemIR::LocId loc_id,
                               SemIR::InstId expr_id, SemIR::TypeId type_id)
    -> SemIR::InstId;

// Attempted to convert `expr_id` to a value expression of type `type_id`, with
// graceful failure, which does not result in diagnostics. An ErrorInst
// instruction is still returned on failure.
auto TryConvertToValueOfType(Context& context, SemIR::LocId loc_id,
                             SemIR::InstId expr_id, SemIR::TypeId type_id)
    -> SemIR::InstId;

// Converts `value_id` to a value expression of type `bool`.
auto ConvertToBoolValue(Context& context, SemIR::LocId loc_id,
                        SemIR::InstId value_id) -> SemIR::InstId;

// Converts `value_id` to type `type_id` for an `as` expression.
auto ConvertForExplicitAs(Context& context, Parse::NodeId as_node,
                          SemIR::InstId value_id, SemIR::TypeId type_id,
                          bool unsafe) -> SemIR::InstId;

// Implicitly converts a set of arguments to match the parameter types in a
// function call. Returns a block containing the converted implicit and explicit
// argument values for runtime parameters. `is_operator_syntax` indicates that
// this call was generated from an operator rather than from function call
// syntax, so arguments to `ref` parameters aren't required to have `ref` tags.
auto ConvertCallArgs(Context& context, SemIR::LocId call_loc_id,
                     SemIR::InstId self_id,
                     llvm::ArrayRef<SemIR::InstId> arg_refs,
                     llvm::ArrayRef<SemIR::InstId> return_arg_ids,
                     const SemIR::Function& callee,
                     SemIR::SpecificId callee_specific_id,
                     bool is_operator_syntax) -> SemIR::InstBlockId;

// A type that has been converted for use as a type expression.
struct TypeExpr {
  static const TypeExpr None;

  // Returns a TypeExpr describing a type with no associated spelling or type
  // sugar.
  static auto ForUnsugared(Context& context, SemIR::TypeId type_id) -> TypeExpr;

  // The converted expression of type `type`, or `ErrorInst::InstId`.
  SemIR::TypeInstId inst_id;
  // The corresponding type, or `ErrorInst::TypeId`.
  SemIR::TypeId type_id;
};

inline constexpr TypeExpr TypeExpr::None = {.inst_id = SemIR::TypeInstId::None,
                                            .type_id = SemIR::TypeId::None};

// Converts an expression for use as a type.
//
// If `diagnose` is true, errors are diagnosed to the user. Set it to false when
// looking to see if a conversion is possible but with graceful fallback.
//
// TODO: Most of the callers of this function discard the `inst_id` and lose
// track of the conversion. In most cases we should be retaining that as the
// operand of some downstream instruction.
auto ExprAsType(Context& context, SemIR::LocId loc_id, SemIR::InstId value_id,
                bool diagnose = true) -> TypeExpr;

// Converts an expression for use as a form. If the expression is a type
// expression, it is interpreted as an initializing form.
auto ExprAsReturnForm(Context& context, SemIR::LocId loc_id,
                      SemIR::InstId value_id) -> Context::FormExpr;

// Handles an expression whose result value is unused.
auto DiscardExpr(Context& context, SemIR::InstId expr_id) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CONVERT_H_

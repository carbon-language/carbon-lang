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
    // Perform no conversion. The source expression must already have type
    // `type_id`.
    NoOp,
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
    // Convert to an initializing expression that uses `type_id`'s initializing
    // representation. The resulting expression will usually be a
    // repr-initializing expression, but may be an in-place initializing
    // expression if the source expression was. If `storage_id` is present, it
    // is used as the storage argument for the converted expression, and it must
    // be present if the initializing representation might be in-place.
    Initializing,
    // Convert to an in-place initializing expression whose storage is
    // designated by `storage_id` (which must not be `None`)
    InPlaceInitializing,
    Last = InPlaceInitializing
  };
  // The kind of the target for this conversion.
  Kind kind;
  // The target type for the conversion.
  SemIR::TypeId type_id;
  // The storage being initialized, if any.
  SemIR::InstId storage_id = SemIR::InstId::None;
  // For an initializer, a block of pending instructions that `storage_id`
  // depends on, and that can be discarded if `storage_id` is not accessed.
  // If this is not null or empty, its last element must be storage_id.
  PendingBlock* storage_access_block = nullptr;
  // Whether failure of conversion is an error and is diagnosed to the user.
  // When looking for a possible conversion but with graceful fallback, diagnose
  // should be false.
  bool diagnose = true;

  // Are we converting this value into an initializer for an object?
  auto is_initializer() const -> bool {
    return kind == Initializing || kind == InPlaceInitializing;
  }
  // Is this some kind of explicit `as` conversion?
  auto is_explicit_as() const -> bool {
    return kind == ExplicitAs || kind == ExplicitUnsafeAs;
  }
};

// Convert a value to another type and expression category.
auto Convert(Context& context, SemIR::LocId loc_id, SemIR::InstId expr_id,
             ConversionTarget target) -> SemIR::InstId;

// Converts `value_id` to an initializing expression of the type of
// `storage_id`, and returns the possibly-converted initializing expression.
// `storage_id` is used as the storage argument of the resulting expression
// except as noted below, and when it is used as the storage argument it must
// precede `value_id`. The caller is responsible for passing the result to an
// inst that is documented as consuming it, such as `Assign`.
//
// `for_return` indicates that this conversion is initializing the operand of a
// `return` statement. This means that `storage_id` will be the return slot
// parameter, which isn't valid to access if the type's initializing
// representation is not in-place, so in that case `storage_id` will be used
// solely for its type.
//
// TODO: Consider making the target type a separate parameter, and making
// storage_id optional.
auto Initialize(Context& context, SemIR::LocId loc_id, SemIR::InstId storage_id,
                SemIR::InstId value_id, bool for_return = false)
    -> SemIR::InstId;

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

// Converts an expression in a form position for use as a form.
//
// Note that the right-hand side of a `->` return type declaration is not
// a form position for this purpose, because it uses a special syntax to specify
// forms. `ReturnExprAsForm` should be used instead in that case.
//
// `diagnose` has the same effect as in `ExprAsType`.
auto FormExprAsForm(Context& context, SemIR::LocId loc_id,
                    SemIR::InstId value_id) -> Context::FormExpr;

// Evaluates an expression in the return-type position (following `->`, not
// `->?`) for use as a form, following the special-case language rules for
// evaluating an expression in that position.
auto ReturnExprAsForm(Context& context, SemIR::LocId loc_id,
                      SemIR::InstId value_id) -> Context::FormExpr;

// Handles an expression whose result value is unused.
auto DiscardExpr(Context& context, SemIR::InstId expr_id) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CONVERT_H_

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
    // designated by `storage_id` (which must not be `None`).
    InPlaceInitializing,
    Last = InPlaceInitializing
  };
  // The kind of the target for this conversion.
  Kind kind;
  // The target type for the conversion.
  SemIR::TypeId type_id;
  // The storage being initialized, if any. It must be valid to reference this
  // instruction after splicing in `storage_access_block` (if specified), so it
  // must either dominate the initializer or be one of the instructions in
  // `storage_access_block`.
  SemIR::InstId storage_id = SemIR::InstId::None;
  // For an initializer, a block of pending instructions that `storage_id`
  // depends on. This block will be spliced or merged before any reference to
  // `storage_id`, and may be discarded if `storage_id` is not accessed.
  PendingBlock* storage_access_block = nullptr;
  // Whether failure of conversion is an error and is diagnosed to the user.
  // When looking for a possible conversion but with graceful fallback,
  // `diagnose` should be false. If `diagnose` is false, an `ErrorInst` may be
  // returned, but it must be discarded.
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

// Performs initialization of `storage_id` from the expression `value_id`, which
// is converted to an initializing expression of the type of `storage_id` if
// necessary, and returns the possibly-converted initializing expression.
//
// `storage_id` is used as the storage argument of the resulting expression
// except as noted below. As a consequence, `storage_id` must dominate
// `value_id` and its subexpressions.  This will typically only be the case if
// `storage_id` syntactically precedes `value_id`. Otherwise, some action will
// need to be taken to reorder the code, such as instead calling `Initialize`
// with a pending block containing `storage_id`, or creating a separate
// `InstBlock` to hold either the storage or the initializer.
//
// `for_return` indicates that this conversion is initializing the operand of a
// `return` statement. This means that `storage_id` will be the return slot
// parameter, which isn't valid to access if the type's initializing
// representation is not in-place, so in that case `storage_id` will be used
// solely for its type.
//
// This function does not guarantee to perform an in-place initialization, so
// the caller is responsible for passing the returned `InstId` to an inst that
// is documented as consuming it, such as `Assign`.
auto InitializeExisting(Context& context, SemIR::LocId loc_id,
                        SemIR::InstId storage_id, SemIR::InstId value_id,
                        bool for_return = false) -> SemIR::InstId;

// Result of Initialize.
struct InitializeResult {
  // The storage location that contains the initialized value. This may be
  // different from the `storage_id` that was passed to `Initialize` if the
  // storage block was written over existing instructions rather than being
  // spliced in.
  SemIR::InstId storage_id;
  // The converted initializing expression used to initialize the storage.
  SemIR::InstId init_id;
};

// Performs initialization of `storage_id` from the expression `value_id`, which
// is converted to an initializing expression of the type of `storage_id` if
// necessary. `storage_access_block` should be used to supply a pending block
// that allocates the storage, and typically contains `storage_id`. The target
// of the initialization will be either `storage_id` itself, or an existing
// storage argument instruction that is overwritten to hold a copy of
// `storage_id` as an optimization for SemIR compactness.
//
// The storage instruction will only be written over an existing instruction if
// it is the sole instruction in the pending block. This is expected to be a
// common case. After this happens, the copy of the instruction in the pending
// block is expected to be unreachable from the SemIR::File. As a result, the
// `storage_id` instruction should not be referenced again after calling this
// function, and this function takes it by rvalue reference to remind the caller
// of this.
//
// If the overwrite optimization is not performed, `storage_access_block` will
// be inserted before any use of the storage by the initializer, and will be
// inserted even if the initializer does not actually use the storage. It must
// be valid to reference `storage_id` after splicing in `storage_access_block`,
// so `storage_id` must either dominate the initializer (but see the TODO below)
// or be one of the instructions in `storage_access_block`. If `storage_id` is
// known to always dominate the initializer, `InitializeExisting` should be used
// instead.
//
// TODO: We don't have an implementation of a proper dominance check, so we
// fake one up by comparing the order in which the insts were created.
//
// This function does not guarantee to perform an in-place initialization, so
// the caller is responsible for passing the returned `inst_id` to an inst that
// is documented as consuming it, such as `Assign`.
auto Initialize(Context& context, SemIR::LocId loc_id,
                SemIR::InstId&& storage_id, PendingBlock&& storage_access_block,
                SemIR::InstId value_id) -> InitializeResult;

// Convert the given expression to a value expression of the same type.
auto ConvertToValueExpr(Context& context, SemIR::InstId expr_id)
    -> SemIR::InstId;

// Convert the given expression to a value or reference expression of the same
// type.
auto ConvertToValueOrRefExpr(Context& context, SemIR::InstId expr_id)
    -> SemIR::InstId;

// Converts `expr_id` to a value expression of type `type_id`.
//
// If `diagnose` is true, errors are diagnosed to the user. Set it to false when
// looking to see if a conversion is possible but with graceful fallback. If
// `diagnose` is false, an `ErrorInst` may be returned, but it must be
// discarded.
auto ConvertToValueOfType(Context& context, SemIR::LocId loc_id,
                          SemIR::InstId expr_id, SemIR::TypeId type_id,
                          bool diagnose = true) -> SemIR::InstId;

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
// argument values for runtime parameters. `is_desugared` indicates that this
// call was produced by desugaring, not written as a function call in user code,
// so arguments to `ref` parameters aren't required to have `ref` tags.
auto ConvertCallArgs(Context& context, SemIR::LocId call_loc_id,
                     SemIR::InstId self_id,
                     llvm::ArrayRef<SemIR::InstId> arg_refs,
                     SemIR::InstId return_arg_id, const SemIR::Function& callee,
                     SemIR::SpecificId callee_specific_id, bool is_desugared)
    -> SemIR::InstBlockId;

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
// looking to see if a conversion is possible but with graceful fallback. If
// `diagnose` is false, an `ErrorInst` may be returned, but it must be
// discarded.
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

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
    // Convert for an explicit `as` cast. This allows any expression category
    // as the result, and uses the `As` interface instead of the `ImplicitAs`
    // interface.
    ExplicitAs,
    // The result of the conversion is discarded. It can't be an initializing
    // expression, but can be anything else.
    Discarded,
    // Convert to an initializer for the object denoted by `init_id`.
    Initializer,
    // Convert to an initializer for the object denoted by `init_id`,
    // including a final destination store if needed.
    FullInitializer,
    Last = FullInitializer
  };
  // The kind of the target for this conversion.
  Kind kind;
  // The target type for the conversion.
  SemIR::TypeId type_id;
  // For an initializer, the object being initialized.
  SemIR::InstId init_id = SemIR::InstId::None;
  // For an initializer, a block of pending instructions that are needed to
  // form the value of `init_id`, and that can be discarded if no
  // initialization is needed.
  PendingBlock* init_block = nullptr;
  // Whether failure of conversion is an error and is diagnosed to the user.
  // When looking for a possible conversion but with graceful fallback, diagnose
  // should be false.
  bool diagnose = true;

  // Are we converting this value into an initializer for an object?
  auto is_initializer() const -> bool {
    return kind == Initializer || kind == FullInitializer;
  }
};

// Convert a value to another type and expression category.
// TODO: The `vtable_id` parameter is too much of a special case here, and
// should be removed - once partial classes are implemented, the vtable pointer
// initialization will be done not in this conversion, but during initialization
// of the object of non-partial class time from the object of partial class
// type.
auto Convert(Context& context, SemIR::LocId loc_id, SemIR::InstId expr_id,
             ConversionTarget target,
             SemIR::InstId vtable_id = SemIR::InstId::None) -> SemIR::InstId;

// Performs initialization of `target_id` from `value_id`. Returns the
// possibly-converted initializing expression, which should be assigned to the
// target using a suitable node for the kind of initialization.
auto Initialize(Context& context, SemIR::LocId loc_id, SemIR::InstId target_id,
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
                          SemIR::InstId value_id, SemIR::TypeId type_id)
    -> SemIR::InstId;

// Implicitly converts a set of arguments to match the parameter types in a
// function call. Returns a block containing the converted implicit and explicit
// argument values for runtime parameters.
auto ConvertCallArgs(Context& context, SemIR::LocId call_loc_id,
                     SemIR::InstId self_id,
                     llvm::ArrayRef<SemIR::InstId> arg_refs,
                     SemIR::InstId return_slot_arg_id,
                     const SemIR::Function& callee,
                     SemIR::SpecificId callee_specific_id)
    -> SemIR::InstBlockId;

// A type that has been converted for use as a type expression.
struct TypeExpr {
  // The converted expression of type `type`, or `ErrorInst::SingletonInstId`.
  SemIR::InstId inst_id;
  // The corresponding type, or `ErrorInst::SingletonTypeId`.
  SemIR::TypeId type_id;
};

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

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CONVERT_H_

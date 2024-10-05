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
    // TODO: Use of an interface for conversions is not yet supported.
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
  SemIR::InstId init_id = SemIR::InstId::Invalid;
  // For an initializer, a block of pending instructions that are needed to
  // form the value of `init_id`, and that can be discarded if no
  // initialization is needed.
  PendingBlock* init_block = nullptr;

  // Are we converting this value into an initializer for an object?
  auto is_initializer() const -> bool {
    return kind == Initializer || kind == FullInitializer;
  }
};

// Convert a value to another type and expression category.
auto Convert(Context& context, SemIR::LocId loc_id, SemIR::InstId expr_id,
             ConversionTarget target) -> SemIR::InstId;

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

// Converts `value_id` to a value expression of type `bool`.
auto ConvertToBoolValue(Context& context, SemIR::LocId loc_id,
                        SemIR::InstId value_id) -> SemIR::InstId;

// Converts `value_id` to type `type_id` for an `as` expression.
auto ConvertForExplicitAs(Context& context, Parse::NodeId as_node,
                          SemIR::InstId value_id, SemIR::TypeId type_id)
    -> SemIR::InstId;

// Information about the parameters of a callee. This information is extracted
// from the EntityWithParamsBase before calling ConvertCallArgs, because
// conversion can trigger importing of more entities, which can invalidate the
// reference to the callee.
struct CalleeParamsInfo {
  explicit CalleeParamsInfo(const SemIR::EntityWithParamsBase& callee)
      : callee_loc(callee.latest_decl_id()),
        implicit_param_refs_id(callee.implicit_param_refs_id),
        param_refs_id(callee.param_refs_id) {}

  // The location of the callee to use in diagnostics.
  SemIRLoc callee_loc;
  // The implicit parameters of the callee.
  SemIR::InstBlockId implicit_param_refs_id;
  // The explicit parameters of the callee.
  SemIR::InstBlockId param_refs_id;
};

// Implicitly converts a set of arguments to match the parameter types in a
// function call. Returns a block containing the converted implicit and explicit
// argument values for runtime parameters.
auto ConvertCallArgs(Context& context, SemIR::LocId call_loc_id,
                     SemIR::InstId self_id,
                     llvm::ArrayRef<SemIR::InstId> arg_refs,
                     SemIR::InstId return_storage_id,
                     const CalleeParamsInfo& callee,
                     SemIR::SpecificId callee_specific_id)
    -> SemIR::InstBlockId;

// A type that has been converted for use as a type expression.
struct TypeExpr {
  // The converted expression of type `type`, or `InstId::BuiltinError`.
  SemIR::InstId inst_id;
  // The corresponding type, or `TypeId::Error`.
  SemIR::TypeId type_id;
};

// Converts an expression for use as a type.
// TODO: Most of the callers of this function discard the `inst_id` and lose
// track of the conversion. In most cases we should be retaining that as the
// operand of some downstream instruction.
auto ExprAsType(Context& context, SemIR::LocId loc_id, SemIR::InstId value_id)
    -> TypeExpr;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CONVERT_H_

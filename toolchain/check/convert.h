// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CONVERT_H_
#define CARBON_TOOLCHAIN_CHECK_CONVERT_H_

#include "toolchain/check/context.h"
#include "toolchain/check/pending_block.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/node.h"

namespace Carbon::Check {

// Description of the target of a conversion.
struct ConversionTarget {
  enum Kind {
    // Convert to a value of type `type`.
    Value,
    // Convert to either a value or a reference of type `type`.
    ValueOrReference,
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
  SemIR::NodeId init_id = SemIR::NodeId::Invalid;
  // For an initializer, a block of pending instructions that are needed to
  // form the value of `target_id`, and that can be discarded if no
  // initialization is needed.
  PendingBlock* init_block = nullptr;

  // Are we converting this value into an initializer for an object?
  bool is_initializer() const {
    return kind == Initializer || kind == FullInitializer;
  }
};

// Convert a value to another type and expression category.
auto Convert(Context& context, Parse::Node parse_node, SemIR::NodeId value_id,
             ConversionTarget target) -> SemIR::NodeId;

// Performs initialization of `target_id` from `value_id`. Returns the
// possibly-converted initializing expression, which should be assigned to the
// target using a suitable node for the kind of initialization.
auto Initialize(Context& context, Parse::Node parse_node,
                SemIR::NodeId target_id, SemIR::NodeId value_id)
    -> SemIR::NodeId;

// Convert the given expression to a value expression of the same type.
auto ConvertToValueExpression(Context& context, SemIR::NodeId expr_id)
    -> SemIR::NodeId;

// Convert the given expression to a value or reference expression of the same
// type.
auto ConvertToValueOrReferenceExpression(Context& context,
                                         SemIR::NodeId expr_id)
    -> SemIR::NodeId;

// Converts `value_id` to a value expression of type `type_id`.
auto ConvertToValueOfType(Context& context, Parse::Node parse_node,
                          SemIR::NodeId value_id, SemIR::TypeId type_id)
    -> SemIR::NodeId;

// Converts `value_id` to a value expression of type `bool`.
auto ConvertToBoolValue(Context& context, Parse::Node parse_node,
                        SemIR::NodeId value_id) -> SemIR::NodeId;

// Implicitly converts a set of arguments to match the parameter types in a
// function call.
auto ConvertCallArgs(Context& context, Parse::Node call_parse_node,
                     SemIR::NodeBlockId arg_refs_id,
                     Parse::Node param_parse_node,
                     SemIR::NodeBlockId param_refs_id, bool has_return_slot)
    -> bool;

// Converts an expression for use as a type.
auto ExpressionAsType(Context& context, Parse::Node parse_node,
                      SemIR::NodeId value_id) -> SemIR::TypeId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CONVERT_H_

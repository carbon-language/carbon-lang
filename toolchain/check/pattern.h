// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_PATTERN_H_
#define CARBON_TOOLCHAIN_CHECK_PATTERN_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Marks the start of a region of insts in a pattern context that might contain
// an expression. Typically this is called when handling a parse node that can
// immediately precede a subpattern (such as `let` or a `,` in a pattern list).
// `End[Empty]Subpattern` should be called later by the consumer of the
// subpattern.
auto BeginSubpattern(Context& context) -> void;

// Consumes the expression in a region started by the most recent
// BeginSubpattern, and returns the ID of the region. The region will not yet
// have any control-flow edges into or out of it.
auto ConsumeSubpatternExpr(Context& context, SemIR::InstId result_id)
    -> SemIR::ExprRegionId;

// Ends a region started by BeginSubpattern (in stack order), asserting that
// it either had no expression content or the expression has been consumed.
auto EndEmptySubpattern(Context& context) -> void;

// Ends a region started by BeginSubpattern (in stack order). If the top of the
// node stack is an expression, the subpattern region is consumed and converted
// to an expression pattern, which replaces the expression on the node stack.
// Otherwise, the top of the node stack should be a pattern, in which case this
// asserts that the subpattern region is either empty or has been consumed.
//
// The node stack is passed explicitly as a reminder that this function affects
// the node stack, unlike the other *Subpattern functions.
auto EndSubpattern(Context& context, NodeStack& node_stack) -> void;

// Information about a created binding pattern.
struct BindingPatternInfo {
  SemIR::InstId pattern_id;
  SemIR::InstId bind_id;
};

// The phase of a binding pattern.
enum class BindingPhase { Template, Symbolic, Runtime };

// Creates an entity name for a binding pattern with the given properties.
auto AddBindingEntityName(Context& context, SemIR::NameId name_id,
                          SemIR::InstId form_id, bool is_unused,
                          BindingPhase phase) -> SemIR::EntityNameId;

// Creates a binding pattern and the associated binding inst, and returns their
// IDs. `scrutinee_type_id` is the type of the binding, and `type_region_id` is
// the region representing that type expression. The binding is added to
// `context.bind_name_map()`, with a placeholder value.
auto AddBindingPattern(Context& context, SemIR::LocId name_loc,
                       SemIR::ExprRegionId type_region_id,
                       SemIR::TypeId scrutinee_type_id,
                       SemIR::AnyBindingPattern pattern) -> BindingPatternInfo;

// Creates a binding inst with the given type and value, to represent the result
// of matching the given binding pattern. The binding is not added to any block,
// or to `context.bind_name_map()`.
auto AddBindingForPattern(Context& context, SemIR::LocId name_loc,
                          SemIR::AnyBindingPattern pattern,
                          SemIR::TypeId binding_type_id, SemIR::InstId value_id)
    -> SemIR::InstId;

// Creates storage for `var` patterns nested within the given pattern at the
// current location in the output SemIR. For a `returned var`, this
// reuses the function's return slot when present.
auto AddPatternVarStorage(Context& context, SemIR::InstBlockId pattern_block_id,
                          bool is_returned_var) -> void;

// Kinds of parameters that can be added by `AddParamPattern`.
enum class ParamPatternKind {
  // A value parameter, `x: T`.
  Value,
  // A reference parameter, `ref x: T`.
  Ref,
  // A variable parameter, `var x: T`.
  Var,
};

// Returns the `ParamPatternKind` of the parameter instruction `param_inst_id`.
auto GetParamPatternKind(Context& context, SemIR::InstId param_inst_id)
    -> ParamPatternKind;

// Adds a parameter pattern with the specified name and type information. The
// pattern emulates `x: T`, `ref x: T`, or `var x: T` depending on the value of
// `kind`. This only sets up the parameter pattern, binding pattern and type;
// callers are expected to add the returned parameter pattern instruction to
// appropriate blocks. This is used when generating functions, rather than
// processing a user-authored declaration.
auto AddParamPattern(Context& context, SemIR::LocId loc_id,
                     SemIR::NameId name_id,
                     SemIR::ExprRegionId type_expr_region_id,
                     SemIR::TypeId type_id, ParamPatternKind kind)
    -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_PATTERN_H_

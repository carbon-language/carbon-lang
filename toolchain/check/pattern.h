// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_PATTERN_H_
#define CARBON_TOOLCHAIN_CHECK_PATTERN_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Marks the start of a region of insts in a pattern context that might
// represent an expression or a pattern. Typically this is called when
// handling a parse node that can immediately precede a subpattern (such
// as `let` or a `,` in a pattern list), and the handler for the subpattern
// node makes the matching `EndSubpatternAs*` call.
auto BeginSubpattern(Context& context) -> void;

// Ends a region started by BeginSubpattern (in stack order), treating it as
// an expression with the given result, and returns the ID of the region. The
// region will not yet have any control-flow edges into or out of it.
auto EndSubpatternAsExpr(Context& context, SemIR::InstId result_id)
    -> SemIR::ExprRegionId;

// Ends a region started by BeginSubpattern (in stack order), asserting that
// it had no expression content.
auto EndSubpatternAsNonExpr(Context& context) -> void;

// Information about a created binding pattern.
struct BindingPatternInfo {
  SemIR::InstId pattern_id;
  SemIR::InstId bind_id;
};

// TODO: Add EndSubpatternAsPattern, when needed.

// Creates a binding pattern. Returns the binding pattern and the bind name
// instruction.
// TODO: remove is_template once we have a separate InstKind for template
// bindings.
auto AddBindingPattern(Context& context, SemIR::LocId name_loc,
                       SemIR::NameId name_id, SemIR::TypeId type_id,
                       SemIR::ExprRegionId type_region_id,
                       SemIR::InstKind pattern_kind, bool is_template)
    -> BindingPatternInfo;

// Creates storage for `var` patterns nested within the given pattern at the
// current location in the output SemIR. For a `returned var`, this
// reuses the function's return slot when present.
auto AddPatternVarStorage(Context& context, SemIR::InstBlockId pattern_block_id,
                          bool is_returned_var) -> void;

// Adds a `self` parameter pattern with the specified type information. This
// only sets up the binding pattern and type; callers are expected to add the
// returned instruction to appropriate blocks. This is used when generating
// functions, rather than processing a user-authored `self: Self`.
auto AddSelfParamPattern(Context& context, SemIR::LocId loc_id,
                         SemIR::ExprRegionId type_expr_region_id,
                         SemIR::TypeId type_id) -> SemIR::InstId;

// As the above, but for `addr self: Self*`.
auto AddAddrSelfParamPattern(Context& context, SemIR::LocId loc_id,
                             SemIR::ExprRegionId type_expr_region_id,
                             SemIR::TypeInstId type_inst_id) -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_PATTERN_H_

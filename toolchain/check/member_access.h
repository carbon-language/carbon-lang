// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_MEMBER_ACCESS_H_
#define CARBON_TOOLCHAIN_CHECK_MEMBER_ACCESS_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Creates SemIR to perform a member access with base expression `base_id` and
// member name `name_id`. Returns the result of the access.
auto PerformMemberAccess(Context& context, SemIR::LocId loc_id,
                         SemIR::InstId base_id, SemIR::NameId name_id)
    -> SemIR::InstId;

// Creates SemIR to perform a compound member access with base expression
// `base_id` and member name expression `member_expr_id`. Returns the result of
// the access. If specified, `missing_impl_diagnoser()` is used to build an
// error diagnostic when impl binding fails due to a missing `impl`.
auto PerformCompoundMemberAccess(
    Context& context, SemIR::LocId loc_id, SemIR::InstId base_id,
    SemIR::InstId member_expr_id,
    MakeDiagnosticBuilderFn missing_impl_diagnoser = nullptr) -> SemIR::InstId;

// Finds the value of an associated entity (given by assoc_entity_inst_id, a
// member of the interface given by interface_type_id) associated with a type or
// facet (given by base_id). Never does instance binding.
auto GetAssociatedValue(Context& context, SemIR::LocId loc_id,
                        SemIR::InstId base_id,
                        SemIR::InstId assoc_entity_inst_id,
                        SemIR::TypeId interface_type_id) -> SemIR::InstId;

// Creates SemIR to perform a tuple index with base expression `tuple_inst_id`
// and index expression `index_inst_id`. Returns the result of the access.
auto PerformTupleAccess(Context& context, SemIR::LocId loc_id,
                        SemIR::InstId tuple_inst_id,
                        SemIR::InstId index_inst_id) -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_MEMBER_ACCESS_H_

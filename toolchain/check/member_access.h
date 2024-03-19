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
auto PerformMemberAccess(Context& context, Parse::NodeId node_id,
                         SemIR::InstId base_id, SemIR::NameId name_id)
    -> SemIR::InstId;

// Creates SemIR to perform a compound member access with base expression
// `base_id` and member name expression `member_expr_id`. Returns the result of
// the access.
auto PerformCompoundMemberAccess(Context& context, Parse::NodeId node_id,
                                 SemIR::InstId base_id,
                                 SemIR::InstId member_expr_id) -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_MEMBER_ACCESS_H_

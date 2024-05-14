// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_POINTER_DEREFERENCE_H_
#define CARBON_TOOLCHAIN_CHECK_POINTER_DEREFERENCE_H_

#include "llvm/ADT/STLFunctionalExtras.h"
#include "toolchain/check/context.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Creates SemIR to perform a pointer dereference with base expression
// `base_id`. Returns the result of the access.
auto PerformPointerDereference(
    Context& context, Parse::AnyPointerDeferenceExprId node_id,
    SemIR::InstId base_i,
    llvm::function_ref<auto(SemIR::TypeId not_pointer_type_id)->void>
        diagnose_not_pointer) -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_POINTER_DEREFERENCE_H_

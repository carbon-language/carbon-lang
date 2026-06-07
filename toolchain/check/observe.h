// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_OBSERVE_H_
#define CARBON_TOOLCHAIN_CHECK_OBSERVE_H_

#include <tuple>

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {
// Gets all `ObserveId`s visible in the current lexical scopes and globally
// across all interfaces.
//
// Note: We must iterate over all interfaces because an `observe` declaration
// within an interface can be defined for types completely unrelated to that
// interface (e.g., `interface I1 { observe I2.A == I2.B; }`).
auto GetObserveIds(Context& context) -> llvm::SmallVector<SemIR::ObserveId>;

// Extracts the canonical values for all equivalent types and the `impls`
// constraint of an `observe` declaration.
auto UnpackObserveDecl(Context& context, SemIR::ObserveId observe_id)
    -> std::pair<llvm::SmallVector<SemIR::InstId>, SemIR::InstId>;

// Checks if the `observe` operands establish an equivalence between the
// two given types.
auto CheckObserveOperandsForEquivalance(
    Context& context, llvm::ArrayRef<SemIR::InstId> observe_operand_ids,
    SemIR::TypeId lhs_type_id, SemIR::TypeId rhs_type_id) -> bool;
}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_OBSERVE_H_

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_SPECIFIC_NAMED_CONSTRAINT_H_
#define CARBON_TOOLCHAIN_SEM_IR_SPECIFIC_NAMED_CONSTRAINT_H_

#include "toolchain/base/canonical_value_store.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// A pair of a named constraint and a specific for that named constraint.
struct SpecificNamedConstraint {
  static const SpecificNamedConstraint None;

  NamedConstraintId named_constraint_id;
  SpecificId specific_id;

  friend auto operator==(const SpecificNamedConstraint& lhs,
                         const SpecificNamedConstraint& rhs) -> bool = default;
};

inline constexpr SpecificNamedConstraint SpecificNamedConstraint::None = {
    .named_constraint_id = NamedConstraintId::None,
    .specific_id = SpecificId::None};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_SPECIFIC_NAMED_CONSTRAINT_H_

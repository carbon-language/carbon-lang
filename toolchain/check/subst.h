// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_SUBST_H_
#define CARBON_TOOLCHAIN_CHECK_SUBST_H_

#include "common/map.h"
#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// A substitution that is being performed.
struct Substitution {
  // The index of a `BindSymbolicName` instruction that is being replaced.
  SemIR::CompileTimeBindIndex bind_id;
  // The replacement constant value to substitute.
  SemIR::ConstantId replacement_id;
};

using Substitutions = llvm::ArrayRef<Substitution>;

// Replaces the `BindSymbolicName` instruction `bind_id` with `replacement_id`
// throughout the constant `const_id`, and returns the substituted value.
auto SubstConstant(Context& context, SemIR::ConstantId const_id,
                   Substitutions substitutions) -> SemIR::ConstantId;

// Replaces the `BindSymbolicName` instruction `bind_id` with `replacement_id`
// throughout the type `type_id`, and returns the substituted value.
auto SubstType(Context& context, SemIR::TypeId type_id,
               Substitutions substitutions) -> SemIR::TypeId;

// Replaces any symbolic constants used within `type_id` with versions of those
// constants that are specific to `generic_id`. Any instructions created in this
// process are added to the current instruction block. Uses
// `constants_in_generic` as a cache mapping constant instructions to their
// replacement constant ID.
auto SubstAndRebuildTypeForGenericEvalBlock(
    Context& context, SemIR::GenericId generic_id,
    SemIR::GenericInstIndex::Region region,
    Map<SemIR::InstId, SemIR::ConstantId>& constants_in_generic,
    SemIR::TypeId type_id) -> SemIR::TypeId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_SUBST_H_

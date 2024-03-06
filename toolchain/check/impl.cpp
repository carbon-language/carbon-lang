// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/impl.h"

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/impl.h"

namespace Carbon::Check {

auto BuildImplWitness(Context& context, SemIR::ImplId impl_id)
	-> SemIR::InstId {
  auto& impl = context.impls().Get(impl_id);
  CARBON_CHECK(impl.is_being_defined());

  return SemIR::InstId::BuiltinError;
}

}  // namespace Carbon::Check

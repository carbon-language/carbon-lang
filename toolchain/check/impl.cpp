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

  // TODO: Handle non-interface constraints.
  auto interface_type =
      context.types().TryGetAs<SemIR::InterfaceType>(impl.constraint_id);
  if (!interface_type) {
    context.TODO(context.insts().GetParseNode(impl.definition_id),
                 "impl as non-interface");
    return SemIR::InstId::BuiltinError;
  }

  auto interface_id = interface_type->interface_id;

  // TODO: Form the witness table.

  auto table_id = context.inst_blocks().Add({});
  return context.AddInst(SemIR::InterfaceWitness{
      context.GetBuiltinType(SemIR::BuiltinKind::WitnessType), interface_id,
      table_id});
}

}  // namespace Carbon::Check

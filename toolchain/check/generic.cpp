// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/generic.h"

#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

auto StartGenericDecl(Context& /*context*/) -> void {
  // TODO: Start tracking the contents of this declaration.
}

auto StartGenericDefinition(Context& /*context*/,
                            SemIR::GenericId /*generic_id*/) -> void {
  // TODO: Start tracking the contents of this definition.
}

auto FinishGenericDecl(Context& context, SemIR::InstId decl_id)
    -> SemIR::GenericId {
  if (context.scope_stack().compile_time_binding_stack().empty()) {
    return SemIR::GenericId::Invalid;
  }

  auto bindings_id = context.inst_blocks().Add(
      context.scope_stack().compile_time_binding_stack());
  return context.generics().Add(
      SemIR::Generic{.decl_id = decl_id, .bindings_id = bindings_id});
}

auto FinishGenericRedecl(Context& /*context*/, SemIR::InstId /*decl_id*/,
                         SemIR::GenericId /*generic_id*/) -> void {
  // TODO: Compare contents of this declaration with the existing one on the
  // generic.
}

auto FinishGenericDefinition(Context& /*context*/,
                             SemIR::GenericId /*generic_id*/) -> void {
  // TODO: Track contents of this generic definition.
}

auto MakeGenericInstance(Context& context, SemIR::GenericId generic_id,
                         SemIR::InstBlockId args_id)
    -> SemIR::GenericInstanceId {
  auto instance_id = context.generic_instances().GetOrAdd(generic_id, args_id);
  // TODO: Perform substitution into the generic declaration if needed.
  return instance_id;
}

}  // namespace Carbon::Check

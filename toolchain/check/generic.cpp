// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/generic.h"

#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

auto StartGenericDecl(Context& context) -> void {
  context.generic_region_stack().Push();
}

auto StartGenericDefinition(Context& context) -> void {
  // Push a generic region even if we don't have a generic_id. We might still
  // have locally-introduced generic parameters to track:
  //
  // fn F() {
  //   let T:! type = i32;
  //   var x: T;
  // }
  context.generic_region_stack().Push();
}

auto FinishGenericDecl(Context& context, SemIR::InstId decl_id)
    -> SemIR::GenericId {
  auto all_bindings =
      context.scope_stack().compile_time_bindings_stack().PeekAllValues();

  if (all_bindings.empty()) {
    CARBON_CHECK(context.generic_region_stack().PeekDependentInsts().empty())
        << "Have dependent instructions but no compile time bindings are in "
           "scope.";
    context.generic_region_stack().Pop();
    return SemIR::GenericId::Invalid;
  }

  auto bindings_id = context.inst_blocks().Add(all_bindings);
  // TODO: Track the list of dependent instructions in this region.
  context.generic_region_stack().Pop();
  return context.generics().Add(
      SemIR::Generic{.decl_id = decl_id, .bindings_id = bindings_id});
}

auto FinishGenericRedecl(Context& context, SemIR::InstId /*decl_id*/,
                         SemIR::GenericId /*generic_id*/) -> void {
  // TODO: Compare contents of this declaration with the existing one on the
  // generic.
  context.generic_region_stack().Pop();
}

auto FinishGenericDefinition(Context& context, SemIR::GenericId generic_id)
    -> void {
  if (!generic_id.is_valid()) {
    // TODO: We can have symbolic constants in a context that had a non-generic
    // declaration, for example if there's a local generic let binding in a
    // function definition. Handle this case somehow -- perhaps by forming
    // substituted constant values now.
    context.generic_region_stack().Pop();
    return;
  }

  // TODO: Track the list of dependent instructions in this region.
  context.generic_region_stack().Pop();
}

auto MakeGenericInstance(Context& context, SemIR::GenericId generic_id,
                         SemIR::InstBlockId args_id)
    -> SemIR::GenericInstanceId {
  auto instance_id = context.generic_instances().GetOrAdd(generic_id, args_id);
  // TODO: Perform substitution into the generic declaration if needed.
  return instance_id;
}

auto MakeGenericSelfInstance(Context& context, SemIR::GenericId generic_id)
    -> SemIR::GenericInstanceId {
  // TODO: Remove this once we import generics properly.
  if (!generic_id.is_valid()) {
    return SemIR::GenericInstanceId::Invalid;
  }

  auto& generic = context.generics().Get(generic_id);
  auto args = context.inst_blocks().Get(generic.bindings_id);

  // Form a canonical argument list for the generic.
  llvm::SmallVector<SemIR::InstId> arg_ids;
  arg_ids.reserve(args.size());
  for (auto arg_id : args) {
    arg_ids.push_back(context.constant_values().GetConstantInstId(arg_id));
  }
  auto args_id = context.inst_blocks().AddCanonical(arg_ids);

  // Build a corresponding instance.
  // TODO: This could be made more efficient. We don't need to perform
  // substitution here; we know we want identity mappings for all constants and
  // types. We could also consider not storing the mapping at all in this case.
  return MakeGenericInstance(context, generic_id, args_id);
}

}  // namespace Carbon::Check

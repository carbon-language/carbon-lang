// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/generic.h"

#include "common/map.h"
#include "toolchain/check/generic_region_stack.h"
#include "toolchain/check/subst.h"
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

// Builds and returns a block of instructions whose constant values need to be
// evaluated in order to resolve a generic instance.
static auto MakeGenericEvalBlock(Context& context, SemIR::GenericId generic_id,
                                 SemIR::GenericInstIndex::Region region)
    -> SemIR::InstBlockId {
  context.inst_block_stack().Push();

  Map<SemIR::InstId, SemIR::ConstantId> constants;
  // TODO: For the definition region, populate constants from the declaration.

  // TODO: See if we can ensure that the generic region stack is unchanged
  // throughout this work. If so, we can use a range-based loop here instead.
  for (size_t i = 0;
       i != context.generic_region_stack().PeekDependentInsts().size(); ++i) {
    auto [inst_id, dep_kind] =
        context.generic_region_stack().PeekDependentInsts()[i];

    // If the type is symbolic, replace it with a type specific to this generic.
    if ((dep_kind & GenericRegionStack::DependencyKind::SymbolicType) !=
        GenericRegionStack::DependencyKind::None) {
      auto inst = context.insts().Get(inst_id);
      inst.SetType(SubstAndRebuildTypeForGenericEvalBlock(
          context, generic_id, region, constants, inst.type_id()));
      context.sem_ir().insts().Set(inst_id, inst);
    }

    // If the instruction has a symbolic constant value, then make a note that
    // we'll need to evaluate this instruction in the generic instance. Update
    // the constant value of the instruction to refer to the result of that
    // eventual evaluation.
    if ((dep_kind & GenericRegionStack::DependencyKind::SymbolicConstant) !=
        GenericRegionStack::DependencyKind::None) {
      auto const_inst_id = context.constant_values().GetConstantInstId(inst_id);

      // Create a new symbolic constant representing this instruction in this
      // generic, if it doesn't already exist.
      auto result = constants.Insert(const_inst_id, [&] {
        auto index = SemIR::GenericInstIndex(
            region,
            context.inst_block_stack().PeekCurrentBlockContents().size());
        context.inst_block_stack().AddInstId(inst_id);
        return context.constant_values().AddSymbolicConstant(
            {.inst_id = const_inst_id,
             .generic_id = generic_id,
             .index = index});
      });

      context.constant_values().Set(inst_id, result.value());
    }
  }

  return context.inst_block_stack().Pop();
}

auto FinishGenericDecl(Context& context, SemIR::InstId decl_id)
    -> SemIR::GenericId {
  if (context.scope_stack().compile_time_binding_stack().empty()) {
    CARBON_CHECK(context.generic_region_stack().PeekDependentInsts().empty())
        << "Have dependent instructions but no compile time bindings are in "
           "scope.";
    context.generic_region_stack().Pop();
    return SemIR::GenericId::Invalid;
  }

  auto bindings_id = context.inst_blocks().Add(
      context.scope_stack().compile_time_binding_stack());
  auto generic_id = context.generics().Add(
      SemIR::Generic{.decl_id = decl_id, .bindings_id = bindings_id});

  auto decl_block_id = MakeGenericEvalBlock(
      context, generic_id, SemIR::GenericInstIndex::Region::Declaration);
  context.generic_region_stack().Pop();

  context.generics().Get(generic_id).decl_block_id = decl_block_id;
  return generic_id;
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

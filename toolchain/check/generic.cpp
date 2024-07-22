// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/generic.h"

#include "common/map.h"
#include "toolchain/check/eval.h"
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

// Adds an instruction `generic_inst_id` to the eval block for a generic region,
// which is the current instruction block. The instruction `generic_inst_id` is
// expected to compute the value of the constant described by `const_inst_id` in
// each instance of the generic. Forms and returns a corresponding symbolic
// constant ID that refers to the substituted value of that instruction in each
// instance of the generic.
static auto AddGenericConstantInstToEvalBlock(
    Context& context, SemIR::GenericId generic_id,
    SemIR::GenericInstIndex::Region region, SemIR::InstId const_inst_id,
    SemIR::InstId generic_inst_id) -> SemIR::ConstantId {
  auto index = SemIR::GenericInstIndex(
      region, context.inst_block_stack().PeekCurrentBlockContents().size());
  context.inst_block_stack().AddInstId(generic_inst_id);
  return context.constant_values().AddSymbolicConstant(
      {.inst_id = const_inst_id, .generic_id = generic_id, .index = index});
}

namespace {
// Substitution callbacks to rebuild a generic constant in the eval block for a
// generic region.
class RebuildGenericConstantInEvalBlockCallbacks final
    : public SubstInstCallbacks {
 public:
  RebuildGenericConstantInEvalBlockCallbacks(
      Context& context, SemIR::GenericId generic_id,
      SemIR::GenericInstIndex::Region region,
      Map<SemIR::InstId, SemIR::InstId>& constants_in_generic)
      : context_(context),
        generic_id_(generic_id),
        region_(region),
        constants_in_generic_(constants_in_generic) {}

  // Check for instructions for which we already have a mapping into the eval
  // block, and substitute them for the instructions in the eval block.
  auto Subst(SemIR::InstId& inst_id) const -> bool override {
    auto const_id = context_.constant_values().Get(inst_id);
    if (!const_id.is_symbolic()) {
      // This instruction doesn't have a symbolic constant value, so can't
      // contain any bindings that need to be substituted.
      return true;
    }

    // If this instruction is in the map, return the known result.
    if (auto result = constants_in_generic_.Lookup(
            context_.constant_values().GetInstId(const_id))) {
      // In order to reuse instructions from the generic as often as possible,
      // keep this instruction as-is if it already has the desired symbolic
      // constant value.
      if (const_id != context_.constant_values().Get(result.value())) {
        inst_id = result.value();
      }
      CARBON_CHECK(inst_id.is_valid());
      return true;
    }

    // If the instruction is a symbolic binding, build a version in the eval
    // block.
    if (auto binding =
            context_.insts().TryGetAs<SemIR::BindSymbolicName>(inst_id)) {
      inst_id = Rebuild(inst_id, *binding);
      return true;
    }

    return false;
  }

  // Build a new instruction in the eval block corresponding to the given
  // constant.
  auto Rebuild(SemIR::InstId orig_inst_id, SemIR::Inst new_inst) const
      -> SemIR::InstId override {
    auto const_inst_id =
        context_.constant_values().GetConstantInstId(orig_inst_id);
    // We might already have an instruction in the eval block if a transitive
    // operand of this instruction has the same constant value.
    auto result = constants_in_generic_.Insert(const_inst_id, [&] {
      // TODO: Add a function on `Context` to add the instruction without
      // inserting it into the dependent instructions list or computing a
      // constant value for it.
      auto inst_id = context_.sem_ir().insts().AddInNoBlock(
          SemIR::LocIdAndInst::NoLoc(new_inst));
      auto const_id = AddGenericConstantInstToEvalBlock(
          context_, generic_id_, region_, const_inst_id, inst_id);
      context_.constant_values().Set(inst_id, const_id);
      return inst_id;
    });
    return result.value();
  }

 private:
  Context& context_;
  SemIR::GenericId generic_id_;
  SemIR::GenericInstIndex::Region region_;
  Map<SemIR::InstId, SemIR::InstId>& constants_in_generic_;
};
}  // namespace

// Adds instructions to compute the substituted version of `type_id` in each
// instance of a generic into the eval block for the generic, which is the
// current instruction block. Returns a symbolic type ID that refers to the
// substituted type in each instance of the generic.
static auto AddGenericTypeToEvalBlock(
    Context& context, SemIR::GenericId generic_id,
    SemIR::GenericInstIndex::Region region,
    Map<SemIR::InstId, SemIR::InstId>& constants_in_generic,
    SemIR::TypeId type_id) -> SemIR::TypeId {
  // Substitute into the type's constant instruction and rebuild it in the eval
  // block.
  auto type_inst_id =
      SubstInst(context, context.types().GetInstId(type_id),
                RebuildGenericConstantInEvalBlockCallbacks(
                    context, generic_id, region, constants_in_generic));
  return context.GetTypeIdForTypeInst(type_inst_id);
}

// Adds instructions to compute the substituted value of `inst_id` in each
// instance of a generic into the eval block for the generic, which is the
// current instruction block. Returns a symbolic constant instruction ID that
// refers to the substituted constant value in each instance of the generic.
static auto AddGenericConstantToEvalBlock(
    Context& context, SemIR::GenericId generic_id,
    SemIR::GenericInstIndex::Region region,
    Map<SemIR::InstId, SemIR::InstId>& constants_in_generic,
    SemIR::InstId inst_id) -> SemIR::ConstantId {
  // Substitute into the constant value and rebuild it in the eval block if
  // we've not encountered it before.
  auto const_inst_id = context.constant_values().GetConstantInstId(inst_id);
  auto new_inst_id =
      SubstInst(context, const_inst_id,
                RebuildGenericConstantInEvalBlockCallbacks(
                    context, generic_id, region, constants_in_generic));
  CARBON_CHECK(new_inst_id != const_inst_id)
      << "Did not apply any substitutions to symbolic constant "
      << context.insts().Get(const_inst_id);
  return context.constant_values().Get(new_inst_id);
}

// Builds and returns a block of instructions whose constant values need to be
// evaluated in order to resolve a generic instance.
static auto MakeGenericEvalBlock(Context& context, SemIR::GenericId generic_id,
                                 SemIR::GenericInstIndex::Region region)
    -> SemIR::InstBlockId {
  context.inst_block_stack().Push();

  Map<SemIR::InstId, SemIR::InstId> constants_in_generic;

  // For the definition region, populate constants from the declaration.
  if (region == SemIR::GenericInstIndex::Region::Definition) {
    auto decl_eval_block = context.inst_blocks().Get(
        context.generics().Get(generic_id).decl_block_id);
    for (auto inst_id : decl_eval_block) {
      constants_in_generic.Insert(
          context.constant_values().GetConstantInstId(inst_id), inst_id);
    }
  }

  // The work done in this loop might invalidate iterators into the generic
  // region stack, but shouldn't add new dependent instructions to the current
  // region.
  auto num_dependent_insts =
      context.generic_region_stack().PeekDependentInsts().size();
  for (auto i : llvm::seq(num_dependent_insts)) {
    auto [inst_id, dep_kind] =
        context.generic_region_stack().PeekDependentInsts()[i];

    // If the type is symbolic, replace it with a type specific to this generic.
    if ((dep_kind & GenericRegionStack::DependencyKind::SymbolicType) !=
        GenericRegionStack::DependencyKind::None) {
      auto inst = context.insts().Get(inst_id);
      auto type_id = AddGenericTypeToEvalBlock(
          context, generic_id, region, constants_in_generic, inst.type_id());
      // TODO: Eventually, completeness requirements should be modeled as
      // constraints on the generic rather than properties of the type. For now,
      // require the transformed type to be complete if the original was.
      // TODO: We'll also need to do this when evaluating the eval block.
      if (context.types().IsComplete(inst.type_id())) {
        context.TryToCompleteType(type_id);
      }
      inst.SetType(type_id);
      context.sem_ir().insts().Set(inst_id, inst);
    }

    // If the instruction has a symbolic constant value, then make a note that
    // we'll need to evaluate this instruction in the generic instance. Update
    // the constant value of the instruction to refer to the result of that
    // eventual evaluation.
    if ((dep_kind & GenericRegionStack::DependencyKind::SymbolicConstant) !=
        GenericRegionStack::DependencyKind::None) {
      // Update the constant value to refer to this generic.
      context.constant_values().Set(
          inst_id,
          AddGenericConstantToEvalBlock(context, generic_id, region,
                                        constants_in_generic, inst_id));
    }
  }

  CARBON_CHECK(num_dependent_insts ==
               context.generic_region_stack().PeekDependentInsts().size())
      << "Building eval block added new dependent insts, for example "
      << context.insts().Get(context.generic_region_stack()
                                 .PeekDependentInsts()[num_dependent_insts]
                                 .inst_id);

  return context.inst_block_stack().Pop();
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

  // Build the new Generic object. Note that we intentionally do not hold a
  // persistent reference to it throughout this function, because the `generics`
  // collection can have items added to it by import resolution while we are
  // building this generic.
  auto bindings_id = context.inst_blocks().Add(all_bindings);
  auto generic_id = context.generics().Add(
      SemIR::Generic{.decl_id = decl_id,
                     .bindings_id = bindings_id,
                     .self_instance_id = SemIR::GenericInstanceId::Invalid});

  auto decl_block_id = MakeGenericEvalBlock(
      context, generic_id, SemIR::GenericInstIndex::Region::Declaration);
  context.generic_region_stack().Pop();
  context.generics().Get(generic_id).decl_block_id = decl_block_id;

  auto self_instance_id = MakeGenericSelfInstance(context, generic_id);
  context.generics().Get(generic_id).self_instance_id = self_instance_id;
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

  auto definition_block_id = MakeGenericEvalBlock(
      context, generic_id, SemIR::GenericInstIndex::Region::Definition);
  context.generics().Get(generic_id).definition_block_id = definition_block_id;

  context.generic_region_stack().Pop();
}

auto MakeGenericInstance(Context& context, SemIR::GenericId generic_id,
                         SemIR::InstBlockId args_id)
    -> SemIR::GenericInstanceId {
  auto instance_id = context.generic_instances().GetOrAdd(generic_id, args_id);

  // TODO: Remove this once we import generics properly.
  if (!generic_id.is_valid()) {
    return instance_id;
  }

  // If this is the first time we've formed this instance, evaluate its decl
  // block to form information about the instance.
  if (!context.generic_instances().Get(instance_id).decl_block_id.is_valid()) {
    auto decl_block_id = TryEvalBlockForSpecific(
        context, instance_id, SemIR::GenericInstIndex::Region::Declaration);
    // Note that TryEvalBlockForSpecific may reallocate the list of generic
    // instances, so re-lookup the instance here.
    context.generic_instances().Get(instance_id).decl_block_id = decl_block_id;
  }

  return instance_id;
}

auto MakeGenericSelfInstance(Context& context, SemIR::GenericId generic_id)
    -> SemIR::GenericInstanceId {
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

auto ResolveSpecificDefinition(Context& context,
                               SemIR::GenericInstanceId specific_id) -> bool {
  auto& specific = context.generic_instances().Get(specific_id);
  auto generic_id = specific.generic_id;

  // TODO: Remove this once we import generics properly.
  if (!generic_id.is_valid()) {
    return true;
  }

  if (!specific.definition_block_id.is_valid()) {
    // Evaluate the eval block for the definition of the generic.
    auto& generic = context.generics().Get(generic_id);
    if (!generic.definition_block_id.is_valid()) {
      // The generic is not defined yet.
      return false;
    }
    auto definition_block_id = TryEvalBlockForSpecific(
        context, specific_id, SemIR::GenericInstIndex::Region::Definition);
    // Note that TryEvalBlockForSpecific may reallocate the list of generic
    // instances, so re-lookup the instance here.
    context.generic_instances().Get(specific_id).definition_block_id =
        definition_block_id;
  }
  return true;
}

}  // namespace Carbon::Check

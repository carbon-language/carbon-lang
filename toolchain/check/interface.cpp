// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/interface.h"

#include "toolchain/check/context.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto BuildAssociatedEntity(Context& context, SemIR::InterfaceId interface_id,
                           SemIR::InstId decl_id) -> SemIR::InstId {
  auto& interface_info = context.interfaces().Get(interface_id);
  if (!interface_info.is_being_defined()) {
    // This should only happen if the interface is erroneously defined more than
    // once.
    // TODO: Find a way to CHECK this.
    return SemIR::ErrorInst::SingletonInstId;
  }

  // The interface type is the type of `Self`.
  auto self_type_id =
      context.insts().Get(interface_info.self_param_id).type_id();

  // Register this declaration as declaring an associated entity.
  auto index = SemIR::ElementIndex(
      context.args_type_info_stack().PeekCurrentBlockContents().size());
  context.args_type_info_stack().AddInstId(decl_id);

  // Name lookup for the declaration's name should name the associated entity,
  // not the declaration itself.
  auto type_id = context.GetAssociatedEntityType(self_type_id);
  return context.AddInst<SemIR::AssociatedEntity>(
      context.insts().GetLocId(decl_id),
      {.type_id = type_id, .index = index, .decl_id = decl_id});
}

auto GetSelfSpecificForInterfaceMemberWithSelfType(
    Context& context, SemIRLoc loc, SemIR::SpecificId enclosing_specific_id,
    SemIR::GenericId generic_id, SemIR::TypeId self_type_id,
    SemIR::InstId witness_inst_id) -> SemIR::SpecificId {
  const auto& generic = context.generics().Get(generic_id);
  auto bindings = context.inst_blocks().Get(generic.bindings_id);

  llvm::SmallVector<SemIR::InstId> arg_ids;
  arg_ids.reserve(bindings.size());

  // Start with the enclosing arguments.
  if (enclosing_specific_id.has_value()) {
    auto enclosing_specific_args_id =
        context.specifics().Get(enclosing_specific_id).args_id;
    auto enclosing_specific_args =
        context.inst_blocks().Get(enclosing_specific_args_id);
    arg_ids.assign(enclosing_specific_args.begin(),
                   enclosing_specific_args.end());
  }

  // Add the `Self` argument. First find the `Self` binding.
  auto self_binding =
      context.insts().GetAs<SemIR::BindSymbolicName>(bindings[arg_ids.size()]);
  CARBON_CHECK(
      context.entity_names().Get(self_binding.entity_name_id).name_id ==
          SemIR::NameId::SelfType,
      "Expected a Self binding, found {0}", self_binding);
  // Create a facet value to be the value of `Self` in the interface.
  // TODO: Pass this in instead of creating it here. The caller sometimes
  // already has a facet value.
  auto type_inst_id = context.types().GetInstId(self_type_id);
  auto facet_value_const_id =
      TryEvalInst(context, SemIR::InstId::None,
                  SemIR::FacetValue{.type_id = self_binding.type_id,
                                    .type_inst_id = type_inst_id,
                                    .witness_inst_id = witness_inst_id});
  arg_ids.push_back(context.constant_values().GetInstId(facet_value_const_id));

  // Take any trailing argument values from the self specific.
  // TODO: If these refer to outer arguments, for example in their types, we may
  // need to perform extra substitutions here.
  auto self_specific_args = context.inst_blocks().Get(
      context.specifics().Get(generic.self_specific_id).args_id);
  for (auto arg_id : self_specific_args.drop_front(arg_ids.size())) {
    arg_ids.push_back(context.constant_values().GetConstantInstId(arg_id));
  }

  auto args_id = context.inst_blocks().AddCanonical(arg_ids);
  return MakeSpecific(context, loc, generic_id, args_id);
}

auto GetTypeForSpecificAssociatedEntity(Context& context, SemIRLoc loc,
                                        SemIR::SpecificId interface_specific_id,
                                        SemIR::InstId decl_id,
                                        SemIR::TypeId self_type_id,
                                        SemIR::InstId self_witness_id)
    -> SemIR::TypeId {
  auto decl =
      context.insts().Get(context.constant_values().GetConstantInstId(decl_id));

  auto specific_id = interface_specific_id;
  if (auto assoc_const = decl.TryAs<SemIR::AssociatedConstantDecl>()) {
    specific_id = GetSelfSpecificForInterfaceMemberWithSelfType(
        context, loc, interface_specific_id,
        context.associated_constants()
            .Get(assoc_const->assoc_const_id)
            .generic_id,
        self_type_id, self_witness_id);
  }
  // TODO: For a `FunctionDecl`, should we substitute `Self` into the type?

  return SemIR::GetTypeInSpecific(context.sem_ir(), specific_id,
                                  context.insts().Get(decl_id).type_id());
}

}  // namespace Carbon::Check

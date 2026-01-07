// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/require_impls.h"

#include "toolchain/check/generic.h"
#include "toolchain/check/interface.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

static auto GetEnclosingDeclFromEnclosingSpecificId(
    Context& context, SemIR::SpecificId enclosing_specific_id) -> SemIR::Inst {
  return context.insts().Get(
      context.generics()
          .Get(context.specifics().Get(enclosing_specific_id).generic_id)
          .decl_id);
}

static auto GetSpecificArgsFromEnclosingSpecific(
    Context& context, SemIR::SpecificId enclosing_specific_id)
    -> llvm::SmallVector<SemIR::InstId> {
  auto enclosing_specific_args_id =
      context.specifics().GetArgsOrEmpty(enclosing_specific_id);
  auto enclosing_specific_args =
      context.inst_blocks().Get(enclosing_specific_args_id);
  llvm::SmallVector<SemIR::InstId> arg_ids;
  // Reserve space for the `Self` after the enclosing specific's args.
  arg_ids.reserve(enclosing_specific_args.size() + 1);
  llvm::append_range(arg_ids, enclosing_specific_args);
  return arg_ids;
}

auto GetRequireImplsSpecificFromEnclosingSpecific(
    Context& context, const SemIR::RequireImpls& require,
    SemIR::SpecificId enclosing_specific_id) -> RequireImplsSpecific {
  if (enclosing_specific_id.has_value()) {
    auto enclosing_generic_decl =
        GetEnclosingDeclFromEnclosingSpecificId(context, enclosing_specific_id);
    CARBON_CHECK(enclosing_generic_decl.Is<SemIR::InterfaceDecl>() ||
                     enclosing_generic_decl.Is<SemIR::NamedConstraintDecl>(),
                 "Incorrect enclosing specific for RequireImpls. Expected an "
                 "interface or named constraint. Found {0}.",
                 enclosing_generic_decl);
  }

  auto arg_ids =
      GetSpecificArgsFromEnclosingSpecific(context, enclosing_specific_id);

  // Specifics inside an interface/constraint also include the `Self` of the
  // enclosing entity. We copy that `Self` from the self-specific of the
  // RequireImpls generic.
  const auto& require_generic = context.generics().Get(require.generic_id);
  const auto& require_self_specific =
      context.specifics().Get(require_generic.self_specific_id);
  auto require_self_specific_args =
      context.inst_blocks().Get(require_self_specific.args_id);
  // The last argument of a `require` generic is always `Self`, as `require`
  // can not have any parameters of its own, only enclosing parameters.
  auto self_inst_id = require_self_specific_args.back();
  CARBON_CHECK(context.insts().Is<SemIR::SymbolicBinding>(self_inst_id));
  arg_ids.push_back(self_inst_id);

  auto specific_id = MakeSpecific(context, SemIR::LocId(require.decl_id),
                                  require.generic_id, arg_ids);
  // TODO: Cache the specific on Context.
  return {.specific_id = specific_id};
}

auto GetRequireImplsSpecificFromEnclosingSpecificWithSelfType(
    Context& context, const SemIR::RequireImpls& require,
    SemIR::SpecificId enclosing_specific_id, SemIR::TypeInstId self_id,
    SemIR::InstId witness_id) -> RequireImplsSpecific {
  if (enclosing_specific_id.has_value()) {
    auto enclosing_generic_decl =
        GetEnclosingDeclFromEnclosingSpecificId(context, enclosing_specific_id);
    CARBON_CHECK(enclosing_generic_decl.Is<SemIR::InterfaceDecl>(),
                 "Incorrect enclosing specific for RequireImpls with explicit "
                 "self type. Expected an interface. Found {0}.",
                 enclosing_generic_decl);
  }

  // Construct a facet value around the `self_id` type of the correct facet
  // type for the `Self` in the require's self-specific.
  auto self_facet_value = GetSelfFacetValueForInterfaceMemberSpecific(
      context, enclosing_specific_id, require.generic_id,
      context.types().GetTypeIdForTypeInstId(self_id), witness_id);

  auto arg_ids =
      GetSpecificArgsFromEnclosingSpecific(context, enclosing_specific_id);
  arg_ids.push_back(self_facet_value);

  auto specific_id = MakeSpecific(context, SemIR::LocId(require.decl_id),
                                  require.generic_id, arg_ids);
  // TODO: Cache the specific on Context.
  return {.specific_id = specific_id};
}

auto GetConstantValueInRequireImplsSpecific(Context& context,
                                            RequireImplsSpecific specific,
                                            SemIR::InstId inst_id)
    -> SemIR::ConstantId {
  auto const_id = SemIR::GetConstantValueInSpecific(
      context.sem_ir(), specific.specific_id, inst_id);
  CARBON_CHECK(const_id.has_value(), "The specific has not been resolved?");
  return const_id;
}

}  // namespace Carbon::Check

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/custom_witness.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/impl.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/type.h"

namespace Carbon::Check {

// Given a value whose type `IsFacetTypeOrError`, returns the corresponding
// type.
static auto GetFacetAsType(Context& context, SemIR::LocId loc_id,
                           SemIR::ConstantId facet_or_type_const_id)
    -> SemIR::TypeId {
  auto facet_or_type_id =
      context.constant_values().GetInstId(facet_or_type_const_id);
  auto type_type_id = context.insts().Get(facet_or_type_id).type_id();
  CARBON_CHECK(context.types().IsFacetTypeOrError(type_type_id));

  if (context.types().Is<SemIR::FacetType>(type_type_id)) {
    // It's a facet; access its type.
    facet_or_type_id = GetOrAddInst<SemIR::FacetAccessType>(
        context, loc_id,
        {.type_id = SemIR::TypeType::TypeId,
         .facet_value_inst_id = facet_or_type_id});
  }
  return context.types().GetTypeIdForTypeInstId(facet_or_type_id);
}

auto BuildCustomWitness(Context& context, SemIR::LocId loc_id,
                        SemIR::ConstantId query_self_const_id,
                        SemIR::SpecificInterface specific_interface,
                        llvm::ArrayRef<SemIR::InstId> values) -> SemIR::InstId {
  const auto& interface =
      context.interfaces().Get(specific_interface.interface_id);
  auto assoc_entities =
      context.inst_blocks().GetOrEmpty(interface.associated_entities_id);
  if (assoc_entities.size() != values.size()) {
    context.TODO(loc_id, ("Unsupported definition of interface " +
                          context.names().GetFormatted(interface.name_id))
                             .str());
    return SemIR::ErrorInst::InstId;
  }

  llvm::SmallVector<SemIR::InstId> entries;

  // Build a witness with the current contents of the witness table. This will
  // grow as we progress through the impl. In theory this will build O(n^2)
  // table entries, but in practice n <= 2, so that's OK.
  //
  // This is necessary because later associated entities may refer to earlier
  // associated entities in their signatures. In particular, an associated
  // result type may be used as the return type of an associated function.
  //
  // TODO: Consider building one witness after all associated constants, and
  // then a second after all associated functions, rather than building one at
  // each step. For now this doesn't really matter since we don't have more than
  // one of each anyway.
  auto make_witness = [&] {
    return context.constant_values().GetInstId(
        EvalOrAddInst<SemIR::CustomWitness>(
            context, loc_id,
            {.type_id =
                 GetSingletonType(context, SemIR::WitnessType::TypeInstId),
             .elements_id = context.inst_blocks().Add(entries)}));
  };

  // Fill in the witness table.
  for (const auto& [assoc_entity_id, value_id] :
       llvm::zip_equal(assoc_entities, values)) {
    LoadImportRef(context, assoc_entity_id);
    auto decl_id =
        context.constant_values().GetInstId(SemIR::GetConstantValueInSpecific(
            context.sem_ir(), specific_interface.specific_id, assoc_entity_id));
    CARBON_CHECK(decl_id.has_value(), "Non-constant associated entity");
    auto decl = context.insts().Get(decl_id);
    CARBON_KIND_SWITCH(decl) {
      case CARBON_KIND(SemIR::StructValue struct_value): {
        if (struct_value.type_id == SemIR::ErrorInst::TypeId) {
          return SemIR::ErrorInst::InstId;
        }
        auto self_type_id =
            GetFacetAsType(context, loc_id, query_self_const_id);
        // TODO: If a thunk is needed, this will build a different value each
        // time it's called, so we won't properly deduplicate repeated
        // witnesses.
        // TODO: Skip calling make_witness if this function signature doesn't
        // involve `Self`.
        entries.push_back(CheckAssociatedFunctionImplementation(
            context,
            context.types().GetAs<SemIR::FunctionType>(struct_value.type_id),
            value_id, self_type_id, make_witness(),
            /*defer_thunk_definition=*/false));
        break;
      }
      case SemIR::AssociatedConstantDecl::Kind: {
        context.TODO(loc_id,
                     "Associated constant in interface with synthesized impl");
        return SemIR::ErrorInst::InstId;
      }
      default:
        CARBON_CHECK(decl_id == SemIR::ErrorInst::InstId,
                     "Unexpected kind of associated entity {0}", decl);
        return SemIR::ErrorInst::InstId;
    }
  }

  return make_witness();
}

auto GetCoreInterface(Context& context, SemIR::InterfaceId interface_id)
    -> CoreInterface {
  const auto& interface = context.interfaces().Get(interface_id);
  if (!context.name_scopes().IsCorePackage(interface.parent_scope_id) ||
      !interface.name_id.AsIdentifierId().has_value()) {
    return CoreInterface::Unknown;
  }

  for (auto [core_identifier, core_interface] :
       {std::pair{CoreIdentifier::Copy, CoreInterface::Copy},
        std::pair{CoreIdentifier::Destroy, CoreInterface::Destroy}}) {
    if (interface.name_id ==
        context.core_identifiers().AddNameId(core_identifier)) {
      return core_interface;
    }
  }
  return CoreInterface::Unknown;
}

}  // namespace Carbon::Check

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/custom_witness.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/facet_type.h"
#include "toolchain/check/function.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/impl.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/type.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Given a value whose type `IsFacetTypeOrError`, returns the corresponding
// type.
static auto GetFacetAsType(Context& context,
                           SemIR::ConstantId facet_or_type_const_id)
    -> SemIR::TypeId {
  auto facet_or_type_id =
      context.constant_values().GetInstId(facet_or_type_const_id);
  auto type_type_id = context.insts().Get(facet_or_type_id).type_id();
  CARBON_CHECK(context.types().IsFacetTypeOrError(type_type_id));

  if (context.types().Is<SemIR::FacetType>(type_type_id)) {
    // It's a facet; access its type.
    facet_or_type_id = context.types().GetTypeInstId(
        GetFacetAccessType(context, facet_or_type_id));
  }
  return context.types().GetTypeIdForTypeInstId(facet_or_type_id);
}

// Returns a manufactured no-op function with `self_const_id` as parameter.
// TODO: This is somewhat temporary, but we may want to keep something similar
// long-term where names are based on type structure (potentially also for
// copy/move).
static auto MakeNoOpFunction(Context& context, SemIR::LocId loc_id,
                             SemIR::NameScopeId name_scope_id,
                             SemIR::NameId name_id,
                             SemIR::ConstantId self_const_id) -> SemIR::InstId {
  auto self_type_id = GetFacetAsType(context, self_const_id);
  return MakeBuiltinFunction(context, loc_id, SemIR::BuiltinFunctionKind::NoOp,
                             name_scope_id, name_id,
                             {.self_type_id = self_type_id});
}

static auto MakeCustomWitnessConstantInst(
    Context& context, SemIR::LocId loc_id,
    SemIR::SpecificInterfaceId query_specific_interface_id,
    SemIR::InstBlockId associated_entities_block_id) -> SemIR::InstId {
  // The witness is a CustomWitness of the query interface with a table that
  // contains each associated entity.
  auto const_id = EvalOrAddInst<SemIR::CustomWitness>(
      context, loc_id,
      {.type_id = GetSingletonType(context, SemIR::WitnessType::TypeInstId),
       .elements_id = associated_entities_block_id,
       .query_specific_interface_id = query_specific_interface_id});
  return context.constant_values().GetInstId(const_id);
}

struct TypesForSelfFacet {
  // A FacetType that contains only the query interface.
  SemIR::TypeId facet_type_for_query_specific_interface;
  // The query self as a type, which involves a conversion if it was a facet.
  SemIR::TypeId query_self_as_type_id;
};

static auto GetTypesForSelfFacet(
    Context& context, SemIR::LocId loc_id,
    SemIR::ConstantId query_self_const_id,
    SemIR::SpecificInterfaceId query_specific_interface_id)
    -> TypesForSelfFacet {
  const auto query_specific_interface =
      context.specific_interfaces().Get(query_specific_interface_id);

  // The Self facet will have type FacetType, for the query interface.
  auto facet_type_for_query_specific_interface =
      context.types().GetTypeIdForTypeConstantId(
          EvalOrAddInst<SemIR::FacetType>(
              context, loc_id,
              FacetTypeFromInterface(context,
                                     query_specific_interface.interface_id,
                                     query_specific_interface.specific_id)));
  // The Self facet needs to point to a type value. If it's not one already,
  // convert to type.
  auto query_self_as_type_id = GetFacetAsType(context, query_self_const_id);
  return {facet_type_for_query_specific_interface, query_self_as_type_id};
}

// Build a new facet from the query self, using a CustomWitness for the query
// interface with an entry for each associated entity so far.
static auto MakeSelfFacetWithCustomWitness(
    Context& context, SemIR::LocId loc_id, TypesForSelfFacet query_types,
    SemIR::SpecificInterfaceId query_specific_interface_id,
    SemIR::InstBlockId associated_entities_block_id) -> SemIR::ConstantId {
  // We are building a facet value for a single interface, so the witness block
  // is a single witness for that interface.
  auto witnesses_block_id =
      context.inst_blocks().Add({MakeCustomWitnessConstantInst(
          context, loc_id, query_specific_interface_id,
          associated_entities_block_id)});

  return EvalOrAddInst<SemIR::FacetValue>(
      context, loc_id,
      {.type_id = query_types.facet_type_for_query_specific_interface,
       .type_inst_id =
           context.types().GetTypeInstId(query_types.query_self_as_type_id),
       .witnesses_block_id = witnesses_block_id});
}

auto BuildCustomWitness(Context& context, SemIR::LocId loc_id,
                        SemIR::ConstantId query_self_const_id,
                        SemIR::SpecificInterfaceId query_specific_interface_id,
                        llvm::ArrayRef<SemIR::InstId> values) -> SemIR::InstId {
  const auto query_specific_interface =
      context.specific_interfaces().Get(query_specific_interface_id);
  const auto& interface =
      context.interfaces().Get(query_specific_interface.interface_id);
  auto assoc_entities =
      context.inst_blocks().GetOrEmpty(interface.associated_entities_id);
  if (assoc_entities.size() != values.size()) {
    context.TODO(loc_id, ("Unsupported definition of interface " +
                          context.names().GetFormatted(interface.name_id))
                             .str());
    return SemIR::ErrorInst::InstId;
  }

  auto query_types_for_self_facet = GetTypesForSelfFacet(
      context, loc_id, query_self_const_id, query_specific_interface_id);

  // The values that will go in the witness table.
  llvm::SmallVector<SemIR::InstId> entries;

  // Fill in the witness table.
  for (const auto& [assoc_entity_id, value_id] :
       llvm::zip_equal(assoc_entities, values)) {
    LoadImportRef(context, assoc_entity_id);

    // Build a witness with the current contents of the witness table. This will
    // grow as we progress through the impl. In theory this will build O(n^2)
    // table entries, but in practice n <= 2, so that's OK.
    //
    // This is necessary because later associated entities may refer to earlier
    // associated entities in their signatures. In particular, an associated
    // result type may be used as the return type of an associated function.
    auto self_facet = MakeSelfFacetWithCustomWitness(
        context, loc_id, query_types_for_self_facet,
        query_specific_interface_id, context.inst_blocks().Add(entries));
    auto interface_with_self_specific_id = MakeSpecificWithInnerSelf(
        context, loc_id, interface.generic_id, interface.generic_with_self_id,
        query_specific_interface.specific_id, self_facet);

    auto decl_id =
        context.constant_values().GetInstId(SemIR::GetConstantValueInSpecific(
            context.sem_ir(), interface_with_self_specific_id,
            assoc_entity_id));
    CARBON_CHECK(decl_id.has_value(), "Non-constant associated entity");
    auto decl = context.insts().Get(decl_id);
    CARBON_KIND_SWITCH(decl) {
      case CARBON_KIND(SemIR::StructValue struct_value): {
        if (struct_value.type_id == SemIR::ErrorInst::TypeId) {
          return SemIR::ErrorInst::InstId;
        }
        // TODO: If a thunk is needed, this will build a different value each
        // time it's called, so we won't properly deduplicate repeated
        // witnesses.
        entries.push_back(CheckAssociatedFunctionImplementation(
            context,
            context.types().GetAs<SemIR::FunctionType>(struct_value.type_id),
            query_specific_interface.specific_id, value_id,
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

  // TODO: Consider building one witness after all associated constants, and
  // then a second after all associated functions, rather than building one in
  // each `StructValue`. Right now the code is written assuming at most one
  // function, though this CHECK can be removed as a temporary workaround.
  CARBON_CHECK(entries.size() <= 1,
               "TODO: Support multiple associated functions");

  return MakeCustomWitnessConstantInst(context, loc_id,
                                       query_specific_interface_id,
                                       context.inst_blocks().Add(entries));
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

// Returns true if the `Self` should impl `Destroy`.
static auto TypeCanDestroy(Context& context,
                           SemIR::ConstantId query_self_const_id,
                           SemIR::InterfaceId destroy_interface_id) -> bool {
  auto inst = context.insts().Get(context.constant_values().GetInstId(
      GetCanonicalFacetOrTypeValue(context, query_self_const_id)));

  // For facet values, look if the FacetType provides the same.
  if (auto facet_type =
          context.types().TryGetAs<SemIR::FacetType>(inst.type_id())) {
    const auto& info = context.facet_types().Get(facet_type->facet_type_id);
    for (auto interface : info.extend_constraints) {
      if (interface.interface_id == destroy_interface_id) {
        return true;
      }
    }
  }

  CARBON_KIND_SWITCH(inst) {
    case CARBON_KIND(SemIR::ClassType class_type): {
      auto class_info = context.classes().Get(class_type.class_id);
      // Incomplete and abstract classes can't be destroyed.
      if (!class_info.is_complete() ||
          class_info.inheritance_kind ==
              SemIR::Class::InheritanceKind::Abstract) {
        return false;
      }

      // `LookupCppImpl` handles C++ types.
      if (context.name_scopes().Get(class_info.scope_id).is_cpp_scope()) {
        return false;
      }

      // TODO: Return false if the object repr doesn't impl `Destroy`.
      return true;
    }
    case SemIR::ArrayType::Kind:
    case SemIR::ConstType::Kind:
    case SemIR::MaybeUnformedType::Kind:
    case SemIR::PartialType::Kind:
    case SemIR::StructType::Kind:
    case SemIR::TupleType::Kind:
      // TODO: Return false for types that indirectly reference a type that
      // doesn't impl `Destroy`.
      return true;
    case SemIR::BoolType::Kind:
    case SemIR::PointerType::Kind:
      // Trivially destructible.
      return true;
    default:
      return false;
  }
}

auto LookupCustomWitness(Context& context, SemIR::LocId loc_id,
                         CoreInterface core_interface,
                         SemIR::ConstantId query_self_const_id,
                         SemIR::SpecificInterfaceId query_specific_interface_id)
    -> std::optional<SemIR::InstId> {
  // TODO: Handle more interfaces, particularly copy, move, and conversion.
  if (core_interface != CoreInterface::Destroy) {
    return std::nullopt;
  }

  auto query_specific_interface =
      context.specific_interfaces().Get(query_specific_interface_id);

  if (!TypeCanDestroy(context, query_self_const_id,
                      query_specific_interface.interface_id)) {
    return std::nullopt;
  }

  if (query_self_const_id.is_symbolic()) {
    return SemIR::InstId::None;
  }

  // TODO: This needs more complex logic to apply the correct behavior. Also, we
  // should avoid building a new function on each lookup since a similar query
  // could result in identical functions.
  auto noop_id = MakeNoOpFunction(
      context, loc_id, SemIR::NameScopeId::None,
      SemIR::NameId::ForIdentifier(context.identifiers().Add("DestroyOp")),
      query_self_const_id);
  return BuildCustomWitness(context, loc_id, query_self_const_id,
                            query_specific_interface_id, {noop_id});
}

}  // namespace Carbon::Check

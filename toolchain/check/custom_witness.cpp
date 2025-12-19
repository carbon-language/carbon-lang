// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/custom_witness.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/impl.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/pattern.h"
#include "toolchain/check/pattern_match.h"
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

// Returns a manufactured no-op function with `self_const_id` as parameter.
// TODO: This is somewhat temporary, but we may want to keep something similar
// long-term where names are based on type structure (potentially also for
// copy/move). It'll probably be good to look at refactoring with function
// construction in thunk.cpp and cpp/import.cpp.
static auto MakeNoOpFunction(Context& context, SemIR::LocId loc_id,
                             SemIR::ConstantId self_const_id,
                             SemIR::SpecificId specific_id) -> SemIR::InstId {
  // Build the parameters, with `[ref self: Self]`
  context.scope_stack().PushForDeclName();
  context.inst_block_stack().Push();
  context.pattern_block_stack().Push();

  BeginSubpattern(context);
  auto type_id = GetFacetAsType(context, loc_id, self_const_id);
  SemIR::ExprRegionId type_expr_region_id =
      EndSubpatternAsExpr(context, context.types().GetInstId(type_id));
  auto self_param_id = AddParamPattern(
      context, loc_id, SemIR::NameId::SelfValue, type_expr_region_id, type_id,
      /*is_ref=*/true);
  auto implicit_param_patterns_id = context.inst_blocks().Add({self_param_id});
  auto call_params_id =
      CalleePatternMatch(context, implicit_param_patterns_id,
                         /*param_patterns_id=*/SemIR::InstBlockId::Empty,
                         /*return_patterns_id=*/SemIR::InstBlockId::None);

  auto pattern_block_id = context.pattern_block_stack().Pop();
  auto decl_block_id = context.inst_block_stack().Pop();
  context.scope_stack().Pop();

  // Add the function declaration.
  SemIR::FunctionDecl function_decl = {.type_id = SemIR::TypeId::None,
                                       .function_id = SemIR::FunctionId::None,
                                       .decl_block_id = decl_block_id};
  auto noop_id = AddPlaceholderInstInNoBlock(
      context, SemIR::LocIdAndInst::UncheckedLoc(loc_id, function_decl));

  auto noop_name_id =
      SemIR::NameId::ForIdentifier(context.identifiers().Add("DestroyOp"));

  // Build the function entity.
  auto noop_function = SemIR::Function{
      {
          .name_id = noop_name_id,
          .parent_scope_id = SemIR::NameScopeId::None,
          .generic_id = SemIR::GenericId::None,
          .first_param_node_id = Parse::NodeId::None,
          .last_param_node_id = Parse::NodeId::None,
          .pattern_block_id = pattern_block_id,
          .implicit_param_patterns_id = implicit_param_patterns_id,
          .param_patterns_id = SemIR::InstBlockId::Empty,
          .is_extern = false,
          .extern_library_id = SemIR::LibraryNameId::None,
          .non_owning_decl_id = SemIR::InstId::None,
          .first_owning_decl_id = noop_id,
      },
      {
          .call_params_id = call_params_id,
          .return_type_inst_id = SemIR::TypeInstId::None,
          .return_patterns_id = SemIR::InstBlockId::None,
          .self_param_id = self_param_id,
      }};
  noop_function.SetBuiltinFunction(SemIR::BuiltinFunctionKind::NoOp);
  function_decl.function_id = context.functions().Add(noop_function);
  function_decl.type_id =
      GetFunctionType(context, function_decl.function_id, specific_id);
  ReplaceInstBeforeConstantUse(context, noop_id, function_decl);
  return noop_id;
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
             .elements_id = context.inst_blocks().Add(entries),
             .query_specific_interface_id = query_specific_interface_id}));
  };

  // Fill in the witness table.
  for (const auto& [assoc_entity_id, value_id] :
       llvm::zip_equal(assoc_entities, values)) {
    LoadImportRef(context, assoc_entity_id);
    auto decl_id =
        context.constant_values().GetInstId(SemIR::GetConstantValueInSpecific(
            context.sem_ir(), query_specific_interface.specific_id,
            assoc_entity_id));
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
      // TODO: Return false if the object repr doesn't impl `Destroy`.
      // TODO: This should probably be skipped for all C++ types, but currently
      // must handle those for trivial destruction.
      return class_info.is_complete() &&
             class_info.inheritance_kind !=
                 SemIR::Class::InheritanceKind::Abstract;
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
    -> SemIR::InstId {
  // TODO: Handle more interfaces, particularly copy, move, and conversion.
  if (core_interface != CoreInterface::Destroy) {
    return SemIR::InstId::None;
  }

  auto query_specific_interface =
      context.specific_interfaces().Get(query_specific_interface_id);

  if (!TypeCanDestroy(context, query_self_const_id,
                      query_specific_interface.interface_id)) {
    return SemIR::InstId::None;
  }

  // TODO: This needs more complex logic to apply the correct behavior. Also, we
  // should avoid building a new function on each lookup since a similar query
  // could result in identical functions.
  auto noop_id = MakeNoOpFunction(context, loc_id, query_self_const_id,
                                  query_specific_interface.specific_id);
  return BuildCustomWitness(context, loc_id, query_self_const_id,
                            query_specific_interface_id, {noop_id});
}

}  // namespace Carbon::Check

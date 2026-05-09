// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/custom_witness.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/facet_type.h"
#include "toolchain/check/function.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/impl.h"
#include "toolchain/check/impl_lookup.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/sem_ir/associated_constant.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/constant.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/type_info.h"
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

// Returns a manufactured `Copy.Op` function with the `self` parameter typed
// to `self_type_id`.
static auto MakeCopyOpFunction(Context& context, SemIR::LocId loc_id,
                               SemIR::TypeId self_type_id,
                               SemIR::NameScopeId parent_scope_id)
    -> SemIR::InstId {
  auto name_id = context.core_identifiers().AddNameId(CoreIdentifier::Op);

  auto [decl_id, function_id] =
      MakeGeneratedFunctionDecl(context, loc_id,
                                {.parent_scope_id = parent_scope_id,
                                 .name_id = name_id,
                                 .self_type_id = self_type_id,
                                 .self_kind = ParamPatternKind::Value,
                                 .return_type_id = self_type_id});

  auto& function = context.functions().Get(function_id);
  function.SetCoreWitness(SemIR::BuiltinFunctionKind::PrimitiveCopy);

  return decl_id;
}

// Returns the body for `Destroy.Op`. This will return `None` if using the
// builtin `NoOp` is appropriate.
// Returns a FacetType that contains only the query interface.
static auto GetFacetTypeForQuerySpecificInterface(
    Context& context, SemIR::LocId loc_id,
    SemIR::SpecificInterfaceId query_specific_interface_id)
    -> SemIR::ConstantId {
  const auto query_specific_interface =
      context.specific_interfaces().Get(query_specific_interface_id);

  // The Self facet will have type FacetType, for the query interface.
  auto const_id = EvalOrAddInst<SemIR::FacetType>(
      context, loc_id,
      FacetTypeFromInterface(context, query_specific_interface.interface_id,
                             query_specific_interface.specific_id));
  return const_id;
}

// Starts a block for lookup-related instructions, and returns the `FacetType`
// for lookups in `HasWitnessForRepeatedField`.
static auto PrepareForHasWitness(
    Context& context, SemIR::LocId loc_id,
    SemIR::SpecificInterfaceId query_specific_interface_id)
    -> SemIR::ConstantId {
  context.inst_block_stack().Push();
  StartGenericDecl(context);

  return GetFacetTypeForQuerySpecificInterface(context, loc_id,
                                               query_specific_interface_id);
}

// Cleans up state `PrepareForHasWitness`.
static auto CleanupAfterHasWitness(Context& context) -> void {
  DiscardGenericDecl(context);
  context.inst_block_stack().PopAndDiscard();
}

// Returns true if `type_inst_id` has a witness for the query interface, which
// comes from `PrepareForHasWitness`.
static auto HasWitnessForRepeatedField(
    Context& context, SemIR::LocId loc_id, SemIR::InstId type_inst_id,
    SemIR::ConstantId query_facet_type_const_id) -> bool {
  auto type_const_id = context.constant_values().Get(type_inst_id);
  auto block_or_err = LookupImplWitness(context, loc_id, type_const_id,
                                        query_facet_type_const_id);
  return block_or_err.has_value();
}

// The format for `Destroy.Op`.
enum class DestroyFormat {
  NoDestroy,
  Trivial,
  NonTrivial,
};

// Similar to `HasWitnessForRepeatedField`, but for cases where there's only one
// field, this can handle the call to `PrepareForHasWitness`.
static auto HasWitnessForOneField(
    Context& context, SemIR::LocId loc_id, SemIR::InstId field_inst_id,
    SemIR::SpecificInterfaceId query_specific_interface_id) -> DestroyFormat {
  auto query_facet_type_const_id =
      PrepareForHasWitness(context, loc_id, query_specific_interface_id);
  auto has_witness = HasWitnessForRepeatedField(context, loc_id, field_inst_id,
                                                query_facet_type_const_id);
  CleanupAfterHasWitness(context);
  return has_witness ? DestroyFormat::NonTrivial : DestroyFormat::NoDestroy;
}

// Returns true if `class_type` should impl `Destroy`.
static auto CanDestroyClass(
    Context& context, SemIR::LocId loc_id, SemIR::ClassType class_type,
    const SemIR::CompleteTypeInfo& complete_info,
    SemIR::SpecificInterfaceId query_specific_interface_id, bool is_partial)
    -> DestroyFormat {
  // Abstract classes can't be destroyed.
  if (!is_partial && complete_info.IsAbstract()) {
    return DestroyFormat::NoDestroy;
  }

  auto class_info = context.classes().Get(class_type.class_id);

  // `LookupCppImpl` handles C++ types.
  if (context.name_scopes().Get(class_info.scope_id).is_cpp_scope()) {
    return DestroyFormat::NoDestroy;
  }

  auto object_repr_id =
      class_info.GetObjectRepr(context.sem_ir(), class_type.specific_id);
  return HasWitnessForOneField(context, loc_id,
                               context.types().GetTypeInstId(object_repr_id),
                               query_specific_interface_id);
}

// Returns true if the `Self` should impl `Destroy`. This will recurse into impl
// lookup of `Destroy` for members, similar to `where .Self.members each impls
// Destroy`.
static auto CanDestroyType(
    Context& context, SemIR::LocId loc_id,
    SemIR::ConstantId query_self_const_id,
    SemIR::SpecificInterfaceId query_specific_interface_id) -> DestroyFormat {
  auto query_specific_interface =
      context.specific_interfaces().Get(query_specific_interface_id);
  auto destroy_interface_id = query_specific_interface.interface_id;

  auto inst_id = context.constant_values().GetInstId(
      GetCanonicalFacetOrTypeValue(context, query_self_const_id));
  auto inst = context.insts().Get(inst_id);

  // For facet values, look if the FacetType provides the same.
  if (auto facet_type =
          context.types().TryGetAs<SemIR::FacetType>(inst.type_id())) {
    const auto& info = context.facet_types().Get(facet_type->facet_type_id);
    for (auto interface : info.extend_constraints) {
      if (interface.interface_id == destroy_interface_id) {
        return DestroyFormat::Trivial;
      }
    }
    return DestroyFormat::NoDestroy;
  }

  // Incomplete types can not be destroyed.
  auto type_id = context.types().GetTypeIdForTypeInstId(inst_id);
  if (!TryToCompleteType(context, type_id, loc_id)) {
    return DestroyFormat::NoDestroy;
  }

  CARBON_KIND_SWITCH(inst) {
    case SemIR::ImplWitnessAccess::Kind:
    case SemIR::SymbolicBinding::Kind: {
      // A symbolic facet of type `type`. Such symbolic values can't be
      // destroyed.
      return DestroyFormat::NoDestroy;
    }

    case CARBON_KIND(SemIR::ArrayType array_type): {
      // A zero element array is always trivially destructible.
      if (auto int_bound =
              context.sem_ir().GetZExtIntValue(array_type.bound_id);
          !int_bound || *int_bound == 0) {
        return DestroyFormat::Trivial;
      }

      // Verify the element can be destroyed.
      return HasWitnessForOneField(context, loc_id,
                                   array_type.element_type_inst_id,
                                   query_specific_interface_id);
    }

    case SemIR::Call::Kind:
      // TODO: These seem like they shouldn't be getting directly queried for
      // destroy. The use is in a test that was TODO before this TODO.
      return DestroyFormat::NoDestroy;

    case CARBON_KIND(SemIR::ClassType class_type): {
      return CanDestroyClass(context, loc_id, class_type,
                             context.types().GetCompleteTypeInfo(type_id),
                             query_specific_interface_id,
                             /*is_partial=*/false);
    }

    case CARBON_KIND(SemIR::ConstType const_type): {
      return HasWitnessForOneField(context, loc_id, const_type.inner_id,
                                   query_specific_interface_id);
    }

    case CARBON_KIND(SemIR::MaybeUnformedType maybe_unformed_type): {
      return HasWitnessForOneField(context, loc_id,
                                   maybe_unformed_type.inner_id,
                                   query_specific_interface_id);
    }

    case CARBON_KIND(SemIR::PartialType partial_type): {
      // In contrast with something like `const`, need to treat the inner
      // class differently based on the `partial` modifier.
      auto class_type =
          context.insts().GetAs<SemIR::ClassType>(partial_type.inner_id);
      return CanDestroyClass(context, loc_id, class_type,
                             context.types().GetCompleteTypeInfo(type_id),
                             query_specific_interface_id,
                             /*is_partial=*/true);
    }

    case CARBON_KIND(SemIR::StructType struct_type): {
      auto fields = context.struct_type_fields().Get(struct_type.fields_id);
      if (fields.empty()) {
        return DestroyFormat::Trivial;
      }
      auto query_facet_type_const_id =
          PrepareForHasWitness(context, loc_id, query_specific_interface_id);
      bool has_witness = true;
      for (const auto& field : fields) {
        if (!HasWitnessForRepeatedField(context, loc_id, field.type_inst_id,
                                        query_facet_type_const_id)) {
          has_witness = false;
          break;
        }
      }
      CleanupAfterHasWitness(context);
      return has_witness ? DestroyFormat::NonTrivial : DestroyFormat::NoDestroy;
    }

    case CARBON_KIND(SemIR::TupleType tuple_type): {
      auto block = context.inst_blocks().Get(tuple_type.type_elements_id);
      if (block.empty()) {
        return DestroyFormat::Trivial;
      }
      auto query_facet_type_const_id =
          PrepareForHasWitness(context, loc_id, query_specific_interface_id);
      bool has_witness = true;
      for (const auto& element_id : block) {
        if (!HasWitnessForRepeatedField(context, loc_id, element_id,
                                        query_facet_type_const_id)) {
          has_witness = false;
          break;
        }
      }
      CleanupAfterHasWitness(context);
      return has_witness ? DestroyFormat::NonTrivial : DestroyFormat::NoDestroy;
    }

    case SemIR::BoolType::Kind:
    case SemIR::FacetType::Kind:
    case SemIR::FloatType::Kind:
    case SemIR::IntLiteralType::Kind:
    case SemIR::IntType::Kind:
    case SemIR::PointerType::Kind:
    case SemIR::TypeType::Kind:
      // Trivially destructible.
      return DestroyFormat::Trivial;

    default:
      CARBON_FATAL("Unexpected type for CanDestroyType: {0}", inst.kind());
  }
}

// Returns the body for `Destroy.Op`.
//
// TODO: This is a placeholder still not actually destroying things, intended to
// maintain mostly-consistent behavior with current logic while working. That
// also means using `self`.
static auto MakeDestroyOpBody(Context& context, SemIR::LocId loc_id,
                              SemIR::TypeId self_type_id,
                              SemIR::InstId self_param_id)
    -> SemIR::InstBlockId {
  context.inst_block_stack().Push();
  auto inst = context.types().GetAsInst(self_type_id);

  CARBON_KIND_SWITCH(inst) {
    case SemIR::ArrayType::Kind:
    case SemIR::ClassType::Kind:
    case SemIR::ConstType::Kind:
    case SemIR::MaybeUnformedType::Kind:
    case SemIR::PartialType::Kind:
    case SemIR::StructType::Kind:
    case SemIR::TupleType::Kind:
      (void)self_param_id;
      // TODO: Implement destruction of the type.
      break;
    default:
      CARBON_FATAL("Unexpected type for MakeDestroyOpBody: {0}", inst);
  }

  AddInst(context, loc_id, SemIR::Return{});
  return context.inst_block_stack().Pop();
}

// Returns a manufactured `Destroy.Op` function with the `self` parameter typed
// to `self_type_id`.
static auto MakeDestroyOpFunction(Context& context, SemIR::LocId loc_id,
                                  SemIR::TypeId self_type_id,
                                  SemIR::NameScopeId parent_scope_id,
                                  DestroyFormat format) -> SemIR::InstId {
  auto name_id = context.core_identifiers().AddNameId(CoreIdentifier::Op);

  auto [decl_id, function_id] =
      MakeGeneratedFunctionDecl(context, loc_id,
                                {.parent_scope_id = parent_scope_id,
                                 .name_id = name_id,
                                 .self_type_id = self_type_id,
                                 .self_kind = ParamPatternKind::Ref});

  auto& function = context.functions().Get(function_id);

  if (format == DestroyFormat::Trivial) {
    function.SetCoreWitness(SemIR::BuiltinFunctionKind::NoOp);
  } else {
    CARBON_CHECK(format == DestroyFormat::NonTrivial);
    function.SetCoreWitness(SemIR::BuiltinFunctionKind::None);
    auto body_id = MakeDestroyOpBody(context, loc_id, self_type_id,
                                     function.self_param_id);
    function.body_block_ids.push_back(body_id);
  }

  return decl_id;
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
  // The Self facet will have type FacetType, for the query interface.
  auto facet_type_for_query_specific_interface =
      context.types().GetTypeIdForTypeConstantId(
          GetFacetTypeForQuerySpecificInterface(context, loc_id,
                                                query_specific_interface_id));
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

  auto interface_with_self_specific_id = SemIR::SpecificId::None;

  enum class AssociatedEntityState {
    None,
    AssociatedConstant,
    AssociatedFunction
  };
  auto associated_entity_state = AssociatedEntityState::None;
  // Build a witness with the current contents of the witness table. Each
  // specific interface has at most two witness tables: an optional table
  // for associated constants and a table for associated functions.
  //
  // We need to separate associated constants and associated functions since
  // associated entities can't depend on other entities in their own witness
  // table. To ensure that we don't end up with n^2 witness tables, and so
  // that we don't need to loop over the associated entities more than once,
  // we require that interfaces with a custom witness specify all of their
  // associated constants before their associated functions. Interfaces with
  // custom witness tables are hardcoded in the compiler, so this
  // restriction doesn't impact whether or not a type is definable.
  auto update_interface_with_self_specific_id =
      [&](SemIR::InstId assoc_entity_id) {
        auto decl_id =
            context.constant_values().GetConstantInstId(assoc_entity_id);
        CARBON_CHECK(decl_id.has_value(), "Non-constant associated entity");
        auto decl = context.insts().Get(decl_id);
        auto new_associated_entity_state =
            decl.kind() == SemIR::AssociatedConstantDecl::Kind
                ? AssociatedEntityState::AssociatedConstant
                : AssociatedEntityState::AssociatedFunction;
        CARBON_CHECK(new_associated_entity_state >= associated_entity_state,
                     "Implementation restriction: associated constants must be "
                     "defined before associated functions");
        if (associated_entity_state < new_associated_entity_state) {
          auto self_facet = MakeSelfFacetWithCustomWitness(
              context, loc_id, query_types_for_self_facet,
              query_specific_interface_id, context.inst_blocks().Add(entries));
          interface_with_self_specific_id = MakeSpecificWithInnerSelf(
              context, loc_id, interface.generic_id,
              interface.generic_with_self_id,
              query_specific_interface.specific_id, self_facet);
          associated_entity_state = new_associated_entity_state;
        }
      };

  // Fill in the witness table.
  for (const auto& [assoc_entity_id, value_id] :
       llvm::zip_equal(assoc_entities, values)) {
    LoadImportRef(context, assoc_entity_id);
    update_interface_with_self_specific_id(assoc_entity_id);
    CARBON_CHECK(
        !context.specifics().Get(interface_with_self_specific_id).HasError());
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
      case CARBON_KIND(SemIR::AssociatedConstantDecl decl): {
        if (decl.type_id == SemIR::ErrorInst::TypeId) {
          return SemIR::ErrorInst::InstId;
        }

        // TODO: remove once we have a test-case for all associated constants.
        // Special-case the ones we want to support in this if-statement, until
        // we're able to account for everything.
        if (decl.type_id != SemIR::TypeType::TypeId) {
          context.TODO(loc_id,
                       "Associated constant of type other than `TypeType` in "
                       "synthesized impl");
          return SemIR::ErrorInst::InstId;
        }

        auto type_id = context.insts().Get(value_id).type_id();
        CARBON_CHECK(type_id == SemIR::TypeType::TypeId ||
                     type_id == SemIR::ErrorInst::TypeId);
        auto impl_witness_associated_constant =
            AddInst<SemIR::ImplWitnessAssociatedConstant>(
                context, loc_id, {.type_id = type_id, .inst_id = value_id});
        entries.push_back(impl_witness_associated_constant);
        break;
      }
      default:
        CARBON_CHECK(decl_id == SemIR::ErrorInst::InstId,
                     "Unexpected kind of associated entity {0}", decl);
        return SemIR::ErrorInst::InstId;
    }
  }

  return MakeCustomWitnessConstantInst(context, loc_id,
                                       query_specific_interface_id,
                                       context.inst_blocks().Add(entries));
}

auto AsCoreIdentifier(SemIR::CoreInterface core_interface) -> CoreIdentifier {
  switch (core_interface) {
#define CARBON_SEM_IR_CORE_INTERFACE_EXCLUDE_UNKNOWN
#define CARBON_SEM_IR_CORE_INTERFACE_KIND(Name) \
  case SemIR::CoreInterface::Name:              \
    return CoreIdentifier::Name;
#include "toolchain/sem_ir/core_interface_kind.def"
    case SemIR::CoreInterface::Unknown:
      CARBON_FATAL("{0} doesn't have a `CoreIdentifier` mapping",
                   core_interface);
  }
}

auto GetCoreInterface(Context& context, SemIR::InterfaceId interface_id)
    -> SemIR::CoreInterface {
  const auto& interface = context.interfaces().Get(interface_id);
  if (!context.name_scopes().IsCorePackage(interface.parent_scope_id) ||
      !interface.name_id.AsIdentifierId().has_value()) {
    return SemIR::CoreInterface::Unknown;
  }

  return interface.core_interface;
}

auto BuildPrimitiveCopyWitness(
    Context& context, SemIR::LocId loc_id, SemIR::NameScopeId parent_scope_id,
    SemIR::ConstantId query_self_const_id,
    SemIR::SpecificInterfaceId query_specific_interface_id) -> SemIR::InstId {
  auto self_type_id = GetFacetAsType(context, query_self_const_id);
  auto op_id =
      MakeCopyOpFunction(context, loc_id, self_type_id, parent_scope_id);
  return BuildCustomWitness(context, loc_id, query_self_const_id,
                            query_specific_interface_id, {op_id});
}
static auto MakeDestroyWitness(
    Context& context, SemIR::LocId loc_id,
    SemIR::ConstantId query_self_const_id,
    SemIR::SpecificInterfaceId query_specific_interface_id, bool build_witness)
    -> std::optional<SemIR::InstId> {
  auto format = CanDestroyType(context, loc_id, query_self_const_id,
                               query_specific_interface_id);
  if (format == DestroyFormat::NoDestroy) {
    return std::nullopt;
  }

  if (!build_witness || query_self_const_id.is_symbolic()) {
    // The type can be destroyed, but we shouldn't make a witness right now.
    return SemIR::InstId::None;
  }

  // Mark functions with the interface's scope as a hint to mangling. This
  // does not add them to the scope.
  auto query_specific_interface =
      context.specific_interfaces().Get(query_specific_interface_id);
  auto parent_scope_id = context.interfaces()
                             .Get(query_specific_interface.interface_id)
                             .scope_without_self_id;

  auto self_type_id = GetFacetAsType(context, query_self_const_id);
  auto op_id = MakeDestroyOpFunction(context, loc_id, self_type_id,
                                     parent_scope_id, format);
  return BuildCustomWitness(context, loc_id, query_self_const_id,
                            query_specific_interface_id, {op_id});
}

static auto MakeIntFitsInWitness(
    Context& context, SemIR::LocId loc_id,
    SemIR::ConstantId query_self_const_id,
    SemIR::SpecificInterfaceId query_specific_interface_id, bool build_witness)
    -> std::optional<SemIR::InstId> {
  auto query_specific_interface =
      context.specific_interfaces().Get(query_specific_interface_id);

  auto args_id = query_specific_interface.specific_id;
  if (!args_id.has_value()) {
    return std::nullopt;
  }
  auto args_block_id = context.specifics().Get(args_id).args_id;
  auto args_block = context.inst_blocks().Get(args_block_id);
  if (args_block.size() != 1) {
    return std::nullopt;
  }

  auto dest_const_id = context.constant_values().Get(args_block[0]);
  if (!dest_const_id.is_constant()) {
    return std::nullopt;
  }

  auto src_type_id = GetFacetAsType(context, query_self_const_id);
  auto dest_type_id = GetFacetAsType(context, dest_const_id);

  auto context_fn = [](DiagnosticContextBuilder& /*builder*/) -> void {};
  if (!RequireCompleteType(context, src_type_id, loc_id, context_fn) ||
      !RequireCompleteType(context, dest_type_id, loc_id, context_fn)) {
    return std::nullopt;
  }

  auto src_info = context.types().TryGetIntTypeInfo(src_type_id);
  auto dest_info = context.types().TryGetIntTypeInfo(dest_type_id);

  if (!src_info || !dest_info) {
    return std::nullopt;
  }

  // If the bit width is unknown (e.g., due to symbolic evaluation), we cannot
  // determine whether it fits yet.
  if (src_info->bit_width == IntId::None ||
      dest_info->bit_width == IntId::None) {
    return std::nullopt;
  }

  const auto& src_width = context.ints().Get(src_info->bit_width);
  const auto& dest_width = context.ints().Get(dest_info->bit_width);

  bool fits = false;
  if (src_info->is_signed && !dest_info->is_signed) {
    // Signed -> unsigned: would truncate the sign bit.
    fits = false;
  } else if (src_info->is_signed == dest_info->is_signed) {
    // Signed -> signed or unsigned -> unsigned: allow widening or preserving
    // width.
    fits = src_width.sle(dest_width);
  } else {
    // Unsigned -> signed: strict widening required.
    fits = src_width.slt(dest_width);
  }

  if (!fits) {
    return std::nullopt;
  }

  if (!build_witness) {
    return SemIR::InstId::None;
  }

  return BuildCustomWitness(context, loc_id, query_self_const_id,
                            query_specific_interface_id, {});
}

auto LookupCustomWitness(Context& context, SemIR::LocId loc_id,
                         SemIR::CoreInterface core_interface,
                         SemIR::ConstantId query_self_const_id,
                         SemIR::SpecificInterfaceId query_specific_interface_id,
                         bool build_witness) -> std::optional<SemIR::InstId> {
  switch (core_interface) {
    case SemIR::CoreInterface::Destroy:
      return MakeDestroyWitness(context, loc_id, query_self_const_id,
                                query_specific_interface_id, build_witness);
    case SemIR::CoreInterface::IntFitsIn:
      return MakeIntFitsInWitness(context, loc_id, query_self_const_id,
                                  query_specific_interface_id, build_witness);
    case SemIR::CoreInterface::AddAssignWith:
    case SemIR::CoreInterface::AddWith:
    case SemIR::CoreInterface::Copy:
    case SemIR::CoreInterface::CppUnsafeDeref:
    case SemIR::CoreInterface::Dec:
    case SemIR::CoreInterface::Default:
    case SemIR::CoreInterface::DivAssignWith:
    case SemIR::CoreInterface::DivWith:
    case SemIR::CoreInterface::Inc:
    case SemIR::CoreInterface::ModAssignWith:
    case SemIR::CoreInterface::ModWith:
    case SemIR::CoreInterface::MulAssignWith:
    case SemIR::CoreInterface::MulWith:
    case SemIR::CoreInterface::Negate:
    case SemIR::CoreInterface::SubAssignWith:
    case SemIR::CoreInterface::SubWith:
    case SemIR::CoreInterface::Unknown:
      // TODO: Handle more interfaces, particularly copy, move, and conversion.
      return std::nullopt;
  }
}

}  // namespace Carbon::Check

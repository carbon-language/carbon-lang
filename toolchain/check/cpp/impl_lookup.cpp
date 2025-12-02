// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/impl_lookup.h"

#include "clang/Sema/Sema.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/cpp/import.h"
#include "toolchain/check/cpp/location.h"
#include "toolchain/check/cpp/overload_resolution.h"
#include "toolchain/check/impl.h"
#include "toolchain/check/impl_lookup.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/type.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// If the given type is a C++ class type, returns the corresponding class
// declaration. Otherwise returns nullptr.
// TODO: Handle qualified types.
static auto TypeAsClassDecl(Context& context, SemIR::TypeId type_id)
    -> clang::CXXRecordDecl* {
  auto class_type = context.types().TryGetAs<SemIR::ClassType>(type_id);
  if (!class_type) {
    // Not a class.
    return nullptr;
  }

  SemIR::NameScopeId class_scope_id =
      context.classes().Get(class_type->class_id).scope_id;
  if (!class_scope_id.has_value()) {
    return nullptr;
  }

  const auto& scope = context.name_scopes().Get(class_scope_id);
  auto decl_id = scope.clang_decl_context_id();
  if (!decl_id.has_value()) {
    return nullptr;
  }

  return dyn_cast<clang::CXXRecordDecl>(
      context.clang_decls().Get(decl_id).key.decl);
}

// Builds a witness that the given type implements the given interface,
// populating it with the specified set of values. Returns a corresponding
// lookup result. Produces a diagnostic and returns `None` if the specified
// values aren't suitable for the interface.
static auto BuildWitness(Context& context, SemIR::LocId loc_id,
                         SemIR::TypeId self_type_id,
                         SemIR::SpecificInterface specific_interface,
                         llvm::ArrayRef<SemIR::InstId> values)
    -> SemIR::InstId {
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

  // Prepare an empty witness table.
  auto witness_table_id =
      context.inst_blocks().AddUninitialized(assoc_entities.size());
  auto witness_table = context.inst_blocks().GetMutable(witness_table_id);
  for (auto& witness_value_id : witness_table) {
    witness_value_id = SemIR::InstId::ImplWitnessTablePlaceholder;
  }

  // Build a witness. We use an `ImplWitness` with an `impl_id` of `None` to
  // represent a synthesized witness.
  // TODO: Stop using `ImplWitnessTable` here and add a distinct instruction
  // that doesn't contain an `InstId` and supports deduplication.
  auto witness_table_inst_id = AddInst<SemIR::ImplWitnessTable>(
      context, loc_id,
      {.elements_id = witness_table_id, .impl_id = SemIR::ImplId::None});
  auto witness_id = AddInst<SemIR::ImplWitness>(
      context, loc_id,
      {.type_id = GetSingletonType(context, SemIR::WitnessType::TypeInstId),
       .witness_table_id = witness_table_inst_id,
       .specific_id = SemIR::SpecificId::None});

  // Fill in the witness table.
  for (const auto& [assoc_entity_id, value_id, witness_value_id] :
       llvm::zip_equal(assoc_entities, values, witness_table)) {
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
        witness_value_id = CheckAssociatedFunctionImplementation(
            context,
            context.types().GetAs<SemIR::FunctionType>(struct_value.type_id),
            value_id, self_type_id, witness_id,
            /*defer_thunk_definition=*/false);
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

  return witness_id;
}

static auto LookupCopyImpl(Context& context, SemIR::LocId loc_id,
                           SemIR::TypeId self_type_id,
                           SemIR::SpecificInterface specific_interface)
    -> SemIR::InstId {
  auto* class_decl = TypeAsClassDecl(context, self_type_id);
  if (!class_decl) {
    // TODO: Should we also provide a `Copy` implementation for enumerations?
    return SemIR::InstId::None;
  }

  auto* ctor = context.clang_sema().LookupCopyingConstructor(
      class_decl, clang::Qualifiers::Const);
  if (!ctor) {
    // TODO: If the impl lookup failure is an error, we should produce a
    // diagnostic explaining why the class is not copyable.
    return SemIR::InstId::None;
  }

  auto ctor_id =
      context.clang_sema().DiagnoseUseOfOverloadedDecl(
          ctor, GetCppLocation(context, loc_id))
          ? SemIR::ErrorInst::InstId
          : ImportCppFunctionDecl(context, loc_id, ctor, /*num_params=*/1);
  if (auto ctor_decl =
          context.insts().TryGetAsWithId<SemIR::FunctionDecl>(ctor_id)) {
    CheckCppOverloadAccess(context, loc_id,
                           clang::DeclAccessPair::make(ctor, ctor->getAccess()),
                           ctor_decl->inst_id);
  } else {
    CARBON_CHECK(ctor_id == SemIR::ErrorInst::InstId);
    return SemIR::ErrorInst::InstId;
  }
  return BuildWitness(context, loc_id, self_type_id, specific_interface,
                      {ctor_id});
}

auto LookupCppImpl(Context& context, SemIR::LocId loc_id,
                   SemIR::TypeId self_type_id,
                   SemIR::SpecificInterface specific_interface,
                   const TypeStructure* best_impl_type_structure,
                   SemIR::LocId best_impl_loc_id) -> SemIR::InstId {
  // Determine whether this is an interface that we have special knowledge of.
  auto& interface = context.interfaces().Get(specific_interface.interface_id);
  if (!context.name_scopes().IsCorePackage(interface.parent_scope_id)) {
    return SemIR::InstId::None;
  }
  if (!interface.name_id.AsIdentifierId().has_value()) {
    return SemIR::InstId::None;
  }

  if (context.identifiers().Get(interface.name_id.AsIdentifierId()) == "Copy") {
    return LookupCopyImpl(context, loc_id, self_type_id, specific_interface);
  }

  // TODO: Handle other interfaces.

  // TODO: Infer a C++ type structure and check whether it's less strict than
  // the best Carbon type structure.
  static_cast<void>(best_impl_type_structure);
  static_cast<void>(best_impl_loc_id);

  return SemIR::InstId::None;
}

}  // namespace Carbon::Check

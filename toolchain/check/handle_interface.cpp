// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/facet_type.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/merge.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/check/name_component.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/type.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::InterfaceIntroducerId node_id)
    -> bool {
  // Create an instruction block to hold the instructions created as part of the
  // interface signature, such as generic parameters.
  context.inst_block_stack().Push();
  // Push the bracketing node.
  context.node_stack().Push(node_id);
  // Optional modifiers and the name follow.
  context.decl_introducer_state_stack().Push<Lex::TokenKind::Interface>();
  context.decl_name_stack().PushScopeAndStartName();
  // This interface is potentially generic.
  StartGenericDecl(context);
  return true;
}

static auto BuildInterfaceDecl(Context& context,
                               Parse::AnyInterfaceDeclId node_id,
                               bool is_definition)
    -> std::tuple<SemIR::InterfaceId, SemIR::InstId> {
  auto name = PopNameComponent(context);
  auto name_context = context.decl_name_stack().FinishName(name);
  context.node_stack()
      .PopAndDiscardSoloNodeId<Parse::NodeKind::InterfaceIntroducer>();

  // Process modifiers.
  auto [_, parent_scope_inst] =
      context.name_scopes().GetInstIfValid(name_context.parent_scope_id);
  auto introducer =
      context.decl_introducer_state_stack().Pop<Lex::TokenKind::Interface>();
  CheckAccessModifiersOnDecl(context, introducer, parent_scope_inst);
  LimitModifiersOnDecl(context, introducer, KeywordModifierSet::Access);

  auto decl_block_id = context.inst_block_stack().Pop();

  // Add the interface declaration.
  auto interface_decl =
      SemIR::InterfaceDecl{SemIR::TypeType::SingletonTypeId,
                           SemIR::InterfaceId::None, decl_block_id};
  auto interface_decl_id =
      AddPlaceholderInst(context, SemIR::LocIdAndInst(node_id, interface_decl));

  SemIR::Interface interface_info = {name_context.MakeEntityWithParamsBase(
      name, interface_decl_id, /*is_extern=*/false,
      SemIR::LibraryNameId::None)};

  // Check whether this is a redeclaration.
  SemIR::ScopeLookupResult lookup_result =
      context.decl_name_stack().LookupOrAddName(
          name_context, interface_decl_id,
          introducer.modifier_set.GetAccessKind());
  if (lookup_result.is_poisoned()) {
    // This is a declaration of a poisoned name.
    DiagnosePoisonedName(context, name_context.name_id_for_new_inst(),
                         lookup_result.poisoning_loc_id(), name_context.loc_id);
  } else if (lookup_result.is_found()) {
    SemIR::InstId existing_id = lookup_result.target_inst_id();
    if (auto existing_interface_decl =
            context.insts().Get(existing_id).TryAs<SemIR::InterfaceDecl>()) {
      auto existing_interface =
          context.interfaces().Get(existing_interface_decl->interface_id);
      if (CheckRedeclParamsMatch(
              context,
              DeclParams(interface_decl_id, name.first_param_node_id,
                         name.last_param_node_id,
                         name.implicit_param_patterns_id,
                         name.param_patterns_id),
              DeclParams(existing_interface))) {
        // TODO: This should be refactored a little, particularly for
        // prev_import_ir_id. See similar logic for classes and functions, which
        // might also be refactored to merge.
        DiagnoseIfInvalidRedecl(
            context, Lex::TokenKind::Interface, existing_interface.name_id,
            RedeclInfo(interface_info, node_id, is_definition),
            RedeclInfo(existing_interface, existing_interface.latest_decl_id(),
                       existing_interface.has_definition_started()),
            /*prev_import_ir_id=*/SemIR::ImportIRId::None);

        // Can't merge interface definitions due to the generic requirements.
        if (!is_definition || !existing_interface.has_definition_started()) {
          // This is a redeclaration of an existing interface.
          interface_decl.interface_id = existing_interface_decl->interface_id;
          interface_decl.type_id = existing_interface_decl->type_id;
          // TODO: If the new declaration is a definition, keep its parameter
          // and implicit parameter lists rather than the ones from the
          // previous declaration.
        }
      }
    } else {
      // This is a redeclaration of something other than a interface.
      DiagnoseDuplicateName(context, name_context.name_id, name_context.loc_id,
                            existing_id);
    }
  }

  // Create a new interface if this isn't a valid redeclaration.
  if (!interface_decl.interface_id.has_value()) {
    // TODO: If this is an invalid redeclaration of a non-interface entity or
    // there was an error in the qualifier, we will have lost track of the
    // interface name here. We should keep track of it even if the name is
    // invalid.
    interface_info.generic_id = BuildGenericDecl(context, interface_decl_id);
    interface_decl.interface_id = context.interfaces().Add(interface_info);
    if (interface_info.has_parameters()) {
      interface_decl.type_id =
          GetGenericInterfaceType(context, interface_decl.interface_id,
                                  context.scope_stack().PeekSpecificId());
    }
  } else {
    FinishGenericRedecl(
        context, interface_decl_id,
        context.interfaces().Get(interface_decl.interface_id).generic_id);
  }

  // Write the interface ID into the InterfaceDecl.
  ReplaceInstBeforeConstantUse(context, interface_decl_id, interface_decl);

  return {interface_decl.interface_id, interface_decl_id};
}

auto HandleParseNode(Context& context, Parse::InterfaceDeclId node_id) -> bool {
  BuildInterfaceDecl(context, node_id, /*is_definition=*/false);
  context.decl_name_stack().PopScope();
  return true;
}

auto HandleParseNode(Context& context,
                     Parse::InterfaceDefinitionStartId node_id) -> bool {
  auto [interface_id, interface_decl_id] =
      BuildInterfaceDecl(context, node_id, /*is_definition=*/true);
  auto& interface_info = context.interfaces().Get(interface_id);

  // Track that this declaration is the definition.
  CARBON_CHECK(!interface_info.has_definition_started(),
               "Can't merge with defined interfaces.");
  interface_info.definition_id = interface_decl_id;
  interface_info.scope_id = context.name_scopes().Add(
      interface_decl_id, SemIR::NameId::None, interface_info.parent_scope_id);
  context.name_scopes()
      .Get(interface_info.scope_id)
      .set_is_interface_definition();

  auto self_specific_id =
      context.generics().GetSelfSpecific(interface_info.generic_id);

  StartGenericDefinition(context);

  context.inst_block_stack().Push();
  context.node_stack().Push(node_id, interface_id);

  // We use the arg stack to build the witness table type.
  context.args_type_info_stack().Push();

  // Declare and introduce `Self`.
  SemIR::FacetType facet_type =
      FacetTypeFromInterface(context, interface_id, self_specific_id);
  SemIR::TypeId self_type_id = context.types().GetTypeIdForTypeConstantId(
      TryEvalInst(context, SemIR::InstId::None, facet_type));

  // We model `Self` as a symbolic binding whose type is the interface.
  // Because there is no equivalent non-symbolic value, we use `None` as
  // the `value_id` on the `BindSymbolicName`.
  auto entity_name_id = context.entity_names().AddSymbolicBindingName(
      SemIR::NameId::SelfType, interface_info.scope_id,
      context.scope_stack().AddCompileTimeBinding(),
      /*is_template=*/false);
  interface_info.self_param_id =
      AddInst(context, SemIR::LocIdAndInst::NoLoc<SemIR::BindSymbolicName>(
                           {.type_id = self_type_id,
                            .entity_name_id = entity_name_id,
                            .value_id = SemIR::InstId::None}));
  context.scope_stack().PushCompileTimeBinding(interface_info.self_param_id);
  context.name_scopes().AddRequiredName(interface_info.scope_id,
                                        SemIR::NameId::SelfType,
                                        interface_info.self_param_id);

  // Enter the interface scope.
  context.scope_stack().Push(interface_decl_id, interface_info.scope_id,
                             self_specific_id);

  // TODO: Handle the case where there's control flow in the interface body. For
  // example:
  //
  //   interface C {
  //     let v: if true then i32 else f64;
  //   }
  //
  // We may need to track a list of instruction blocks here, as we do for a
  // function.
  interface_info.body_block_id = context.inst_block_stack().PeekOrAdd();
  return true;
}

auto HandleParseNode(Context& context, Parse::InterfaceDefinitionId /*node_id*/)
    -> bool {
  auto interface_id =
      context.node_stack().Pop<Parse::NodeKind::InterfaceDefinitionStart>();
  context.inst_block_stack().Pop();
  auto associated_entities_id = context.args_type_info_stack().Pop();

  // The interface type is now fully defined.
  auto& interface_info = context.interfaces().Get(interface_id);
  if (!interface_info.associated_entities_id.has_value()) {
    interface_info.associated_entities_id = associated_entities_id;
  }

  FinishGenericDefinition(context, interface_info.generic_id);

  // The decl_name_stack and scopes are popped by `ProcessNodeIds`.
  return true;
}

}  // namespace Carbon::Check

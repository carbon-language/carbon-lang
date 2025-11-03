// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <tuple>

#include "toolchain/check/context.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/facet_type.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/interface.h"
#include "toolchain/check/merge.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/check/name_component.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/type.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/interface.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::InterfaceIntroducerId node_id)
    -> bool {
  // This interface is potentially generic.
  StartGenericDecl(context);
  // Create an instruction block to hold the instructions created as part of the
  // interface signature, such as generic parameters.
  context.inst_block_stack().Push();
  // Optional modifiers and the name follow.
  context.decl_introducer_state_stack().Push<Lex::TokenKind::Interface>();
  context.decl_name_stack().PushScopeAndStartName();

  // Push the bracketing node.
  context.node_stack().Push(node_id);
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
  auto interface_decl = SemIR::InterfaceDecl{
      SemIR::TypeType::TypeId, SemIR::InterfaceId::None, decl_block_id};
  auto decl_inst_id = AddPlaceholderInst(context, node_id, interface_decl);

  SemIR::Interface interface_info = {name_context.MakeEntityWithParamsBase(
      name, decl_inst_id, /*is_extern=*/false, SemIR::LibraryNameId::None)};

  DiagnoseIfGenericMissingExplicitParameters(context, interface_info);

  // Check whether this is a redeclaration.
  SemIR::ScopeLookupResult lookup_result =
      context.decl_name_stack().LookupOrAddName(
          name_context, decl_inst_id, introducer.modifier_set.GetAccessKind());
  if (auto existing_decl = TryGetExistingDecl(context, name, lookup_result,
                                              interface_info, is_definition)) {
    auto existing_interface_decl = existing_decl->As<SemIR::InterfaceDecl>();
    interface_decl.interface_id = existing_interface_decl.interface_id;
    interface_decl.type_id = existing_interface_decl.type_id;
    // TODO: If the new declaration is a definition, keep its parameter
    // and implicit parameter lists rather than the ones from the
    // previous declaration.

    auto prev_decl_generic_id =
        context.interfaces().Get(interface_decl.interface_id).generic_id;
    FinishGenericRedecl(context, prev_decl_generic_id);
  } else {
    // Create a new interface if this isn't a valid redeclaration.
    interface_info.generic_id = BuildGenericDecl(context, decl_inst_id);
    interface_decl.interface_id = context.interfaces().Add(interface_info);
    if (interface_info.has_parameters()) {
      interface_decl.type_id =
          GetGenericInterfaceType(context, interface_decl.interface_id,
                                  context.scope_stack().PeekSpecificId());
    }
  }

  // Write the interface ID into the InterfaceDecl.
  ReplaceInstBeforeConstantUse(context, decl_inst_id, interface_decl);

  return {interface_decl.interface_id, decl_inst_id};
}

auto HandleParseNode(Context& context, Parse::InterfaceDeclId node_id) -> bool {
  BuildInterfaceDecl(context, node_id, /*is_definition=*/false);
  context.decl_name_stack().PopScope();
  return true;
}

auto HandleParseNode(Context& context,
                     Parse::InterfaceDefinitionStartId node_id) -> bool {
  auto [interface_id, decl_inst_id] =
      BuildInterfaceDecl(context, node_id, /*is_definition=*/true);
  auto& interface_info = context.interfaces().Get(interface_id);

  // Track that this declaration is the definition.
  CARBON_CHECK(!interface_info.has_definition_started(),
               "Can't merge with defined interfaces.");
  interface_info.definition_id = decl_inst_id;
  interface_info.scope_id = context.name_scopes().Add(
      decl_inst_id, SemIR::NameId::None, interface_info.parent_scope_id);
  context.name_scopes()
      .Get(interface_info.scope_id)
      .set_is_interface_definition();

  auto self_specific_id =
      context.generics().GetSelfSpecific(interface_info.generic_id);

  StartGenericDefinition(context, interface_info.generic_id);

  context.inst_block_stack().Push();

  // We use the arg stack to build the witness table type.
  context.args_type_info_stack().Push();

  // Declare and introduce `Self`. We model `Self` as a symbolic binding whose
  // type is the interface, excluding any other interfaces mentioned by
  // `require` declarations.
  SemIR::TypeId self_type_id =
      GetInterfaceType(context, interface_id, self_specific_id);
  interface_info.self_param_id =
      AddSelfGenericParameter(context, node_id, self_type_id,
                              interface_info.scope_id, /*is_template=*/false);

  // Enter the interface scope.
  context.scope_stack().PushForEntity(decl_inst_id, interface_info.scope_id,
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

  context.node_stack().Push(node_id, interface_id);
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

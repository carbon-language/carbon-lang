// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/modifiers.h"

namespace Carbon::Check {

auto HandleInterfaceIntroducer(Context& context, Parse::NodeId parse_node)
    -> bool {
  // Create an instruction block to hold the instructions created as part of the
  // interface signature, such as generic parameters.
  context.inst_block_stack().Push();
  // Push the bracketing node.
  context.node_stack().Push(parse_node);
  // Optional modifiers and the name follow.
  context.decl_state_stack().Push(DeclState::Interface, parse_node);
  context.decl_name_stack().PushScopeAndStartName();
  return true;
}

static auto BuildInterfaceDecl(Context& context)
    -> std::tuple<SemIR::InterfaceId, SemIR::InstId> {
  auto name_context = context.decl_name_stack().FinishName();
  context.node_stack()
      .PopAndDiscardSoloParseNode<Parse::NodeKind::InterfaceIntroducer>();
  auto first_node = context.decl_state_stack().innermost().first_node;

  // Process modifiers.
  CheckAccessModifiersOnDecl(context, Lex::TokenKind::Interface);
  LimitModifiersOnDecl(context, KeywordModifierSet::Access,
                       Lex::TokenKind::Interface);

  auto modifiers = context.decl_state_stack().innermost().modifier_set;
  if (!!(modifiers & KeywordModifierSet::Access)) {
    context.TODO(context.decl_state_stack().innermost().saw_access_modifier,
                 "access modifier");
  }
  context.decl_state_stack().Pop(DeclState::Interface);

  auto decl_block_id = context.inst_block_stack().Pop();

  // Add the interface declaration.
  auto interface_decl = SemIR::InterfaceDecl{
      first_node, SemIR::InterfaceId::Invalid, decl_block_id};
  auto interface_decl_id = context.AddInst(interface_decl);

  // Check whether this is a redeclaration.
  auto existing_id = context.decl_name_stack().LookupOrAddName(
      name_context, interface_decl_id);
  if (existing_id.is_valid()) {
    if (auto existing_interface_decl =
            context.insts().Get(existing_id).TryAs<SemIR::InterfaceDecl>()) {
      // This is a redeclaration of an existing interface.
      interface_decl.interface_id = existing_interface_decl->interface_id;

      // TODO: Check that the generic parameter list agrees with the prior
      // declaration.
    } else {
      // This is a redeclaration of something other than a interface.
      context.DiagnoseDuplicateName(name_context.parse_node, existing_id);
    }
  }

  // Create a new interface if this isn't a valid redeclaration.
  if (!interface_decl.interface_id.is_valid()) {
    // TODO: If this is an invalid redeclaration of a non-interface entity or
    // there was an error in the qualifier, we will have lost track of the
    // interface name here. We should keep track of it even if the name is
    // invalid.
    // TODO: should have a `Self` type id member
    interface_decl.interface_id = context.interfaces().Add(
        {.name_id =
             name_context.state == DeclNameStack::NameContext::State::Unresolved
                 ? name_context.unresolved_name_id
                 : SemIR::NameId::Invalid,
         .decl_id = interface_decl_id});
  }

  // Write the interface ID into the InterfaceDecl.
  context.insts().Set(interface_decl_id, interface_decl);

  return {interface_decl.interface_id, interface_decl_id};
}

auto HandleInterfaceDecl(Context& context, Parse::NodeId /*parse_node*/)
    -> bool {
  BuildInterfaceDecl(context);
  context.decl_name_stack().PopScope();
  return true;
}

auto HandleInterfaceDefinitionStart(Context& context, Parse::NodeId parse_node)
    -> bool {
  auto [interface_id, interface_decl_id] = BuildInterfaceDecl(context);
  auto& interface_info = context.interfaces().Get(interface_id);

  // Track that this declaration is the definition.
  if (interface_info.definition_id.is_valid()) {
    CARBON_DIAGNOSTIC(InterfaceRedefinition, Error,
                      "Redefinition of interface {0}.", std::string);
    CARBON_DIAGNOSTIC(InterfacePreviousDefinition, Note,
                      "Previous definition was here.");
    context.emitter()
        .Build(parse_node, InterfaceRedefinition,
               context.names().GetFormatted(interface_info.name_id).str())
        .Note(context.insts().Get(interface_info.definition_id).parse_node(),
              InterfacePreviousDefinition)
        .Emit();
  } else {
    interface_info.definition_id = interface_decl_id;
    interface_info.scope_id = context.name_scopes().Add();
  }

  // Enter the interface scope.
  context.PushScope(interface_decl_id, interface_info.scope_id);

  // TODO: Introduce `Self`.

  context.inst_block_stack().Push();
  context.node_stack().Push(parse_node, interface_id);
  // TODO: Perhaps use the args_type_info_stack for a witness table.

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

auto HandleInterfaceDefinition(Context& context, Parse::NodeId /*parse_node*/)
    -> bool {
  auto interface_id =
      context.node_stack().Pop<Parse::NodeKind::InterfaceDefinitionStart>();
  context.inst_block_stack().Pop();
  context.PopScope();
  context.decl_name_stack().PopScope();

  // The interface type is now fully defined.
  auto& interface_info = context.interfaces().Get(interface_id);
  interface_info.defined = true;
  return true;
}

}  // namespace Carbon::Check

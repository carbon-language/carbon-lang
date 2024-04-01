// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/interface.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto HandleLetIntroducer(Context& context, Parse::LetIntroducerId node_id)
    -> bool {
  context.decl_state_stack().Push(DeclState::Let);
  // Push a bracketing node to establish the pattern context.
  context.node_stack().Push(node_id);
  return true;
}

auto HandleLetInitializer(Context& context, Parse::LetInitializerId node_id)
    -> bool {
  context.node_stack().Push(node_id);
  return true;
}

static auto BuildAssociatedConstantDecl(
    Context& context, Parse::LetDeclId node_id, SemIR::InstId pattern_id,
    SemIR::LocIdAndInst pattern, SemIR::InterfaceId interface_id) -> void {
  auto& interface_info = context.interfaces().Get(interface_id);

  auto binding_pattern = pattern.inst.TryAs<SemIR::BindSymbolicName>();
  if (!binding_pattern) {
    CARBON_DIAGNOSTIC(ExpectedSymbolicBindingInAssociatedConstant, Error,
                      "Pattern in associated constant declaration must be a "
                      "single `:!` binding.");
    context.emitter().Emit(pattern.loc_id,
                           ExpectedSymbolicBindingInAssociatedConstant);
    context.name_scopes().Get(interface_info.scope_id).has_error = true;
    return;
  }

  // Replace the tentative BindName instruction with the associated constant
  // declaration.
  auto name_id =
      context.bind_names().Get(binding_pattern->bind_name_id).name_id;
  context.ReplaceLocIdAndInstBeforeConstantUse(
      pattern_id, {node_id, SemIR::AssociatedConstantDecl{
                                binding_pattern->type_id, name_id}});
  auto decl_id = pattern_id;
  context.inst_block_stack().AddInstId(decl_id);

  // Add an associated entity name to the interface scope.
  auto assoc_id = BuildAssociatedEntity(context, interface_id, decl_id);
  auto name_context =
      context.decl_name_stack().MakeUnqualifiedName(pattern.loc_id, name_id);
  context.decl_name_stack().AddNameToLookup(name_context, assoc_id);
}

auto HandleLetDecl(Context& context, Parse::LetDeclId node_id) -> bool {
  // Pop the optional initializer.
  std::optional<SemIR::InstId> value_id;
  if (context.node_stack().PeekNextIs<Parse::NodeKind::LetInitializer>()) {
    value_id = context.node_stack().PopExpr();
    context.node_stack()
        .PopAndDiscardSoloNodeId<Parse::NodeKind::LetInitializer>();
  }

  if (context.node_stack().PeekIs<Parse::NodeKind::TuplePattern>()) {
    return context.TODO(node_id, "tuple pattern in let");
  }
  SemIR::InstId pattern_id = context.node_stack().PopPattern();
  context.node_stack()
      .PopAndDiscardSoloNodeId<Parse::NodeKind::LetIntroducer>();
  // Process declaration modifiers.
  // TODO: For a qualified `let` declaration, this should use the target scope
  // of the name introduced in the declaration. See #2590.
  CheckAccessModifiersOnDecl(context, Lex::TokenKind::Let,
                             context.scope_stack().PeekNameScopeId());
  RequireDefaultFinalOnlyInInterfaces(context, Lex::TokenKind::Let,
                                      context.scope_stack().PeekNameScopeId());
  LimitModifiersOnDecl(
      context, KeywordModifierSet::Access | KeywordModifierSet::Interface,
      Lex::TokenKind::Let);

  auto modifiers = context.decl_state_stack().innermost().modifier_set;
  if (!!(modifiers & KeywordModifierSet::Access)) {
    context.TODO(context.decl_state_stack().innermost().modifier_node_id(
                     ModifierOrder::Access),
                 "access modifier");
  }
  if (!!(modifiers & KeywordModifierSet::Interface)) {
    context.TODO(context.decl_state_stack().innermost().modifier_node_id(
                     ModifierOrder::Decl),
                 "interface modifier");
  }
  context.decl_state_stack().Pop(DeclState::Let);

  auto pattern = context.insts().GetWithLocId(pattern_id);
  auto interface_scope = context.GetCurrentScopeAs<SemIR::InterfaceDecl>();

  if (value_id) {
    // Convert the value to match the type of the pattern.
    value_id = ConvertToValueOfType(context, node_id, *value_id,
                                    pattern.inst.type_id());
  }

  // At interface scope, we are forming an associated constant, which has
  // different rules.
  if (interface_scope) {
    BuildAssociatedConstantDecl(context, node_id, pattern_id, pattern,
                                interface_scope->interface_id);
    return true;
  }

  if (!value_id) {
    CARBON_DIAGNOSTIC(
        ExpectedInitializerAfterLet, Error,
        "Expected `=`; `let` declaration must have an initializer.");
    context.emitter().Emit(TokenOnly(node_id), ExpectedInitializerAfterLet);
    value_id = SemIR::InstId::BuiltinError;
  }

  // Update the binding with its value and add it to the current block, after
  // the computation of the value.
  // TODO: Support other kinds of pattern here.
  auto bind_name = pattern.inst.As<SemIR::AnyBindName>();
  CARBON_CHECK(!bind_name.value_id.is_valid())
      << "Binding should not already have a value!";
  bind_name.value_id = *value_id;
  context.ReplaceInstBeforeConstantUse(pattern_id, bind_name);
  context.inst_block_stack().AddInstId(pattern_id);

  // Add the name of the binding to the current scope.
  auto name_id = context.bind_names().Get(bind_name.bind_name_id).name_id;
  context.AddNameToLookup(name_id, pattern_id);
  if (context.scope_stack().PeekNameScopeId() == SemIR::NameScopeId::Package) {
    context.AddExport(pattern_id);
  }
  return true;
}

}  // namespace Carbon::Check

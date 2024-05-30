// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/modifiers.h"

#include "toolchain/check/decl_state.h"

namespace Carbon::Check {

static auto DiagnoseNotAllowed(Context& context, Parse::NodeId modifier_node,
                               Lex::TokenKind decl_kind,
                               llvm::StringRef context_string,
                               SemIR::LocId context_loc_id) -> void {
  CARBON_DIAGNOSTIC(ModifierNotAllowedOn, Error,
                    "`{0}` not allowed on `{1}` declaration{2}.",
                    Lex::TokenKind, Lex::TokenKind, std::string);
  auto diag = context.emitter().Build(modifier_node, ModifierNotAllowedOn,
                                      context.token_kind(modifier_node),
                                      decl_kind, context_string.str());
  if (context_loc_id.is_valid()) {
    CARBON_DIAGNOSTIC(ModifierNotInContext, Note,
                      "Containing definition here.");
    diag.Note(context_loc_id, ModifierNotInContext);
  }
  diag.Emit();
}

// Returns the KeywordModifierSet corresponding to the ModifierOrder entry.
static auto ModifierOrderAsSet(ModifierOrder order) -> KeywordModifierSet {
  switch (order) {
    case ModifierOrder::Access:
      return KeywordModifierSet::Access;
    case ModifierOrder::Extern:
      return KeywordModifierSet::Extern;
    case ModifierOrder::Decl:
      return KeywordModifierSet::Decl;
  }
}

auto ForbidModifiersOnDecl(Context& context, KeywordModifierSet forbidden,
                           Lex::TokenKind decl_kind,
                           llvm::StringRef context_string,
                           SemIR::LocId context_loc_id) -> void {
  auto& s = context.decl_state_stack().innermost();
  auto not_allowed = s.modifier_set & forbidden;
  if (not_allowed.empty()) {
    return;
  }

  for (auto order_index = 0;
       order_index <= static_cast<int8_t>(ModifierOrder::Last); ++order_index) {
    auto order = static_cast<ModifierOrder>(order_index);
    if (not_allowed.HasAnyOf(ModifierOrderAsSet(order))) {
      DiagnoseNotAllowed(context, s.modifier_node_id(order), decl_kind,
                         context_string, context_loc_id);
      s.set_modifier_node_id(order, Parse::NodeId::Invalid);
    }
  }

  s.modifier_set.Remove(forbidden);
}

auto CheckAccessModifiersOnDecl(Context& context, Lex::TokenKind decl_kind,
                                std::optional<SemIR::Inst> enclosing_scope_inst)
    -> void {
  if (enclosing_scope_inst) {
    if (enclosing_scope_inst->Is<SemIR::Namespace>()) {
      // TODO: This assumes that namespaces can only be declared at file scope.
      // If we add support for non-file-scope namespaces, we will need to check
      // the parents of the target scope to determine whether we're at file
      // scope.
      ForbidModifiersOnDecl(
          context, KeywordModifierSet::Protected, decl_kind,
          " at file scope, `protected` is only allowed on class members");
      return;
    }

    if (enclosing_scope_inst->Is<SemIR::ClassDecl>()) {
      // Both `private` and `protected` allowed in a class definition.
      return;
    }
  }

  // Otherwise neither `private` nor `protected` allowed.
  ForbidModifiersOnDecl(context, KeywordModifierSet::Protected, decl_kind,
                        ", `protected` is only allowed on class members");
  ForbidModifiersOnDecl(
      context, KeywordModifierSet::Private, decl_kind,
      ", `private` is only allowed on class members and at file scope");
}

auto CheckMethodModifiersOnFunction(
    Context& context, SemIR::InstId enclosing_scope_inst_id,
    std::optional<SemIR::Inst> enclosing_scope_inst) -> void {
  const Lex::TokenKind decl_kind = Lex::TokenKind::Fn;
  if (enclosing_scope_inst) {
    if (auto class_decl = enclosing_scope_inst->TryAs<SemIR::ClassDecl>()) {
      auto inheritance_kind =
          context.classes().Get(class_decl->class_id).inheritance_kind;
      if (inheritance_kind == SemIR::Class::Final) {
        ForbidModifiersOnDecl(
            context, KeywordModifierSet::Virtual, decl_kind,
            " in a non-abstract non-base `class` definition",
            context.insts().GetLocId(enclosing_scope_inst_id));
      }
      if (inheritance_kind != SemIR::Class::Abstract) {
        ForbidModifiersOnDecl(
            context, KeywordModifierSet::Abstract, decl_kind,
            " in a non-abstract `class` definition",
            context.insts().GetLocId(enclosing_scope_inst_id));
      }
      return;
    }
  }

  ForbidModifiersOnDecl(context, KeywordModifierSet::Method, decl_kind,
                        " outside of a class");
}

auto RestrictExternModifierOnDecl(
    Context& context, Lex::TokenKind decl_kind,
    std::optional<SemIR::Inst> enclosing_scope_inst, bool is_definition)
    -> void {
  if (is_definition) {
    ForbidModifiersOnDecl(context, KeywordModifierSet::Extern, decl_kind,
                          " that provides a definition");
  }
  if (enclosing_scope_inst && !enclosing_scope_inst->Is<SemIR::Namespace>()) {
    ForbidModifiersOnDecl(context, KeywordModifierSet::Extern, decl_kind,
                          " that is a member");
  }
}

auto RequireDefaultFinalOnlyInInterfaces(
    Context& context, Lex::TokenKind decl_kind,
    std::optional<SemIR::Inst> enclosing_scope_inst) -> void {
  if (enclosing_scope_inst &&
      enclosing_scope_inst->Is<SemIR::InterfaceDecl>()) {
    // Both `default` and `final` allowed in an interface definition.
    return;
  }
  ForbidModifiersOnDecl(context, KeywordModifierSet::Interface, decl_kind,
                        " outside of an interface");
}

}  // namespace Carbon::Check

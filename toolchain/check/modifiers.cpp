// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/modifiers.h"

#include "toolchain/check/decl_introducer_state.h"

namespace Carbon::Check {

// Returns the TokenKind for a DeclKind.
static auto DeclKindToken(DeclIntroducerState::DeclKind decl_kind)
    -> Lex::TokenKind {
  switch (decl_kind) {
    case DeclIntroducerState::Adapt:
      return Lex::TokenKind::Adapt;
    case DeclIntroducerState::Alias:
      return Lex::TokenKind::Alias;
    case DeclIntroducerState::Base:
      return Lex::TokenKind::Base;
    case DeclIntroducerState::Class:
      return Lex::TokenKind::Class;
    case DeclIntroducerState::Constraint:
      return Lex::TokenKind::Constraint;
    case DeclIntroducerState::Export:
      return Lex::TokenKind::Export;
    case DeclIntroducerState::Fn:
      return Lex::TokenKind::Fn;
    case DeclIntroducerState::Import:
      return Lex::TokenKind::Import;
    case DeclIntroducerState::Impl:
      return Lex::TokenKind::Impl;
    case DeclIntroducerState::Interface:
      return Lex::TokenKind::Interface;
    case DeclIntroducerState::Let:
      return Lex::TokenKind::Let;
    case DeclIntroducerState::Library:
      return Lex::TokenKind::Library;
    case DeclIntroducerState::Namespace:
      return Lex::TokenKind::Namespace;
    case DeclIntroducerState::Package:
      return Lex::TokenKind::Package;
    case DeclIntroducerState::Var:
      return Lex::TokenKind::Var;
  }
}

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

auto ForbidModifiersOnDecl(Context& context, DeclIntroducerState& introducer,
                           KeywordModifierSet forbidden,
                           llvm::StringRef context_string,
                           SemIR::LocId context_loc_id) -> void {
  auto not_allowed = introducer.modifier_set & forbidden;
  if (not_allowed.empty()) {
    return;
  }

  for (auto order_index = 0;
       order_index <= static_cast<int8_t>(ModifierOrder::Last); ++order_index) {
    auto order = static_cast<ModifierOrder>(order_index);
    if (not_allowed.HasAnyOf(ModifierOrderAsSet(order))) {
      DiagnoseNotAllowed(context, introducer.modifier_node_id(order),
                         DeclKindToken(introducer.kind), context_string,
                         context_loc_id);
      introducer.set_modifier_node_id(order, Parse::NodeId::Invalid);
    }
  }

  introducer.modifier_set.Remove(forbidden);
}

auto CheckAccessModifiersOnDecl(Context& context,
                                DeclIntroducerState& introducer,
                                std::optional<SemIR::Inst> parent_scope_inst)
    -> void {
  if (parent_scope_inst) {
    if (parent_scope_inst->Is<SemIR::Namespace>()) {
      // TODO: This assumes that namespaces can only be declared at file scope.
      // If we add support for non-file-scope namespaces, we will need to check
      // the parents of the target scope to determine whether we're at file
      // scope.
      ForbidModifiersOnDecl(
          context, introducer, KeywordModifierSet::Protected,
          " at file scope, `protected` is only allowed on class members");
      return;
    }

    if (parent_scope_inst->Is<SemIR::ClassDecl>()) {
      // Both `private` and `protected` allowed in a class definition.
      return;
    }
  }

  // Otherwise neither `private` nor `protected` allowed.
  ForbidModifiersOnDecl(context, introducer, KeywordModifierSet::Protected,
                        ", `protected` is only allowed on class members");
  ForbidModifiersOnDecl(
      context, introducer, KeywordModifierSet::Private,
      ", `private` is only allowed on class members and at file scope");
}

auto CheckMethodModifiersOnFunction(
    Context& context, DeclIntroducerState& introducer,
    SemIR::InstId parent_scope_inst_id,
    std::optional<SemIR::Inst> parent_scope_inst) -> void {
  if (parent_scope_inst) {
    if (auto class_decl = parent_scope_inst->TryAs<SemIR::ClassDecl>()) {
      auto inheritance_kind =
          context.classes().Get(class_decl->class_id).inheritance_kind;
      if (inheritance_kind == SemIR::Class::Final) {
        ForbidModifiersOnDecl(context, introducer, KeywordModifierSet::Virtual,
                              " in a non-abstract non-base `class` definition",
                              context.insts().GetLocId(parent_scope_inst_id));
      }
      if (inheritance_kind != SemIR::Class::Abstract) {
        ForbidModifiersOnDecl(context, introducer, KeywordModifierSet::Abstract,
                              " in a non-abstract `class` definition",
                              context.insts().GetLocId(parent_scope_inst_id));
      }
      return;
    }
  }

  ForbidModifiersOnDecl(context, introducer, KeywordModifierSet::Method,
                        " outside of a class");
}

auto RestrictExternModifierOnDecl(Context& context,
                                  DeclIntroducerState& introducer,
                                  std::optional<SemIR::Inst> parent_scope_inst,
                                  bool is_definition) -> void {
  if (is_definition) {
    ForbidModifiersOnDecl(context, introducer, KeywordModifierSet::Extern,
                          " that provides a definition");
  }
  if (parent_scope_inst && !parent_scope_inst->Is<SemIR::Namespace>()) {
    ForbidModifiersOnDecl(context, introducer, KeywordModifierSet::Extern,
                          " that is a member");
  }
}

auto RequireDefaultFinalOnlyInInterfaces(
    Context& context, DeclIntroducerState& introducer,
    std::optional<SemIR::Inst> parent_scope_inst) -> void {
  if (parent_scope_inst && parent_scope_inst->Is<SemIR::InterfaceDecl>()) {
    // Both `default` and `final` allowed in an interface definition.
    return;
  }
  ForbidModifiersOnDecl(context, introducer, KeywordModifierSet::Interface,
                        " outside of an interface");
}

}  // namespace Carbon::Check

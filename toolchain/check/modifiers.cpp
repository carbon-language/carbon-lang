// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/modifiers.h"

#include "toolchain/check/decl_introducer_state.h"

namespace Carbon::Check {

// Builds the diagnostic for DiagnoseNotAllowed.
template <typename... TokenKinds>
static auto StartDiagnoseNotAllowed(
    Context& context,
    const Diagnostics::DiagnosticBase<TokenKinds...>& diagnostic_base,
    Parse::NodeId modifier_node, Lex::TokenKind declaration_kind)
    -> DiagnosticBuilder {
  if constexpr (sizeof...(TokenKinds) == 0) {
    return context.emitter().Build(modifier_node, diagnostic_base);
  } else if constexpr (sizeof...(TokenKinds) == 1) {
    return context.emitter().Build(modifier_node, diagnostic_base,
                                   context.token_kind(modifier_node));
  } else {
    static_assert(sizeof...(TokenKinds) == 2);
    return context.emitter().Build(modifier_node, diagnostic_base,
                                   context.token_kind(modifier_node),
                                   declaration_kind);
  }
}

// Diagnoses that a modifier wasn't allowed. Handles adding context when
// possible.
//
// The diagnostic can take up to two TokenKinds: the modifier kind, and the
// declaration kind.
template <typename... TokenKinds>
static auto DiagnoseNotAllowed(
    Context& context,
    const Diagnostics::DiagnosticBase<TokenKinds...>& diagnostic_base,
    Parse::NodeId modifier_node, Lex::TokenKind decl_kind,
    SemIR::LocId context_loc_id) -> void {
  auto diag = StartDiagnoseNotAllowed(context, diagnostic_base, modifier_node,
                                      decl_kind);
  if (context_loc_id.has_value()) {
    CARBON_DIAGNOSTIC(ModifierNotInContext, Note, "containing definition here");
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

// Like `LimitModifiersOnDecl`, except says which modifiers are forbidden, and a
// `context_string` (and optional `context_loc_id`) specifying the context in
// which those modifiers are forbidden.
//
// See DiagnoseNotAllowed for details regarding diagnostic_base.
template <typename DiagnosticBaseT>
static auto ForbidModifiersOnDecl(
    Context& context, const DiagnosticBaseT& diagnostic_base,
    DeclIntroducerState& introducer, KeywordModifierSet forbidden,
    SemIR::LocId context_loc_id = SemIR::LocId::None) -> void {
  auto not_allowed = introducer.modifier_set & forbidden;
  if (not_allowed.empty()) {
    return;
  }

  for (auto order_index = 0;
       order_index <= static_cast<int8_t>(ModifierOrder::Last); ++order_index) {
    auto order = static_cast<ModifierOrder>(order_index);
    if (not_allowed.HasAnyOf(ModifierOrderAsSet(order))) {
      DiagnoseNotAllowed(context, diagnostic_base,
                         introducer.modifier_node_id(order), introducer.kind,
                         context_loc_id);
      introducer.set_modifier_node_id(order, Parse::NodeId::None);
    }
  }

  introducer.modifier_set.Remove(forbidden);
}

auto LimitModifiersOnDecl(Context& context, DeclIntroducerState& introducer,
                          KeywordModifierSet allowed) -> void {
  CARBON_DIAGNOSTIC(ModifierNotAllowedOnDeclaration, Error,
                    "`{0}` not allowed on `{1}` declaration", Lex::TokenKind,
                    Lex::TokenKind);
  ForbidModifiersOnDecl(context, ModifierNotAllowedOnDeclaration, introducer,
                        ~allowed);
}

auto LimitModifiersOnNotDefinition(Context& context,
                                   DeclIntroducerState& introducer,
                                   KeywordModifierSet allowed) -> void {
  CARBON_DIAGNOSTIC(
      ModifierOnlyAllowedOnDefinition, Error,
      "`{0}` not allowed on `{1}` forward declaration, only definition",
      Lex::TokenKind, Lex::TokenKind);
  ForbidModifiersOnDecl(context, ModifierOnlyAllowedOnDefinition, introducer,
                        ~allowed);
}

auto CheckAccessModifiersOnDecl(Context& context,
                                DeclIntroducerState& introducer,
                                std::optional<SemIR::Inst> parent_scope_inst)
    -> void {
  CARBON_DIAGNOSTIC(ModifierProtectedNotAllowed, Error,
                    "`protected` not allowed; requires class scope");
  if (parent_scope_inst) {
    if (parent_scope_inst->Is<SemIR::Namespace>()) {
      // TODO: This assumes that namespaces can only be declared at file scope.
      // If we add support for non-file-scope namespaces, we will need to check
      // the parents of the target scope to determine whether we're at file
      // scope.
      ForbidModifiersOnDecl(context, ModifierProtectedNotAllowed, introducer,
                            KeywordModifierSet::Protected);
      return;
    }

    if (parent_scope_inst->Is<SemIR::ClassDecl>()) {
      // Both `private` and `protected` allowed in a class definition.
      return;
    }
  }

  // Otherwise neither `private` nor `protected` allowed.
  ForbidModifiersOnDecl(context, ModifierProtectedNotAllowed, introducer,
                        KeywordModifierSet::Protected);

  CARBON_DIAGNOSTIC(ModifierPrivateNotAllowed, Error,
                    "`private` not allowed; requires class or file scope");
  ForbidModifiersOnDecl(context, ModifierPrivateNotAllowed, introducer,
                        KeywordModifierSet::Private);
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
        CARBON_DIAGNOSTIC(
            ModifierVirtualNotAllowed, Error,
            "`virtual` not allowed; requires `abstract` or `base` class scope");
        ForbidModifiersOnDecl(context, ModifierVirtualNotAllowed, introducer,
                              KeywordModifierSet::Virtual,
                              context.insts().GetLocId(parent_scope_inst_id));
      }
      if (inheritance_kind != SemIR::Class::Abstract) {
        CARBON_DIAGNOSTIC(
            ModifierAbstractNotAllowed, Error,
            "`abstract` not allowed; requires `abstract` class scope");
        ForbidModifiersOnDecl(context, ModifierAbstractNotAllowed, introducer,
                              KeywordModifierSet::Abstract,
                              context.insts().GetLocId(parent_scope_inst_id));
      }
      return;
    }
  }

  CARBON_DIAGNOSTIC(ModifierRequiresClass, Error,
                    "`{0}` not allowed; requires class scope", Lex::TokenKind);
  ForbidModifiersOnDecl(context, ModifierRequiresClass, introducer,
                        KeywordModifierSet::Method);
}

auto RestrictExternModifierOnDecl(Context& context,
                                  DeclIntroducerState& introducer,
                                  std::optional<SemIR::Inst> parent_scope_inst,
                                  bool is_definition) -> void {
  if (!introducer.modifier_set.HasAnyOf(KeywordModifierSet::Extern)) {
    return;
  }

  if (parent_scope_inst && !parent_scope_inst->Is<SemIR::Namespace>()) {
    CARBON_DIAGNOSTIC(ModifierExternNotAllowed, Error,
                      "`{0}` not allowed; requires file or namespace scope",
                      Lex::TokenKind);
    ForbidModifiersOnDecl(context, ModifierExternNotAllowed, introducer,
                          KeywordModifierSet::Extern);
    // Treat as unset.
    introducer.extern_library = SemIR::LibraryNameId::None;
    return;
  }

  if (introducer.extern_library == context.sem_ir().library_id()) {
    // This prints an error for `extern library`, but doesn't drop it because we
    // assume there is some other, correct value that we just don't know here.
    CARBON_DIAGNOSTIC(ExternLibraryIsCurrentLibrary, Error,
                      "`extern library` cannot specify the current library");
    context.emitter().Emit(introducer.modifier_node_id(ModifierOrder::Extern),
                           ExternLibraryIsCurrentLibrary);
    introducer.extern_library = SemIR::LibraryNameId::Error;
    // Right now this can produce both this and the below diagnostic.
  }

  if (is_definition && introducer.extern_library.has_value()) {
    CARBON_DIAGNOSTIC(ExternLibraryOnDefinition, Error,
                      "a library cannot be provided for an `extern` modifier "
                      "on a definition");
    context.emitter().Emit(introducer.modifier_node_id(ModifierOrder::Extern),
                           ExternLibraryOnDefinition);
  }
}

auto RequireDefaultFinalOnlyInInterfaces(
    Context& context, DeclIntroducerState& introducer,
    std::optional<SemIR::Inst> parent_scope_inst) -> void {
  if (parent_scope_inst && parent_scope_inst->Is<SemIR::InterfaceDecl>()) {
    // Both `default` and `final` allowed in an interface definition.
    return;
  }
  CARBON_DIAGNOSTIC(ModifierRequiresInterface, Error,
                    "`{0}` not allowed; requires interface scope",
                    Lex::TokenKind);
  ForbidModifiersOnDecl(context, ModifierRequiresInterface, introducer,
                        KeywordModifierSet::Interface);
}

}  // namespace Carbon::Check

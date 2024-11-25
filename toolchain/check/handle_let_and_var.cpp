// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/decl_introducer_state.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/interface.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

template <Lex::TokenKind::RawEnumType Kind>
static auto HandleIntroducer(Context& context, Parse::NodeId node_id) -> bool {
  context.decl_introducer_state_stack().Push<Kind>();
  // Push a bracketing node and pattern block to establish the pattern context.
  context.node_stack().Push(node_id);
  context.pattern_block_stack().Push();
  return true;
}

auto HandleParseNode(Context& context, Parse::LetIntroducerId node_id) -> bool {
  return HandleIntroducer<Lex::TokenKind::Let>(context, node_id);
}

auto HandleParseNode(Context& context, Parse::VariableIntroducerId node_id)
    -> bool {
  return HandleIntroducer<Lex::TokenKind::Var>(context, node_id);
}

auto HandleParseNode(Context& context, Parse::ReturnedModifierId node_id)
    -> bool {
  // This is pushed to be seen by HandleBindingPattern.
  context.node_stack().Push(node_id);
  return true;
}

static auto HandleInitializer(Context& context, Parse::NodeId node_id) -> bool {
  if (context.scope_stack().PeekIndex() == ScopeIndex::Package) {
    context.global_init().Resume();
  }
  context.node_stack().Push(node_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::LetInitializerId node_id)
    -> bool {
  return HandleInitializer(context, node_id);
}

auto HandleParseNode(Context& context, Parse::VariableInitializerId node_id)
    -> bool {
  return HandleInitializer(context, node_id);
}

// Builds an associated constant declaration for a `let`.
static auto BuildAssociatedConstantDecl(Context& context,
                                        Parse::LetDeclId node_id,
                                        SemIR::InstId pattern_id,
                                        SemIR::LocIdAndInst pattern,
                                        SemIR::InterfaceId interface_id,
                                        SemIR::AccessKind access_kind) -> void {
  auto& interface_info = context.interfaces().Get(interface_id);

  auto binding_pattern = pattern.inst.TryAs<SemIR::BindSymbolicName>();
  if (!binding_pattern) {
    CARBON_DIAGNOSTIC(ExpectedSymbolicBindingInAssociatedConstant, Error,
                      "pattern in associated constant declaration must be a "
                      "single `:!` binding");
    context.emitter().Emit(pattern.loc_id,
                           ExpectedSymbolicBindingInAssociatedConstant);
    context.name_scopes().Get(interface_info.scope_id).has_error = true;
    return;
  }

  // Replace the tentative BindName instruction with the associated constant
  // declaration.
  auto name_id =
      context.entity_names().Get(binding_pattern->entity_name_id).name_id;
  context.ReplaceLocIdAndInstBeforeConstantUse(
      pattern_id,
      SemIR::LocIdAndInst(node_id, SemIR::AssociatedConstantDecl{
                                       binding_pattern->type_id, name_id}));
  auto decl_id = pattern_id;
  context.inst_block_stack().AddInstId(decl_id);

  // Add an associated entity name to the interface scope.
  auto assoc_id = BuildAssociatedEntity(context, interface_id, decl_id);
  auto name_context =
      context.decl_name_stack().MakeUnqualifiedName(pattern.loc_id, name_id);
  context.decl_name_stack().AddNameOrDiagnoseDuplicate(name_context, assoc_id,
                                                       access_kind);
}

// Adds name bindings. Returns the resulting ID for the references.
static auto HandleNameBinding(Context& context, SemIR::InstId pattern_id,
                              SemIR::AccessKind access_kind) -> SemIR::InstId {
  // Extract the name binding.
  if (auto bind_name =
          context.insts().TryGetAs<SemIR::AnyBindName>(pattern_id)) {
    // Form a corresponding name in the current context, and bind the name to
    // the variable.
    auto name_context = context.decl_name_stack().MakeUnqualifiedName(
        context.insts().GetLocId(pattern_id),
        context.entity_names().Get(bind_name->entity_name_id).name_id);
    context.decl_name_stack().AddNameOrDiagnoseDuplicate(
        name_context, pattern_id, access_kind);
    return bind_name->value_id;
  } else if (auto field_decl =
                 context.insts().TryGetAs<SemIR::FieldDecl>(pattern_id)) {
    // Introduce the field name into the class.
    auto name_context = context.decl_name_stack().MakeUnqualifiedName(
        context.insts().GetLocId(pattern_id), field_decl->name_id);
    context.decl_name_stack().AddNameOrDiagnoseDuplicate(
        name_context, pattern_id, access_kind);
    return pattern_id;
  } else {
    // TODO: Handle other kinds of pattern.
    return pattern_id;
  }
}

namespace {
// State from HandleDecl, returned for type-specific handling.
struct DeclInfo {
  // The optional initializer.
  std::optional<SemIR::InstId> init_id = std::nullopt;
  SemIR::InstId pattern_id = SemIR::InstId::Invalid;
  std::optional<SemIR::Inst> parent_scope_inst = std::nullopt;
  DeclIntroducerState introducer = DeclIntroducerState();
};
}  // namespace

// Handles common logic for `let` and `var` declarations.
// TODO: There's still a lot of divergence here, including logic in
// handle_binding_pattern. These should really be better unified.
template <const Lex::TokenKind& IntroducerTokenKind,
          const Parse::NodeKind& IntroducerNodeKind,
          const Parse::NodeKind& InitializerNodeKind, typename NodeT>
static auto HandleDecl(Context& context, NodeT node_id)
    -> std::optional<DeclInfo> {
  std::optional<DeclInfo> decl_info = DeclInfo();

  // TODO: Update binding-pattern handling to use the pattern block even in
  // a let/var context, and then consume it here.
  context.pattern_block_stack().PopAndDiscard();

  // Handle the optional initializer.
  if (context.node_stack().PeekNextIs<InitializerNodeKind>()) {
    decl_info->init_id = context.node_stack().PopExpr();
    context.node_stack().PopAndDiscardSoloNodeId<InitializerNodeKind>();
  }

  if (context.node_stack().PeekIs<Parse::NodeKind::TuplePattern>()) {
    if (decl_info->init_id &&
        context.scope_stack().PeekIndex() == ScopeIndex::Package) {
      context.global_init().Suspend();
    }
    context.TODO(node_id, "tuple pattern in let/var");
    decl_info = std::nullopt;
    return decl_info;
  }

  decl_info->pattern_id = context.node_stack().PopPattern();

  if constexpr (IntroducerTokenKind == Lex::TokenKind::Var) {
    // Pop the `returned` modifier if present.
    context.node_stack()
        .PopAndDiscardSoloNodeIdIf<Parse::NodeKind::ReturnedModifier>();
  }

  context.node_stack().PopAndDiscardSoloNodeId<IntroducerNodeKind>();

  // Process declaration modifiers.
  // TODO: For a qualified `let` or `var` declaration, this should use the
  // target scope of the name introduced in the declaration. See #2590.
  decl_info->parent_scope_inst =
      context.name_scopes()
          .GetInstIfValid(context.scope_stack().PeekNameScopeId())
          .second;
  decl_info->introducer =
      context.decl_introducer_state_stack().Pop<IntroducerTokenKind>();
  CheckAccessModifiersOnDecl(context, decl_info->introducer,
                             decl_info->parent_scope_inst);

  return decl_info;
}

auto HandleParseNode(Context& context, Parse::LetDeclId node_id) -> bool {
  auto decl_info =
      HandleDecl<Lex::TokenKind::Let, Parse::NodeKind::LetIntroducer,
                 Parse::NodeKind::LetInitializer>(context, node_id);
  if (!decl_info) {
    return false;
  }

  RequireDefaultFinalOnlyInInterfaces(context, decl_info->introducer,
                                      decl_info->parent_scope_inst);
  LimitModifiersOnDecl(
      context, decl_info->introducer,
      KeywordModifierSet::Access | KeywordModifierSet::Interface);

  if (decl_info->introducer.modifier_set.HasAnyOf(
          KeywordModifierSet::Interface)) {
    context.TODO(decl_info->introducer.modifier_node_id(ModifierOrder::Decl),
                 "interface modifier");
  }

  auto pattern = context.insts().GetWithLocId(decl_info->pattern_id);

  if (decl_info->init_id) {
    // Convert the value to match the type of the pattern.
    decl_info->init_id = ConvertToValueOfType(
        context, node_id, *decl_info->init_id, pattern.inst.type_id());
  }

  auto interface_scope = context.GetCurrentScopeAs<SemIR::InterfaceDecl>();

  // At interface scope, we are forming an associated constant, which has
  // different rules.
  if (interface_scope) {
    BuildAssociatedConstantDecl(
        context, node_id, decl_info->pattern_id, pattern,
        interface_scope->interface_id,
        decl_info->introducer.modifier_set.GetAccessKind());
    return true;
  }

  if (!decl_info->init_id) {
    CARBON_DIAGNOSTIC(
        ExpectedInitializerAfterLet, Error,
        "expected `=`; `let` declaration must have an initializer");
    context.emitter().Emit(TokenOnly(node_id), ExpectedInitializerAfterLet);
  }

  // Update the binding with its value and add it to the current block, after
  // the computation of the value.
  // TODO: Support other kinds of pattern here.
  auto bind_name = pattern.inst.As<SemIR::AnyBindName>();
  CARBON_CHECK(!bind_name.value_id.is_valid(),
               "Binding should not already have a value!");
  bind_name.value_id = decl_info->init_id ? *decl_info->init_id
                                          : SemIR::InstId::BuiltinErrorInst;
  context.ReplaceInstBeforeConstantUse(decl_info->pattern_id, bind_name);
  context.inst_block_stack().AddInstId(decl_info->pattern_id);

  HandleNameBinding(context, decl_info->pattern_id,
                    decl_info->introducer.modifier_set.GetAccessKind());

  if (decl_info->init_id &&
      context.scope_stack().PeekIndex() == ScopeIndex::Package) {
    context.global_init().Suspend();
  }

  return true;
}

auto HandleParseNode(Context& context, Parse::VariableDeclId node_id) -> bool {
  auto decl_info =
      HandleDecl<Lex::TokenKind::Var, Parse::NodeKind::VariableIntroducer,
                 Parse::NodeKind::VariableInitializer>(context, node_id);
  if (!decl_info) {
    return false;
  }

  LimitModifiersOnDecl(context, decl_info->introducer,
                       KeywordModifierSet::Access);

  decl_info->pattern_id =
      HandleNameBinding(context, decl_info->pattern_id,
                        decl_info->introducer.modifier_set.GetAccessKind());

  // If there was an initializer, assign it to the storage.
  if (decl_info->init_id) {
    if (context.GetCurrentScopeAs<SemIR::ClassDecl>()) {
      // TODO: In a class scope, we should instead save the initializer
      // somewhere so that we can use it as a default.
      context.TODO(node_id, "Field initializer");
    } else {
      decl_info->init_id = Initialize(context, node_id, decl_info->pattern_id,
                                      *decl_info->init_id);
      // TODO: Consider using different instruction kinds for assignment
      // versus initialization.
      context.AddInst<SemIR::Assign>(node_id, {.lhs_id = decl_info->pattern_id,
                                               .rhs_id = *decl_info->init_id});
    }

    if (context.scope_stack().PeekIndex() == ScopeIndex::Package) {
      context.global_init().Suspend();
    }
  }

  return true;
}

}  // namespace Carbon::Check

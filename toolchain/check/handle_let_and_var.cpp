// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <optional>

#include "toolchain/check/call.h"
#include "toolchain/check/class.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/core_identifier.h"
#include "toolchain/check/decl_introducer_state.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/interface.h"
#include "toolchain/check/keyword_modifier_set.h"
#include "toolchain/check/member_access.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/pattern.h"
#include "toolchain/check/pattern_match.h"
#include "toolchain/check/type.h"
#include "toolchain/diagnostics/emitter.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/parse/typed_nodes.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/pattern.h"
#include "toolchain/sem_ir/type.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Handles the end of the declaration region of an associated constant. This is
// called at the `=` or the `;` of the declaration, whichever comes first.
static auto EndAssociatedConstantDeclRegion(Context& context,
                                            SemIR::InterfaceId interface_id)
    -> void {
  // Peek the pattern. For a valid associated constant, the corresponding
  // instruction will be an `AssociatedConstantDecl` instruction.
  auto decl_id = context.node_stack().PeekPattern();
  auto assoc_const_decl =
      context.insts().GetAs<SemIR::AssociatedConstantDecl>(decl_id);
  auto& assoc_const =
      context.associated_constants().Get(assoc_const_decl.assoc_const_id);

  // Build a corresponding associated entity and add it into scope.
  //
  // TODO: The instruction is added to the associated constant's decl block.
  // It probably should be in the interface-with-self body instead.
  auto assoc_id = BuildAssociatedEntity(context, interface_id, decl_id);
  auto name_context = context.decl_name_stack().MakeUnqualifiedName(
      context.node_stack().PeekNodeId(), assoc_const.name_id);
  auto access_kind = context.decl_introducer_state_stack()
                         .innermost()
                         .modifier_set.GetAccessKind();
  context.decl_name_stack().AddNameOrDiagnose(name_context, assoc_id,
                                              access_kind);
}

template <Lex::TokenKind::RawEnumType Kind>
static auto HandleIntroducer(Context& context, Parse::NodeId node_id) -> bool {
  context.decl_introducer_state_stack().Push<Kind>();
  // Push a bracketing node and pattern block to establish the pattern context.
  context.node_stack().Push(node_id);
  context.pattern_block_stack().Push();
  if (context.scope_stack().TryGetCurrentScopeAs<SemIR::ClassDecl>() &&
      Kind == Lex::TokenKind::Var) {
    context.full_pattern_stack().PushFieldDecl();
  } else {
    context.full_pattern_stack().PushNameBindingDecl();
  }
  BeginSubpattern(context);
  return true;
}

auto HandleParseNode(Context& context, Parse::LetIntroducerId node_id) -> bool {
  return HandleIntroducer<Lex::TokenKind::Let>(context, node_id);
}

auto HandleParseNode(Context& context,
                     Parse::AssociatedConstantIntroducerId node_id) -> bool {
  // Collect the declarations nested in the associated constant in a decl
  // block. This is popped by FinishAssociatedConstantDecl.
  context.inst_block_stack().Push();
  return HandleIntroducer<Lex::TokenKind::Let>(context, node_id);
}

auto HandleParseNode(Context& context, Parse::VariableIntroducerId node_id)
    -> bool {
  return HandleIntroducer<Lex::TokenKind::Var>(context, node_id);
}

auto HandleParseNode(Context& context, Parse::VariablePatternId node_id)
    -> bool {
  auto subpattern_id = context.node_stack().PopPattern();
  auto type_id = context.insts().Get(subpattern_id).type_id();

  if (subpattern_id == SemIR::ErrorInst::InstId) {
    context.node_stack().Push(node_id, SemIR::ErrorInst::InstId);
    return true;
  }

  auto pattern_id = SemIR::InstId::None;
  // In a parameter list, a `var` pattern is always a single `Call` parameter,
  // even if it contains multiple binding patterns.
  switch (context.full_pattern_stack().CurrentKind()) {
    case FullPatternStack::Kind::ExplicitParamList:
    case FullPatternStack::Kind::ImplicitParamList:
      pattern_id = AddInst<SemIR::VarParamPattern>(
          context, node_id,
          {.type_id = type_id, .subpattern_id = subpattern_id});
      break;
    case FullPatternStack::Kind::NameBindingDecl:
      pattern_id = AddInst<SemIR::VarPattern>(
          context, node_id,
          {.type_id = type_id, .subpattern_id = subpattern_id});
      break;
    case FullPatternStack::Kind::FieldDecl:
      // For class fields, a `FieldDecl` has already been created; do
      // not create a var pattern.
      return true;
    case FullPatternStack::Kind::NotInEitherParamList:
      CARBON_FATAL("Unreachable");
  }

  context.node_stack().Push(node_id, pattern_id);
  return true;
}

// Handle the end of the full-pattern of a let/var declaration (before the
// start of the initializer, if any).
static auto EndFullPattern(Context& context) -> void {
  EndSubpattern(context, context.node_stack());
  auto scope_id = context.scope_stack().PeekNameScopeId();
  if (scope_id.has_value() &&
      context.name_scopes().Get(scope_id).is_interface_definition()) {
    // Don't emit NameBindingDecl for an associated constant, because it will
    // always be empty.
    context.pattern_block_stack().PopAndDiscard();
    return;
  }
  auto pattern_block_id = context.pattern_block_stack().Pop();

  // For class fields, a `FieldDecl` has been created; skip creating a
  // name binding and var storage.
  if (context.full_pattern_stack().IsCurrentKindFieldDecl()) {
    return;
  }

  AddInst<SemIR::NameBindingDecl>(context, context.node_stack().PeekNodeId(),
                                  {.pattern_block_id = pattern_block_id});

  // Emit storage for any `var`s in the pattern now.
  bool returned =
      context.decl_introducer_state_stack().innermost().modifier_set.HasAnyOf(
          KeywordModifierSet::Returned);
  AddPatternVarStorage(context, pattern_block_id, returned);
}

static auto StartPatternInitializer(Context& context) -> bool {
  if (context.scope_stack().PeekIndex() == ScopeIndex::Package) {
    context.global_init().Resume();
  }
  context.full_pattern_stack().StartPatternInitializer();
  return true;
}

static auto EndPatternInitializer(Context& context) -> void {
  if (context.scope_stack().PeekIndex() == ScopeIndex::Package) {
    context.global_init().Suspend();
  }
  context.full_pattern_stack().EndPatternInitializer();
}

static auto HandleInitializer(Context& context, Parse::NodeId node_id) -> bool {
  EndFullPattern(context);
  context.node_stack().Push(node_id);
  StartPatternInitializer(context);
  return true;
}

auto HandleParseNode(Context& context, Parse::LetInitializerId node_id)
    -> bool {
  return HandleInitializer(context, node_id);
}

auto HandleParseNode(Context& context,
                     Parse::AssociatedConstantInitializerId node_id) -> bool {
  EndFullPattern(context);
  auto interface_decl =
      context.scope_stack().GetCurrentScopeAs<SemIR::InterfaceWithSelfDecl>();
  EndAssociatedConstantDeclRegion(context, interface_decl.interface_id);
  context.node_stack().Push(node_id);
  StartPatternInitializer(context);
  return true;
}

auto HandleParseNode(Context& context, Parse::VariableInitializerId node_id)
    -> bool {
  if (context.full_pattern_stack().IsCurrentKindFieldDecl()) {
    context.TODO(node_id, "Field initializer");
    return false;
  }

  return HandleInitializer(context, node_id);
}

// Make a default initialization expression for a `var` declaration.
static auto MakeDefaultInit(Context& context, SemIR::LocId loc_id,
                            SemIR::InstId pattern_id) -> SemIR::InstId {
  loc_id = context.insts().GetLocIdForDesugaring(loc_id);

  // Extract the matched type from the pattern.
  //
  // TODO: Diagnose if the pattern doesn't have a type, for example `var 123;`
  // or `var a: auto;`.
  auto pattern_type_id = context.insts().Get(pattern_id).type_id();
  auto type_inst_id = context.types().GetTypeInstId(
      SemIR::ExtractScrutineeType(context.sem_ir(), pattern_type_id));
  if (type_inst_id == SemIR::ErrorInst::InstId) {
    return SemIR::ErrorInst::InstId;
  }

  // Form `Type as Core.DefaultOrUnformed`.
  auto interface_id =
      LookupNameInCore(context, loc_id, CoreIdentifier::DefaultOrUnformed);
  auto interface_type = ExprAsType(context, loc_id, interface_id);
  auto facet_id = ConvertToValueOfType(context, loc_id, type_inst_id,
                                       interface_type.type_id);

  // Form a call to `facet.Op()`.
  auto op_name_id = context.core_identifiers().AddNameId(CoreIdentifier::Op);
  auto op_id = PerformMemberAccess(context, loc_id, facet_id, op_name_id);
  return PerformCall(context, loc_id, op_id, {});
}

namespace {
// State from HandleDecl, returned for type-specific handling.
struct DeclInfo {
  // The optional initializer.
  SemIR::InstId init_id = SemIR::InstId::None;
  // The pattern. For an associated constant, this is the associated constant
  // declaration.
  SemIR::InstId pattern_id = SemIR::InstId::None;
  DeclIntroducerState introducer = DeclIntroducerState();
};
}  // namespace

// Handles common logic for `let` and `var` declarations.
// TODO: There's still a lot of divergence here, including logic in
// handle_binding_pattern. These should really be better unified.
template <const Lex::TokenKind& IntroducerTokenKind,
          const Parse::NodeKind& IntroducerNodeKind,
          const Parse::NodeKind& InitializerNodeKind>
static auto HandleDecl(Context& context, Parse::NodeId node_id) -> DeclInfo {
  DeclInfo decl_info = DeclInfo();
  bool is_field_decl = context.full_pattern_stack().IsCurrentKindFieldDecl();

  // Handle the optional initializer.
  if (context.node_stack().PeekNextIs(InitializerNodeKind)) {
    decl_info.init_id = context.node_stack().PopExpr();
    context.node_stack().PopAndDiscardSoloNodeId<InitializerNodeKind>();
    EndPatternInitializer(context);
  } else {
    EndFullPattern(context);

    // For an associated constant declaration, handle the completed declaration
    // now. We will have done this at the `=` if there was an initializer.
    if constexpr (IntroducerNodeKind ==
                  Parse::NodeKind::AssociatedConstantIntroducer) {
      auto interface_decl =
          context.scope_stack()
              .GetCurrentScopeAs<SemIR::InterfaceWithSelfDecl>();
      EndAssociatedConstantDeclRegion(context, interface_decl.interface_id);
    }

    // A non-class variable declaration without an explicit initializer
    // is initialized by calling `(T as Core.DefaultOrUnformed).Op()`.
    if (!is_field_decl) {
      if constexpr (IntroducerNodeKind == Parse::NodeKind::VariableIntroducer) {
        StartPatternInitializer(context);
        decl_info.init_id = MakeDefaultInit(context, node_id,
                                            context.node_stack().PeekPattern());
        EndPatternInitializer(context);
      }
    }
  }
  context.full_pattern_stack().PopFullPattern();

  if (!is_field_decl) {
    decl_info.pattern_id = context.node_stack().PopPattern();
  }

  context.node_stack().PopAndDiscardSoloNodeId<IntroducerNodeKind>();

  // Process declaration modifiers.
  // TODO: For a qualified `let` or `var` declaration, this should use the
  // target scope of the name introduced in the declaration. See #2590.
  auto parent_scope_inst =
      context.name_scopes()
          .GetInstIfValid(context.scope_stack().PeekNameScopeId())
          .second;
  decl_info.introducer =
      context.decl_introducer_state_stack().Pop<IntroducerTokenKind>();
  CheckAccessModifiersOnDecl(context, decl_info.introducer, parent_scope_inst);

  return decl_info;
}

auto HandleParseNode(Context& context, Parse::LetDeclId node_id) -> bool {
  auto decl_info =
      HandleDecl<Lex::TokenKind::Let, Parse::NodeKind::LetIntroducer,
                 Parse::NodeKind::LetInitializer>(context, node_id);

  LimitModifiersOnDecl(
      context, decl_info.introducer,
      KeywordModifierSet::Access | KeywordModifierSet::Interface);

  // Diagnose interface modifiers given that we're not building an associated
  // constant. We use this rather than `LimitModifiersOnDecl` to get a more
  // specific error.
  RequireDefaultFinalOnlyInInterfaces(context, decl_info.introducer,
                                      SemIR::NameScopeId::None);

  if (decl_info.init_id.has_value()) {
    LocalPatternMatch(context, decl_info.pattern_id, decl_info.init_id);
  } else {
    CARBON_DIAGNOSTIC(
        ExpectedInitializerAfterLet, Error,
        "expected `=`; `let` declaration must have an initializer");
    context.emitter().Emit(LocIdForDiagnostics::TokenOnly(node_id),
                           ExpectedInitializerAfterLet);
  }
  return true;
}

auto HandleParseNode(Context& context, Parse::AssociatedConstantDeclId node_id)
    -> bool {
  auto decl_info = HandleDecl<Lex::TokenKind::Let,
                              Parse::NodeKind::AssociatedConstantIntroducer,
                              Parse::NodeKind::AssociatedConstantInitializer>(
      context, node_id);

  LimitModifiersOnDecl(
      context, decl_info.introducer,
      KeywordModifierSet::Access | KeywordModifierSet::Interface);

  auto interface_scope =
      context.scope_stack().GetCurrentScopeAs<SemIR::InterfaceWithSelfDecl>();
  // The `AssociatedConstantDecl` instruction and the corresponding
  // `AssociatedConstant` entity are built as part of handling the binding
  // pattern, but we still need to attach the default value, if any is
  // specified.
  if (decl_info.pattern_id == SemIR::ErrorInst::InstId) {
    const auto& interface =
        context.interfaces().Get(interface_scope.interface_id);
    context.name_scopes().Get(interface.scope_with_self_id).set_has_error();
    context.inst_block_stack().Pop();
    return true;
  }
  auto decl = context.insts().GetAs<SemIR::AssociatedConstantDecl>(
      decl_info.pattern_id);

  if (decl_info.introducer.modifier_set.HasAnyOf(
          KeywordModifierSet::Interface)) {
    context.TODO(decl_info.introducer.modifier_node_id(ModifierOrder::Decl),
                 "interface modifier");
  }

  // If there was an initializer, convert it and store it on the constant.
  if (decl_info.init_id.has_value()) {
    // TODO: Diagnose if the `default` modifier was not used.
    auto default_value_id =
        ConvertToValueOfType(context, node_id, decl_info.init_id, decl.type_id);
    auto& assoc_const = context.associated_constants().Get(decl.assoc_const_id);
    assoc_const.default_value_id = default_value_id;
  } else {
    // TODO: Either allow redeclarations of associated constants or diagnose if
    // the `default` modifier was used.
  }

  // Store the decl block on the declaration.
  decl.decl_block_id = context.inst_block_stack().Pop();
  ReplaceInstPreservingConstantValue(context, decl_info.pattern_id, decl);

  context.inst_block_stack().AddInstId(decl_info.pattern_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::VariableDeclId node_id) -> bool {
  auto decl_info =
      HandleDecl<Lex::TokenKind::Var, Parse::NodeKind::VariableIntroducer,
                 Parse::NodeKind::VariableInitializer>(context, node_id);

  LimitModifiersOnDecl(
      context, decl_info.introducer,
      KeywordModifierSet::Access | KeywordModifierSet::Returned);

  if (context.scope_stack().TryGetCurrentScopeAs<SemIR::ClassDecl>()) {
    return true;
  }

  LocalPatternMatch(context, decl_info.pattern_id, decl_info.init_id);
  return true;
}

}  // namespace Carbon::Check

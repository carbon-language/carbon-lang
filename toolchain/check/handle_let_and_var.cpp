// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/decl_introducer_state.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/interface.h"
#include "toolchain/check/keyword_modifier_set.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/check/pattern_match.h"
#include "toolchain/check/return.h"
#include "toolchain/check/subpattern.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/pattern.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Handles the start of a declaration of an associated constant.
static auto StartAssociatedConstant(Context& context) -> void {
  // An associated constant is always generic.
  StartGenericDecl(context);
  // Collect the declarations nested in the associated constant in a decl
  // block. This is popped by FinishAssociatedConstantDecl.
  context.inst_block_stack().Push();
}

// Handles the end of the declaration region of an associated constant. This is
// called at the `=` or the `;` of the declaration, whichever comes first.
static auto EndAssociatedConstantDeclRegion(Context& context,
                                            SemIR::InterfaceId interface_id)
    -> void {
  // TODO: Stop special-casing tuple patterns once they behave like other
  // patterns.
  if (context.node_stack().PeekIs(Parse::NodeKind::TuplePattern)) {
    DiscardGenericDecl(context);
    return;
  }

  // Peek the pattern. For a valid associated constant, the corresponding
  // instruction will be an `AssociatedConstantDecl` instruction.
  auto decl_id = context.node_stack().PeekPattern();
  auto assoc_const_decl =
      context.insts().TryGetAs<SemIR::AssociatedConstantDecl>(decl_id);
  if (!assoc_const_decl) {
    // The pattern wasn't suitable for an associated constant. We'll detect
    // and diagnose this later. For now, just clean up the generic stack.
    DiscardGenericDecl(context);
    return;
  }

  // Finish the declaration region of this generic.
  auto& assoc_const =
      context.associated_constants().Get(assoc_const_decl->assoc_const_id);
  assoc_const.generic_id = BuildGenericDecl(context, decl_id);

  // Build a corresponding associated entity and add it into scope. Note
  // that we do this outside the generic region.
  // TODO: The instruction is added to the associated constant's decl block.
  // It probably should be in the interface's body instead.
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
  context.full_pattern_stack().PushFullPattern(
      FullPatternStack::Kind::NameBindingDecl);
  BeginSubpattern(context);
  return true;
}

auto HandleParseNode(Context& context, Parse::LetIntroducerId node_id) -> bool {
  if (context.scope_stack().GetCurrentScopeAs<SemIR::InterfaceDecl>()) {
    StartAssociatedConstant(context);
  }

  return HandleIntroducer<Lex::TokenKind::Let>(context, node_id);
}

auto HandleParseNode(Context& context, Parse::VariableIntroducerId node_id)
    -> bool {
  return HandleIntroducer<Lex::TokenKind::Var>(context, node_id);
}

// Returns a VarStorage inst for the given `var` pattern. If the pattern
// is the body of a returned var, this reuses the return slot, and otherwise it
// adds a new inst.
static auto GetOrAddStorage(Context& context, SemIR::InstId var_pattern_id)
    -> SemIR::InstId {
  if (context.decl_introducer_state_stack().innermost().modifier_set.HasAnyOf(
          KeywordModifierSet::Returned)) {
    auto& function = GetCurrentFunctionForReturn(context);
    auto return_info =
        SemIR::ReturnTypeInfo::ForFunction(context.sem_ir(), function);
    if (return_info.has_return_slot()) {
      return GetCurrentReturnSlot(context);
    }
  }
  auto pattern = context.insts().GetWithLocId(var_pattern_id);

  return AddInst(context, pattern.loc_id,
                 SemIR::VarStorage{
                     .type_id = pattern.inst.type_id(),
                     .pretty_name_id = SemIR::GetPrettyNameFromPatternId(
                         context.sem_ir(),
                         pattern.inst.As<SemIR::VarPattern>().subpattern_id)});
}

auto HandleParseNode(Context& context, Parse::VariablePatternId node_id)
    -> bool {
  auto subpattern_id = context.node_stack().PopPattern();
  auto type_id = context.insts().Get(subpattern_id).type_id();

  // In a parameter list, a `var` pattern is always a single `Call` parameter,
  // even if it contains multiple binding patterns.
  switch (context.full_pattern_stack().CurrentKind()) {
    case FullPatternStack::Kind::ExplicitParamList:
    case FullPatternStack::Kind::ImplicitParamList:
      subpattern_id = AddPatternInst<SemIR::RefParamPattern>(
          context, node_id,
          {.type_id = type_id,
           .subpattern_id = subpattern_id,
           .index = SemIR::CallParamIndex::None});
      break;
    case FullPatternStack::Kind::NameBindingDecl:
      break;
  }

  auto pattern_id = AddPatternInst<SemIR::VarPattern>(
      context, node_id, {.type_id = type_id, .subpattern_id = subpattern_id});
  context.node_stack().Push(node_id, pattern_id);
  return true;
}

// Handle the end of the full-pattern of a let/var declaration (before the
// start of the initializer, if any).
static auto EndFullPattern(Context& context) -> void {
  if (context.scope_stack().GetCurrentScopeAs<SemIR::InterfaceDecl>()) {
    // Don't emit NameBindingDecl for an associated constant, because it will
    // always be empty.
    context.pattern_block_stack().PopAndDiscard();
    return;
  }
  auto pattern_block_id = context.pattern_block_stack().Pop();
  AddInst<SemIR::NameBindingDecl>(context, context.node_stack().PeekNodeId(),
                                  {.pattern_block_id = pattern_block_id});

  // We need to emit the VarStorage insts early, because they may be output
  // arguments for the initializer. However, we can't emit them when we emit
  // the corresponding `VarPattern`s because they're part of the pattern match,
  // not part of the pattern.
  // TODO: find a way to do this without walking the whole pattern block.
  for (auto inst_id : context.inst_blocks().Get(pattern_block_id)) {
    if (context.insts().Is<SemIR::VarPattern>(inst_id)) {
      context.var_storage_map().Insert(inst_id,
                                       GetOrAddStorage(context, inst_id));
    }
  }
}

static auto HandleInitializer(Context& context, Parse::NodeId node_id) -> bool {
  EndFullPattern(context);
  if (context.scope_stack().PeekIndex() == ScopeIndex::Package) {
    context.global_init().Resume();
  }
  context.node_stack().Push(node_id);
  context.full_pattern_stack().StartPatternInitializer();
  return true;
}

auto HandleParseNode(Context& context, Parse::LetInitializerId node_id)
    -> bool {
  if (auto interface_decl =
          context.scope_stack().GetCurrentScopeAs<SemIR::InterfaceDecl>()) {
    EndAssociatedConstantDeclRegion(context, interface_decl->interface_id);

    // Start building the definition region of the constant.
    StartGenericDefinition(context);
  }

  return HandleInitializer(context, node_id);
}

auto HandleParseNode(Context& context, Parse::VariableInitializerId node_id)
    -> bool {
  return HandleInitializer(context, node_id);
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
static auto HandleDecl(Context& context) -> DeclInfo {
  DeclInfo decl_info = DeclInfo();

  // Handle the optional initializer.
  if (context.node_stack().PeekNextIs(InitializerNodeKind)) {
    decl_info.init_id = context.node_stack().PopExpr();
    context.node_stack().PopAndDiscardSoloNodeId<InitializerNodeKind>();
    if (context.scope_stack().PeekIndex() == ScopeIndex::Package) {
      context.global_init().Suspend();
    }
    context.full_pattern_stack().EndPatternInitializer();
  } else {
    // For an associated constant declaration, handle the completed declaration
    // now. We will have done this at the `=` if there was an initializer.
    if (IntroducerTokenKind == Lex::TokenKind::Let) {
      if (auto interface_decl =
              context.scope_stack().GetCurrentScopeAs<SemIR::InterfaceDecl>()) {
        EndAssociatedConstantDeclRegion(context, interface_decl->interface_id);
      }
    }

    EndFullPattern(context);
  }
  context.full_pattern_stack().PopFullPattern();

  decl_info.pattern_id = context.node_stack().PopPattern();

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

// Finishes an associated constant declaration. This is called at the `;` to
// perform any final steps. The `AssociatedConstantDecl` instruction and the
// corresponding `AssociatedConstant` entity are built as part of handling the
// binding pattern, but we still need to finish building the `Generic` object
// and attach the default value, if any is specified.
static auto FinishAssociatedConstant(Context& context, Parse::LetDeclId node_id,
                                     SemIR::InterfaceId interface_id,
                                     DeclInfo& decl_info) -> void {
  auto decl = context.insts().TryGetAs<SemIR::AssociatedConstantDecl>(
      decl_info.pattern_id);
  if (!decl) {
    if (decl_info.pattern_id != SemIR::ErrorInst::SingletonInstId) {
      CARBON_DIAGNOSTIC(ExpectedSymbolicBindingInAssociatedConstant, Error,
                        "pattern in associated constant declaration must be a "
                        "single `:!` binding");
      context.emitter().Emit(context.insts().GetLocId(decl_info.pattern_id),
                             ExpectedSymbolicBindingInAssociatedConstant);
    }
    context.name_scopes()
        .Get(context.interfaces().Get(interface_id).scope_id)
        .set_has_error();
    if (decl_info.init_id.has_value()) {
      DiscardGenericDecl(context);
    }
    context.inst_block_stack().Pop();
    return;
  }

  if (decl_info.introducer.modifier_set.HasAnyOf(
          KeywordModifierSet::Interface)) {
    context.TODO(decl_info.introducer.modifier_node_id(ModifierOrder::Decl),
                 "interface modifier");
  }

  // If there was an initializer, convert it and store it on the constant.
  if (decl_info.init_id.has_value()) {
    // TODO: Diagnose if the `default` modifier was not used.
    auto default_value_id = ConvertToValueOfType(
        context, node_id, decl_info.init_id, decl->type_id);
    auto& assoc_const =
        context.associated_constants().Get(decl->assoc_const_id);
    assoc_const.default_value_id = default_value_id;
    FinishGenericDefinition(context, assoc_const.generic_id);
  } else {
    // TODO: Either allow redeclarations of associated constants or diagnose if
    // the `default` modifier was used.
  }

  // Store the decl block on the declaration.
  decl->decl_block_id = context.inst_block_stack().Pop();
  ReplaceInstPreservingConstantValue(context, decl_info.pattern_id, *decl);

  context.inst_block_stack().AddInstId(decl_info.pattern_id);
}

auto HandleParseNode(Context& context, Parse::LetDeclId node_id) -> bool {
  auto decl_info =
      HandleDecl<Lex::TokenKind::Let, Parse::NodeKind::LetIntroducer,
                 Parse::NodeKind::LetInitializer>(context);

  LimitModifiersOnDecl(
      context, decl_info.introducer,
      KeywordModifierSet::Access | KeywordModifierSet::Interface);

  // At interface scope, we are forming an associated constant, which has
  // different rules.
  if (auto interface_scope =
          context.scope_stack().GetCurrentScopeAs<SemIR::InterfaceDecl>()) {
    FinishAssociatedConstant(context, node_id, interface_scope->interface_id,
                             decl_info);
    return true;
  }

  // Diagnose interface modifiers given that we're not building an associated
  // constant. We use this rather than `LimitModifiersOnDecl` to get a more
  // specific error.
  RequireDefaultFinalOnlyInInterfaces(context, decl_info.introducer,
                                      std::nullopt);

  if (decl_info.init_id.has_value()) {
    LocalPatternMatch(context, decl_info.pattern_id, decl_info.init_id);
  } else {
    CARBON_DIAGNOSTIC(
        ExpectedInitializerAfterLet, Error,
        "expected `=`; `let` declaration must have an initializer");
    context.emitter().Emit(TokenOnly(node_id), ExpectedInitializerAfterLet);
  }
  return true;
}

auto HandleParseNode(Context& context, Parse::VariableDeclId node_id) -> bool {
  auto decl_info =
      HandleDecl<Lex::TokenKind::Var, Parse::NodeKind::VariableIntroducer,
                 Parse::NodeKind::VariableInitializer>(context);

  LimitModifiersOnDecl(
      context, decl_info.introducer,
      KeywordModifierSet::Access | KeywordModifierSet::Returned);

  if (auto class_scope =
          context.scope_stack().GetCurrentScopeAs<SemIR::ClassDecl>()) {
    auto var = context.insts().GetAs<SemIR::VarPattern>(decl_info.pattern_id);
    if (!context.insts().TryGetAs<SemIR::FieldDecl>(var.subpattern_id)) {
      CARBON_DIAGNOSTIC(ExpectedSymbolicBindingInFieldDecl, Error,
                        "pattern in field declaration is not a "
                        "single `:` binding");
      context.emitter().Emit(context.insts().GetLocId(var.subpattern_id),
                             ExpectedSymbolicBindingInFieldDecl);
      context.name_scopes()
          .Get(context.classes().Get(class_scope->class_id).scope_id)
          .set_has_error();
    }
    if (decl_info.init_id.has_value()) {
      // TODO: In a class scope, we should instead save the initializer
      // somewhere so that we can use it as a default.
      context.TODO(node_id, "Field initializer");
    }
    return true;
  }
  if (context.scope_stack().GetCurrentScopeAs<SemIR::InterfaceDecl>()) {
    CARBON_DIAGNOSTIC(VarInInterfaceDecl, Error,
                      "`var` declaration in interface");
    context.emitter().Emit(node_id, VarInInterfaceDecl);
    return true;
  }

  LocalPatternMatch(context, decl_info.pattern_id, decl_info.init_id);
  return true;
}

}  // namespace Carbon::Check

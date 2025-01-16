// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/decl_introducer_state.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/interface.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/check/pattern_match.h"
#include "toolchain/check/return.h"
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
  context.full_pattern_stack().PushFullPattern();
  context.BeginSubpattern();
  return true;
}

auto HandleParseNode(Context& context, Parse::LetIntroducerId node_id) -> bool {
  return HandleIntroducer<Lex::TokenKind::Let>(context, node_id);
}

auto HandleParseNode(Context& context, Parse::VariableIntroducerId node_id)
    -> bool {
  return HandleIntroducer<Lex::TokenKind::Var>(context, node_id);
}

// Returns a VarStorage inst for the given pattern. If the pattern
// is the body of a returned var, this reuses the return slot, and otherwise it
// adds a new inst.
static auto GetOrAddStorage(Context& context, SemIR::InstId pattern_id)
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
  auto pattern = context.insts().GetWithLocId(pattern_id);
  auto subpattern =
      context.insts().Get(pattern.inst.As<SemIR::VarPattern>().subpattern_id);

  // Try to populate name_id on a best-effort basis.
  auto name_id = SemIR::NameId::Invalid;
  if (auto binding_pattern = subpattern.TryAs<SemIR::BindingPattern>()) {
    name_id =
        context.entity_names().Get(binding_pattern->entity_name_id).name_id;
  }
  return context.AddInst(SemIR::LocIdAndInst::UncheckedLoc(
      pattern.loc_id, SemIR::VarStorage{.type_id = subpattern.type_id(),
                                        .pretty_name_id = name_id}));
}

auto HandleParseNode(Context& context, Parse::VariablePatternId node_id)
    -> bool {
  auto subpattern_id = SemIR::InstId::Invalid;
  if (context.node_stack().PeekIs(Parse::NodeKind::TuplePattern)) {
    context.node_stack().PopAndIgnore();
    CARBON_CHECK(
        context.node_stack().PeekIs(Parse::NodeKind::TuplePatternStart));
    context.node_stack().PopAndIgnore();
    context.inst_block_stack().PopAndDiscard();
    context.TODO(node_id, "tuple pattern in let/var");
    subpattern_id = SemIR::ErrorInst::SingletonInstId;
  } else {
    subpattern_id = context.node_stack().PopPattern();
  }
  auto type_id = context.insts().Get(subpattern_id).type_id();

  auto pattern_id = context.AddPatternInst<SemIR::VarPattern>(
      node_id, {.type_id = type_id, .subpattern_id = subpattern_id});
  context.node_stack().Push(node_id, pattern_id);
  return true;
}

// Handle the end of the full-pattern of a let/var declaration (before the
// start of the initializer, if any).
static auto EndFullPattern(Context& context) -> void {
  if (context.GetCurrentScopeAs<SemIR::InterfaceDecl>()) {
    // Don't emit NameBindingDecl for an associated constant, because it will
    // always be empty.
    context.pattern_block_stack().PopAndDiscard();
    return;
  }
  auto pattern_block_id = context.pattern_block_stack().Pop();
  context.AddInst<SemIR::NameBindingDecl>(
      context.node_stack().PeekNodeId(),
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

  // Handle the optional initializer.
  if (context.node_stack().PeekNextIs(InitializerNodeKind)) {
    decl_info->init_id = context.node_stack().PopExpr();
    context.node_stack().PopAndDiscardSoloNodeId<InitializerNodeKind>();
    if (context.scope_stack().PeekIndex() == ScopeIndex::Package) {
      context.global_init().Suspend();
    }
    context.full_pattern_stack().EndPatternInitializer();
  } else {
    EndFullPattern(context);
  }
  context.full_pattern_stack().PopFullPattern();

  if (context.node_stack().PeekIs(Parse::NodeKind::TuplePattern)) {
    if (decl_info->init_id &&
        context.scope_stack().PeekIndex() == ScopeIndex::Package) {
      context.global_init().Suspend();
    }
    context.TODO(node_id, "tuple pattern in let/var");
    decl_info = std::nullopt;
    return decl_info;
  }

  decl_info->pattern_id = context.node_stack().PopPattern();

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

static auto HandleAssociatedConstantDecl(Context& context,
                                         Parse::LetDeclId node_id,
                                         DeclInfo decl_info,
                                         SemIR::InterfaceDecl interface_scope)
    -> void {
  auto pattern = context.insts().GetWithLocId(decl_info.pattern_id);

  if (decl_info.init_id) {
    // Convert the value to match the type of the pattern.
    ConvertToValueOfType(context, node_id, *decl_info.init_id,
                         pattern.inst.type_id());
  }

  if (auto decl = pattern.inst.TryAs<SemIR::AssociatedConstantDecl>();
      !decl.has_value()) {
    CARBON_DIAGNOSTIC(ExpectedSymbolicBindingInAssociatedConstant, Error,
                      "pattern in associated constant declaration must be a "
                      "single `:!` binding");
    context.emitter().Emit(pattern.loc_id,
                           ExpectedSymbolicBindingInAssociatedConstant);
    context.name_scopes()
        .Get(context.interfaces().Get(interface_scope.interface_id).scope_id)
        .set_has_error();
  }
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

  // At interface scope, we are forming an associated constant, which has
  // different rules.
  if (auto interface_scope = context.GetCurrentScopeAs<SemIR::InterfaceDecl>();
      interface_scope) {
    HandleAssociatedConstantDecl(context, node_id, *decl_info,
                                 *interface_scope);
    return true;
  }

  if (decl_info->init_id) {
    LocalPatternMatch(context, decl_info->pattern_id, *decl_info->init_id);
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
                 Parse::NodeKind::VariableInitializer>(context, node_id);
  if (!decl_info) {
    return false;
  }

  LimitModifiersOnDecl(
      context, decl_info->introducer,
      KeywordModifierSet::Access | KeywordModifierSet::Returned);

  if (context.GetCurrentScopeAs<SemIR::ClassDecl>()) {
    if (decl_info->init_id) {
      // TODO: In a class scope, we should instead save the initializer
      // somewhere so that we can use it as a default.
      context.TODO(node_id, "Field initializer");
    }
    return true;
  }
  LocalPatternMatch(context, decl_info->pattern_id,
                    decl_info->init_id.value_or(SemIR::InstId::Invalid));

  return true;
}

}  // namespace Carbon::Check

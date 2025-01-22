// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/decl_introducer_state.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/interface.h"
#include "toolchain/check/keyword_modifier_set.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

template <Lex::TokenKind::RawEnumType Kind>
static auto HandleIntroducer(Context& context, Parse::NodeId node_id) -> bool {
  context.decl_introducer_state_stack().Push<Kind>();
  // Push a bracketing node and pattern block to establish the pattern context.
  context.node_stack().PushOptional(node_id, SemIR::InstId::Invalid);
  context.pattern_block_stack().Push();
  context.BeginSubpattern();
  return true;
}

auto HandleParseNode(Context& context, Parse::LetIntroducerId node_id) -> bool {
  if (context.GetCurrentScopeAs<SemIR::InterfaceDecl>()) {
    // An associated constant is always generic.
    StartGenericDecl(context);
    // Collect the declarations nested in the associated constant in a decl
    // block.
    context.inst_block_stack().Push();
  }

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

// Pops a pattern from the top of the node stack and returns it. The returned
// instruction will be `None` if we popped a tuple pattern.
// TODO: Change tuple patterns to have the same representation on the node stack
// as other patterns.
static auto PopBindingDeclPattern(Context& context)
    -> std::pair<Parse::NodeId, SemIR::InstId> {
  // TODO: Update binding-pattern handling to use the pattern block even in
  // a let/var context, and then consume it here.
  context.pattern_block_stack().PopAndDiscard();

  auto [node_id, block_id] =
      context.node_stack().PopWithNodeIdIf<Parse::NodeKind::TuplePattern>();
  if (block_id) {
    // TODO: Tuple patterns leave behind an entry on the subpattern stack.
    // Popping it here seems wrong, but it's not clear how this should work.
    context.EndSubpatternAsEmpty();
    context.node_stack().PopForSoloNodeId<Parse::NodeKind::TuplePatternStart>();
    return {node_id, SemIR::InstId::Invalid};
  }
  return context.node_stack().PopPatternWithNodeId();
}

// Builds an associated constant declaration for a `let`. This should be called
// with the pattern in an interface-scope let declaration at the top of the node
// stack. Pops the pattern and returns the new declaration.
static auto HandleAssociatedConstantDecl(Context& context,
                                         SemIR::InterfaceId interface_id)
    -> SemIR::InstId {
  auto& interface_info = context.interfaces().Get(interface_id);

  // Pop the pattern. This must be a single symbolic name binding pattern.
  auto [pattern_node_id, pattern_id] = PopBindingDeclPattern(context);

  auto binding_pattern =
      pattern_id.is_valid()
          ? context.insts().TryGetAs<SemIR::BindSymbolicName>(pattern_id)
          : std::nullopt;
  if (!binding_pattern) {
    CARBON_DIAGNOSTIC(ExpectedSymbolicBindingInAssociatedConstant, Error,
                      "pattern in associated constant declaration must be a "
                      "single `:!` binding");
    context.emitter().Emit(pattern_node_id,
                           ExpectedSymbolicBindingInAssociatedConstant);
    context.name_scopes().Get(interface_info.scope_id).set_has_error();
    DiscardGenericDecl(context);
    return SemIR::ErrorInst::SingletonInstId;
  }

  // TODO: Don't create the EntityName object for an associated constant. We
  // don't use it.
  auto entity_name =
      context.entity_names().Get(binding_pattern->entity_name_id);

  // The pattern instruction will be replaced by the associated constant
  // declaration.
  auto decl_id = pattern_id;

  // Create the associated constant.
  auto assoc_const_id = context.associated_constants().Add(
      {.name_id = entity_name.name_id,
       .parent_scope_id = entity_name.parent_scope_id,
       .generic_id = SemIR::GenericId::Invalid,
       .decl_id = decl_id,
       .default_value_id = SemIR::InstId::Invalid});

  // Replace the tentative BindSymbolicName instruction with the associated
  // constant declaration. We will update this again with a location and a decl
  // block at the end of the declaration.
  context.ReplaceLocIdAndInstBeforeConstantUse(
      decl_id, SemIR::LocIdAndInst::UncheckedLoc(
                   SemIR::LocId::Invalid,
                   SemIR::AssociatedConstantDecl{
                       .type_id = binding_pattern->type_id,
                       .assoc_const_id = assoc_const_id,
                       .decl_block_id = SemIR::InstBlockId::Invalid}));

  // Finish the declaration region of the generic associated constant.
  // We can't do this before now because the region needs to contain the
  // associated constant declaration itself, because its type may depend on
  // Self.
  context.associated_constants().Get(assoc_const_id).generic_id =
      BuildGenericDecl(context, decl_id);

  // Add a corresponding associated entity name to the interface scope.
  // TODO: This ends up inside the decl block of the associated constant. Should
  // it be outside instead?
  auto assoc_id = BuildAssociatedEntity(context, interface_id, decl_id);
  auto name_context = context.decl_name_stack().MakeUnqualifiedName(
      pattern_node_id, entity_name.name_id);
  context.decl_name_stack().AddNameOrDiagnose(
      name_context, assoc_id,
      context.decl_introducer_state_stack()
          .innermost()
          .modifier_set.GetAccessKind());
  return decl_id;
}

auto HandleParseNode(Context& context, Parse::LetInitializerId node_id)
    -> bool {
  auto decl_id = SemIR::InstId::Invalid;
  if (auto interface_decl = context.GetCurrentScopeAs<SemIR::InterfaceDecl>()) {
    decl_id = HandleAssociatedConstantDecl(context, interface_decl->interface_id);

    // Store the declaration ID on the introducer node.
    auto [intro_node_id, _] =
        context.node_stack().PopWithNodeId<Parse::NodeKind::LetIntroducer>();
    context.node_stack().Push(intro_node_id, decl_id);

    StartGenericDefinition(context);
  }

  return HandleInitializer(context, node_id);
}

auto HandleParseNode(Context& context, Parse::VariableInitializerId node_id)
    -> bool {
  return HandleInitializer(context, node_id);
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
    context.decl_name_stack().AddNameOrDiagnose(name_context, pattern_id,
                                                access_kind);
    return bind_name->value_id;
  } else if (auto field_decl =
                 context.insts().TryGetAs<SemIR::FieldDecl>(pattern_id)) {
    // Introduce the field name into the class.
    auto name_context = context.decl_name_stack().MakeUnqualifiedName(
        context.insts().GetLocId(pattern_id), field_decl->name_id);
    context.decl_name_stack().AddNameOrDiagnose(name_context, pattern_id,
                                                access_kind);
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
  SemIR::InstId init_id = SemIR::InstId::Invalid;
  // The pattern, if we are not declaring an associated constant.
  SemIR::InstId pattern_id = SemIR::InstId::Invalid;
  // The associated constant declaration, if we are declaring an associated
  // constant.
  SemIR::InstId assoc_const_id = SemIR::InstId::Invalid;
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
  }

  // Next we either have a pattern and an introducer or, if we've already built
  // a declaration, just the introducer.
  if (auto decl_id = context.node_stack().PopIf<IntroducerNodeKind>()) {
    // If we have a declaration already, it's an associated constant
    // declaration.
    decl_info.assoc_const_id = *decl_id;
  } else {
    // Handle the pattern.
    if (auto interface_decl = context.GetCurrentScopeAs<SemIR::InterfaceDecl>();
        interface_decl && IntroducerTokenKind == Lex::TokenKind::Let) {
      // This is an associated constant declaration.
      decl_info.assoc_const_id =
          HandleAssociatedConstantDecl(context, interface_decl->interface_id);
    } else {
      auto [pattern_node_id, pattern_id] = PopBindingDeclPattern(context);
      if (!pattern_id.is_valid()) {
        context.TODO(pattern_node_id, "tuple pattern in let/var");
        pattern_id = SemIR::ErrorInst::SingletonInstId;
      }
      decl_info.pattern_id = pattern_id;
    }

    if constexpr (IntroducerTokenKind == Lex::TokenKind::Var) {
      // Pop the `returned` modifier if present.
      context.node_stack()
          .PopAndDiscardSoloNodeIdIf<Parse::NodeKind::ReturnedModifier>();
    }

    context.node_stack().Pop<IntroducerNodeKind>();
  }

  // Process declaration modifiers.
  // TODO: For a qualified `let` or `var` declaration, this should use the
  // target scope of the name introduced in the declaration. See #2590.
  auto parent_scope_inst =
      context.name_scopes()
          .GetInstIfValid(context.scope_stack().PeekNameScopeId())
          .second;
  decl_info.introducer =
      context.decl_introducer_state_stack().Pop<IntroducerTokenKind>();
  CheckAccessModifiersOnDecl(context, decl_info.introducer,
                             parent_scope_inst);

  return decl_info;
}

// Finishes an associated constant declaration. This is called at the `;` to
// perform any final steps. We already built the declaration instruction.
static auto FinishAssociatedConstantDecl(Context& context,
                                         Parse::LetDeclId node_id,
                                         DeclInfo& decl_info) -> void {
  if (decl_info.assoc_const_id == SemIR::ErrorInst::SingletonInstId) {
    context.inst_block_stack().Pop();
    return;
  }

  if (decl_info.introducer.modifier_set.HasAnyOf(
          KeywordModifierSet::Interface)) {
    context.TODO(decl_info.introducer.modifier_node_id(ModifierOrder::Decl),
                 "interface modifier");
  }

  auto decl = context.insts().GetAs<SemIR::AssociatedConstantDecl>(
      decl_info.assoc_const_id);

  // If there was an initializer, convert it and store it on the constant.
  if (decl_info.init_id.is_valid()) {
    // TODO: Diagnose if the `default` modifier was not used.
    auto default_value_id =
        ConvertToValueOfType(context, node_id, decl_info.init_id, decl.type_id);
    auto& assoc_const = context.associated_constants().Get(decl.assoc_const_id);
    assoc_const.default_value_id = default_value_id;
    FinishGenericDefinition(context, assoc_const.generic_id);
  } else {
    // TODO: Either allow redeclarations of associated constants or diagnose if
    // the `default` modifier was used.
  }

  // Store the decl block on the declaration.
  decl.decl_block_id = context.inst_block_stack().Pop();
  context.ReplaceLocIdAndInstPreservingConstantValue(
      decl_info.assoc_const_id, SemIR::LocIdAndInst(node_id, decl));

  context.inst_block_stack().AddInstId(decl_info.assoc_const_id);
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
  if (decl_info.assoc_const_id.is_valid()) {
    FinishAssociatedConstantDecl(context, node_id, decl_info);
    return true;
  }

  // Diagnose interface modifiers given that we're not building an associated
  // constant. We use this rather than `LimitModifiersOnDecl` to get a more
  // specific error.
  RequireDefaultFinalOnlyInInterfaces(context, decl_info.introducer,
                                      std::nullopt);

  auto pattern = context.insts().GetWithLocId(decl_info.pattern_id);

  if (decl_info.init_id.is_valid()) {
    // Convert the value to match the type of the pattern.
    decl_info.init_id = ConvertToValueOfType(
        context, node_id, decl_info.init_id, pattern.inst.type_id());
  } else {
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
  bind_name.value_id = decl_info.init_id.is_valid()
                           ? decl_info.init_id
                           : SemIR::ErrorInst::SingletonInstId;
  context.ReplaceInstBeforeConstantUse(decl_info.pattern_id, bind_name);
  context.inst_block_stack().AddInstId(decl_info.pattern_id);

  HandleNameBinding(context, decl_info.pattern_id,
                    decl_info.introducer.modifier_set.GetAccessKind());

  if (decl_info.init_id.is_valid() &&
      context.scope_stack().PeekIndex() == ScopeIndex::Package) {
    context.global_init().Suspend();
  }

  return true;
}

auto HandleParseNode(Context& context, Parse::VariableDeclId node_id) -> bool {
  auto decl_info =
      HandleDecl<Lex::TokenKind::Var, Parse::NodeKind::VariableIntroducer,
                 Parse::NodeKind::VariableInitializer>(context);

  LimitModifiersOnDecl(context, decl_info.introducer,
                       KeywordModifierSet::Access);

  decl_info.pattern_id =
      HandleNameBinding(context, decl_info.pattern_id,
                        decl_info.introducer.modifier_set.GetAccessKind());

  // If there was an initializer, assign it to the storage.
  if (decl_info.init_id.is_valid()) {
    if (context.GetCurrentScopeAs<SemIR::ClassDecl>()) {
      // TODO: In a class scope, we should instead save the initializer
      // somewhere so that we can use it as a default.
      context.TODO(node_id, "Field initializer");
    } else {
      decl_info.init_id = Initialize(context, node_id, decl_info.pattern_id,
                                      decl_info.init_id);
      // TODO: Consider using different instruction kinds for assignment
      // versus initialization.
      context.AddInst<SemIR::Assign>(node_id, {.lhs_id = decl_info.pattern_id,
                                               .rhs_id = decl_info.init_id});
    }

    if (context.scope_stack().PeekIndex() == ScopeIndex::Package) {
      context.global_init().Suspend();
    }
  }

  return true;
}

}  // namespace Carbon::Check

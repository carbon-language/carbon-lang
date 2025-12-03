// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <optional>
#include <utility>

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/impl.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/pattern_match.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/parse/typed_nodes.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::ImplIntroducerId node_id)
    -> bool {
  // This might be a generic impl.
  StartGenericDecl(context);

  // Create an instruction block to hold the instructions created for the type
  // and interface.
  context.inst_block_stack().Push();

  // Push the bracketing node.
  context.node_stack().Push(node_id);

  // Optional modifiers follow.
  context.decl_introducer_state_stack().Push<Lex::TokenKind::Impl>();

  // An impl doesn't have a name per se, but it makes the processing more
  // consistent to imagine that it does. This also gives us a scope for implicit
  // parameters.
  context.decl_name_stack().PushScopeAndStartName();
  return true;
}

auto HandleParseNode(Context& context, Parse::ForallId /*node_id*/) -> bool {
  // Push a pattern block for the signature of the `forall`.
  context.pattern_block_stack().Push();
  context.full_pattern_stack().PushFullPattern(
      FullPatternStack::Kind::ImplicitParamList);
  return true;
}

auto HandleParseNode(Context& context, Parse::ImplTypeAsId node_id) -> bool {
  auto [self_node, self_id] = context.node_stack().PopExprWithNodeId();
  auto self_type_inst_id = ExprAsType(context, self_node, self_id).inst_id;
  context.node_stack().Push(node_id, self_type_inst_id);

  // Introduce `Self`. Note that we add this name lexically rather than adding
  // to the `NameScopeId` of the `impl`, because this happens before we enter
  // the `impl` scope or even identify which `impl` we're declaring.
  // TODO: Revisit this once #3714 is resolved.
  AddNameToLookup(context, SemIR::NameId::SelfType, self_type_inst_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::ImplDefaultSelfAsId node_id)
    -> bool {
  auto self_inst_id = SemIR::TypeInstId::None;

  if (auto self_type_id = GetImplDefaultSelfType(context);
      self_type_id.has_value()) {
    // Build the implicit access to the enclosing `Self`.
    // TODO: Consider calling `HandleNameAsExpr` to build this implicit `Self`
    // expression. We've already done the work to check that the enclosing
    // context is a class and found its `Self`, so additionally performing an
    // unqualified name lookup would be redundant work, but would avoid
    // duplicating the handling of the `Self` expression.
    self_inst_id = AddTypeInst(
        context, node_id,
        SemIR::NameRef{.type_id = SemIR::TypeType::TypeId,
                       .name_id = SemIR::NameId::SelfType,
                       .value_id = context.types().GetInstId(self_type_id)});
  } else {
    CARBON_DIAGNOSTIC(ImplAsOutsideClass, Error,
                      "`impl as` can only be used in a class");
    context.emitter().Emit(node_id, ImplAsOutsideClass);
    self_inst_id = SemIR::ErrorInst::TypeInstId;
  }

  // There's no need to push `Self` into scope here, because we can find it in
  // the parent class scope.
  context.node_stack().Push(node_id, self_inst_id);
  return true;
}

// Pops the parameters of an `impl`, forming a `NameComponent` with no
// associated name that describes them.
static auto PopImplIntroducerAndParamsAsNameComponent(
    Context& context, Parse::AnyImplDeclId end_of_decl_node_id)
    -> NameComponent {
  auto [implicit_params_loc_id, implicit_param_patterns_id] =
      context.node_stack()
          .PopWithNodeIdIf<Parse::NodeKind::ImplicitParamList>();

  if (implicit_param_patterns_id) {
    context.node_stack()
        .PopAndDiscardSoloNodeId<Parse::NodeKind::ImplicitParamListStart>();
    // Emit the `forall` match. This shouldn't produce any valid `Call` params,
    // because `impl`s are never actually called at runtime.
    auto call_params_id =
        CalleePatternMatch(context, *implicit_param_patterns_id,
                           SemIR::InstBlockId::None, SemIR::InstId::None);
    CARBON_CHECK(call_params_id == SemIR::InstBlockId::Empty ||
                 llvm::all_of(context.inst_blocks().Get(call_params_id),
                              [](SemIR::InstId inst_id) {
                                return inst_id == SemIR::ErrorInst::InstId;
                              }));
  }

  Parse::NodeId first_param_node_id =
      context.node_stack().PopForSoloNodeId<Parse::NodeKind::ImplIntroducer>();
  // Subtracting 1 since we don't want to include the final `{` or `;` of the
  // declaration when performing syntactic match.
  Parse::Tree::PostorderIterator last_param_iter(end_of_decl_node_id);
  --last_param_iter;

  auto pattern_block_id = SemIR::InstBlockId::None;
  if (implicit_param_patterns_id) {
    pattern_block_id = context.pattern_block_stack().Pop();
    context.full_pattern_stack().PopFullPattern();
  }
  return {.name_loc_id = Parse::NodeId::None,
          .name_id = SemIR::NameId::None,
          .first_param_node_id = first_param_node_id,
          .last_param_node_id = *last_param_iter,
          .implicit_params_loc_id = implicit_params_loc_id,
          .implicit_param_patterns_id =
              implicit_param_patterns_id.value_or(SemIR::InstBlockId::None),
          .params_loc_id = Parse::NodeId::None,
          .param_patterns_id = SemIR::InstBlockId::None,
          .call_params_id = SemIR::InstBlockId::None,
          .return_slot_pattern_id = SemIR::InstId::None,
          .pattern_block_id = pattern_block_id};
}

// Build an ImplDecl describing the signature of an impl. This handles the
// common logic shared by impl forward declarations and impl definitions.
static auto BuildImplDecl(Context& context, Parse::AnyImplDeclId node_id,
                          bool is_definition)
    -> std::pair<SemIR::ImplId, SemIR::InstId> {
  auto [constraint_node, constraint_id] =
      context.node_stack().PopExprWithNodeId();
  auto [self_type_node, self_type_inst_id] =
      context.node_stack().PopWithNodeId<Parse::NodeCategory::ImplAs>();
  // Pop the `impl` introducer and any `forall` parameters as a "name".
  auto name = PopImplIntroducerAndParamsAsNameComponent(context, node_id);
  auto decl_block_id = context.inst_block_stack().Pop();

  // Convert the constraint expression to a type.
  auto [constraint_type_inst_id, constraint_type_id] =
      ExprAsType(context, constraint_node, constraint_id);

  // Process modifiers.
  // TODO: Should we somehow permit access specifiers on `impl`s?
  auto introducer =
      context.decl_introducer_state_stack().Pop<Lex::TokenKind::Impl>();
  LimitModifiersOnDecl(context, introducer, KeywordModifierSet::ImplDecl);

  bool is_final = introducer.modifier_set.HasAnyOf(KeywordModifierSet::Final);

  // Finish processing the name, which should be empty, but might have
  // parameters.
  auto name_context = context.decl_name_stack().FinishImplName();
  CARBON_CHECK(name_context.state == DeclNameStack::NameContext::State::Empty);

  // TODO: Check for an orphan `impl`.

  // Add the impl declaration.
  auto impl_decl_id =
      AddPlaceholderInst(context, node_id,
                         SemIR::ImplDecl{.impl_id = SemIR::ImplId::None,
                                         .decl_block_id = decl_block_id});

  SemIR::Impl impl_info = {name_context.MakeEntityWithParamsBase(
                               name, impl_decl_id,
                               /*is_extern=*/false, SemIR::LibraryNameId::None),
                           {.self_id = self_type_inst_id,
                            .constraint_id = constraint_type_inst_id,
                            // This requires that the facet type is identified.
                            .interface = CheckConstraintIsInterface(
                                context, impl_decl_id, constraint_type_inst_id),
                            .is_final = is_final}};

  std::optional<ExtendImplDecl> extend_impl;
  if (introducer.modifier_set.HasAnyOf(KeywordModifierSet::Extend)) {
    extend_impl = ExtendImplDecl{
        .self_type_node_id = self_type_node,
        .constraint_type_id = constraint_type_id,
        .extend_node_id = introducer.modifier_node_id(ModifierOrder::Extend),
    };
  }

  return StartImplDecl(context, SemIR::LocId(node_id),
                       name.implicit_params_loc_id, impl_info, is_definition,
                       extend_impl);
}

auto HandleParseNode(Context& context, Parse::ImplDeclId node_id) -> bool {
  BuildImplDecl(context, node_id, /*is_definition=*/false);
  context.decl_name_stack().PopScope();
  return true;
}

auto HandleParseNode(Context& context, Parse::ImplDefinitionStartId node_id)
    -> bool {
  auto [impl_id, impl_decl_id] =
      BuildImplDecl(context, node_id, /*is_definition=*/true);
  auto& impl_info = context.impls().Get(impl_id);

  CARBON_CHECK(!impl_info.has_definition_started());
  impl_info.definition_id = impl_decl_id;
  impl_info.scope_id =
      context.name_scopes().Add(impl_decl_id, SemIR::NameId::None,
                                context.decl_name_stack().PeekParentScopeId());

  context.scope_stack().PushForEntity(
      impl_decl_id, impl_info.scope_id,
      context.generics().GetSelfSpecific(impl_info.generic_id));
  StartGenericDefinition(context, impl_info.generic_id);
  // This requires that the facet type is complete.
  ImplWitnessStartDefinition(context, impl_info);
  context.inst_block_stack().Push();
  context.node_stack().Push(node_id, impl_id);

  // TODO: Handle the case where there's control flow in the impl body. For
  // example:
  //
  //   impl C as I {
  //     fn F() -> if true then i32 else f64;
  //   }
  //
  // We may need to track a list of instruction blocks here, as we do for a
  // function.
  impl_info.body_block_id = context.inst_block_stack().PeekOrAdd();
  return true;
}

auto HandleParseNode(Context& context, Parse::ImplDefinitionId /*node_id*/)
    -> bool {
  auto impl_id =
      context.node_stack().Pop<Parse::NodeKind::ImplDefinitionStart>();

  FinishImplWitness(context, impl_id);

  auto& impl_info = context.impls().Get(impl_id);
  impl_info.defined = true;
  FinishGenericDefinition(context, impl_info.generic_id);

  context.inst_block_stack().Pop();
  // The decl_name_stack and scopes are popped by `ProcessNodeIds`.
  return true;
}

}  // namespace Carbon::Check

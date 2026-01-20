// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <optional>
#include <utility>

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/impl.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/name_scope.h"
#include "toolchain/check/pattern_match.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/parse/typed_nodes.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/specific_interface.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Returns the implicit `Self` type for an `impl` when it's in a `class`
// declaration.
//
// TODO: Mixin scopes also have a default `Self` type.
static auto GetImplDefaultSelfType(Context& context,
                                   const ClassScope& class_scope)
    -> SemIR::TypeId {
  return context.classes().Get(class_scope.class_decl.class_id).self_type_id;
}

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
  auto self_type = ExprAsType(context, self_node, self_id);

  const auto& introducer = context.decl_introducer_state_stack().innermost();
  if (introducer.modifier_set.HasAnyOf(KeywordModifierSet::Extend)) {
    // TODO: Also handle the parent scope being a mixin.
    if (auto class_scope = TryAsClassScope(
            context, context.decl_name_stack().PeekParentScopeId())) {
      // If we're not inside a class at all, that will be diagnosed against the
      // `extend` elsewhere.
      auto extend_node = introducer.modifier_node_id(ModifierOrder::Extend);
      CARBON_DIAGNOSTIC(ExtendImplSelfAs, Error,
                        "cannot `extend` an `impl` with an explicit self type");
      auto diag = context.emitter().Build(extend_node, ExtendImplSelfAs);

      if (self_type.type_id == GetImplDefaultSelfType(context, *class_scope)) {
        // If the explicit self type is the default, suggest removing it with a
        // diagnostic, but continue as if no error occurred since the self-type
        // is semantically valid.
        CARBON_DIAGNOSTIC(ExtendImplSelfAsDefault, Note,
                          "remove the explicit `Self` type here");
        diag.Note(self_node, ExtendImplSelfAsDefault);
        if (self_type.type_id != SemIR::ErrorInst::TypeId) {
          diag.Emit();
        }
      } else if (self_type.type_id != SemIR::ErrorInst::TypeId) {
        // Otherwise, the self-type is an error.
        diag.Emit();
        self_type.inst_id = SemIR::ErrorInst::TypeInstId;
      }
    }
  }

  // Introduce `Self`. Note that we add this name lexically rather than adding
  // to the `NameScopeId` of the `impl`, because this happens before we enter
  // the `impl` scope or even identify which `impl` we're declaring.
  // TODO: Revisit this once #3714 is resolved.
  AddNameToLookup(context, SemIR::NameId::SelfType, self_type.inst_id);
  context.node_stack().Push(node_id, self_type.inst_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::ImplDefaultSelfAsId node_id)
    -> bool {
  auto self_inst_id = SemIR::TypeInstId::None;

  if (auto class_scope = TryAsClassScope(
          context, context.decl_name_stack().PeekParentScopeId())) {
    auto self_type_id = GetImplDefaultSelfType(context, *class_scope);
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
    auto [call_param_patterns_id, call_params_id] =
        CalleePatternMatch(context, *implicit_param_patterns_id,
                           SemIR::InstBlockId::None, SemIR::InstBlockId::None);
    CARBON_CHECK(call_params_id == SemIR::InstBlockId::Empty);
    CARBON_CHECK(call_param_patterns_id == SemIR::InstBlockId::Empty);
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
          .call_param_patterns_id = SemIR::InstBlockId::None,
          .call_params_id = SemIR::InstBlockId::None,
          .pattern_block_id = pattern_block_id};
}

// Build an ImplDecl describing the signature of an impl. This handles the
// common logic shared by impl forward declarations and impl definitions.
static auto BuildImplDecl(Context& context, Parse::AnyImplDeclId node_id)
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

  // This requires that the facet type is identified. It returns None if an
  // error was diagnosed.
  auto specific_interface = CheckConstraintIsInterface(
      context, impl_decl_id, self_type_inst_id, constraint_type_inst_id);

  auto impl_id = SemIR::ImplId::None;
  {
    SemIR::Impl impl = {name_context.MakeEntityWithParamsBase(
                            name, impl_decl_id,
                            /*is_extern=*/false, SemIR::LibraryNameId::None),
                        {.self_id = self_type_inst_id,
                         .constraint_id = constraint_type_inst_id,
                         .interface = specific_interface,
                         .is_final = is_final}};
    // There's a bunch of places that may represent a diagnostic that occurred
    // in checking the impl up to this point, which we consolidate into this
    // bool. Due to lack of an instruction to set to `ErrorInst`, an
    // `InterfaceId::None` indicates that the interface could not be identified
    // and an error was diagnosed.
    bool impl_had_error =
        context.types().GetTypeIdForTypeInstId(impl.self_id) ==
            SemIR::ErrorInst::TypeId ||
        context.types().GetTypeIdForTypeInstId(impl.constraint_id) ==
            SemIR::ErrorInst::TypeId ||
        !impl.interface.interface_id.has_value();

    CARBON_KIND_SWITCH(FindImplId(context, impl)) {
      case CARBON_KIND(RedeclaredImpl redeclared_impl): {
        // This is a redeclaration of another impl, now held in `impl_id`.
        impl_id = redeclared_impl.prev_impl_id;

        // Note that we don't reconstruct the witness for a redeclaration, which
        // was the instruction that came last in the first declaration's eval
        // block. And FinishGenericRedecl allows the redecl to have fewer
        // instructions to support this case.
        const auto& prev_impl = context.impls().Get(impl_id);
        FinishGenericRedecl(context, prev_impl.generic_id);
        break;
      }
      case CARBON_KIND(NewImpl new_impl): {
        // This is a new declaration (possibly with an attached definition).
        // Create a new `impl_id`, filling the missing generic and witness in
        // `Impl` structure.
        impl_had_error |= new_impl.find_had_error;

        impl.generic_id = BuildGeneric(context, impl_decl_id);

        if (impl_had_error) {
          // If there's any error in the construction of the impl, then the
          // witness can't be constructed. We set it to `ErrorInst` to make the
          // impl unusable for impl lookup.
          impl.witness_id = SemIR::ErrorInst::InstId;
        } else {
          context.inst_block_stack().Push();
          // This makes either a placeholder witness table or a full witness
          // table. The full witness table is deferred to the impl definition
          // unless the declaration uses rewrite constraints to set values of
          // associated constants in the interface.
          //
          // The witness instruction contains the SelfSpecific that is
          // constructed by BuildGeneric(), but the witness and its rewrites
          // also must be part of the generic eval block by coming before
          // FinishGenericDecl().
          impl.witness_id = AddImplWitnessForDeclaration(
              context, node_id, impl,
              context.generics().GetSelfSpecific(impl.generic_id));
          impl.witness_block_id = context.inst_block_stack().Pop();
        }

        FinishGenericDecl(context, node_id, impl.generic_id);

        auto extend_node = introducer.modifier_node_id(ModifierOrder::Extend);
        impl_id = AddImpl(context, impl, new_impl.lookup_bucket, extend_node,
                          name.implicit_params_loc_id);
      }
    }
  }

  // `FindImplId` returned an existing ImplId, or we added a new id with
  // `AddImpl` above. Write that ImplId into the ImplDecl instruction and finish
  // it.
  auto impl_decl = context.insts().GetAs<SemIR::ImplDecl>(impl_decl_id);
  impl_decl.impl_id = impl_id;
  ReplaceInstBeforeConstantUse(context, impl_decl_id, impl_decl);

  return {impl_id, impl_decl_id};
}

auto HandleParseNode(Context& context, Parse::ImplDeclId node_id) -> bool {
  auto [impl_id, impl_decl_id] = BuildImplDecl(context, node_id);
  auto& impl = context.impls().Get(impl_id);

  context.decl_name_stack().PopScope();

  // Impl definitions are required in the same file as the declaration. We skip
  // this requirement if we've already issued an invalid redeclaration error, or
  // there is an error that would prevent the impl from being legal to define.
  if (impl.witness_id != SemIR::ErrorInst::InstId) {
    context.definitions_required_by_decl().push_back(impl_decl_id);
  }

  return true;
}

auto HandleParseNode(Context& context, Parse::ImplDefinitionStartId node_id)
    -> bool {
  auto [impl_id, impl_decl_id] = BuildImplDecl(context, node_id);
  auto& impl = context.impls().Get(impl_id);

  CheckRequireDeclsSatisfied(context, node_id, impl);

  CARBON_CHECK(!impl.has_definition_started());
  impl.definition_id = impl_decl_id;
  impl.scope_id =
      context.name_scopes().Add(impl_decl_id, SemIR::NameId::None,
                                context.decl_name_stack().PeekParentScopeId());

  context.scope_stack().PushForEntity(
      impl_decl_id, impl.scope_id,
      context.generics().GetSelfSpecific(impl.generic_id));
  StartGenericDefinition(context, impl.generic_id);
  ImplWitnessStartDefinition(context, impl);
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
  impl.body_block_id = context.inst_block_stack().PeekOrAdd();
  return true;
}

auto HandleParseNode(Context& context, Parse::ImplDefinitionId /*node_id*/)
    -> bool {
  auto impl_id =
      context.node_stack().Pop<Parse::NodeKind::ImplDefinitionStart>();
  auto& impl = context.impls().Get(impl_id);

  FinishImplWitness(context, impl);

  impl.defined = true;
  FinishGenericDefinition(context, impl.generic_id);

  context.inst_block_stack().Pop();
  // The decl_name_stack and scopes are popped by `ProcessNodeIds`.
  return true;
}

}  // namespace Carbon::Check

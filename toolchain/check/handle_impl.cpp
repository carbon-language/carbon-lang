// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/impl.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/merge.h"
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

  // This might be a generic impl.
  StartGenericDecl(context);
  return true;
}

auto HandleParseNode(Context& context, Parse::ForallId /*node_id*/) -> bool {
  // Push a pattern block for the signature of the `forall`.
  context.pattern_block_stack().Push();
  context.full_pattern_stack().PushFullPattern(
      FullPatternStack::Kind::ImplicitParamList);
  return true;
}

auto HandleParseNode(Context& context, Parse::TypeImplAsId node_id) -> bool {
  auto [self_node, self_id] = context.node_stack().PopExprWithNodeId();
  self_id = ExprAsType(context, self_node, self_id).inst_id;
  context.node_stack().Push(node_id, self_id);

  // Introduce `Self`. Note that we add this name lexically rather than adding
  // to the `NameScopeId` of the `impl`, because this happens before we enter
  // the `impl` scope or even identify which `impl` we're declaring.
  // TODO: Revisit this once #3714 is resolved.
  AddNameToLookup(context, SemIR::NameId::SelfType, self_id);
  return true;
}

// If the specified name scope corresponds to a class, returns the corresponding
// class declaration.
// TODO: Should this be somewhere more central?
static auto TryAsClassScope(Context& context, SemIR::NameScopeId scope_id)
    -> std::optional<SemIR::ClassDecl> {
  if (!scope_id.has_value()) {
    return std::nullopt;
  }
  auto& scope = context.name_scopes().Get(scope_id);
  if (!scope.inst_id().has_value()) {
    return std::nullopt;
  }
  return context.insts().TryGetAs<SemIR::ClassDecl>(scope.inst_id());
}

static auto GetDefaultSelfType(Context& context) -> SemIR::TypeId {
  auto parent_scope_id = context.decl_name_stack().PeekParentScopeId();

  if (auto class_decl = TryAsClassScope(context, parent_scope_id)) {
    return context.classes().Get(class_decl->class_id).self_type_id;
  }

  // TODO: This is also valid in a mixin.

  return SemIR::TypeId::None;
}

auto HandleParseNode(Context& context, Parse::DefaultSelfImplAsId node_id)
    -> bool {
  auto self_type_id = GetDefaultSelfType(context);
  if (!self_type_id.has_value()) {
    CARBON_DIAGNOSTIC(ImplAsOutsideClass, Error,
                      "`impl as` can only be used in a class");
    context.emitter().Emit(node_id, ImplAsOutsideClass);
    self_type_id = SemIR::ErrorInst::SingletonTypeId;
  }

  // Build the implicit access to the enclosing `Self`.
  // TODO: Consider calling `HandleNameAsExpr` to build this implicit `Self`
  // expression. We've already done the work to check that the enclosing context
  // is a class and found its `Self`, so additionally performing an unqualified
  // name lookup would be redundant work, but would avoid duplicating the
  // handling of the `Self` expression.
  auto self_inst_id = AddInst(
      context, node_id,
      SemIR::NameRef{.type_id = SemIR::TypeType::SingletonTypeId,
                     .name_id = SemIR::NameId::SelfType,
                     .value_id = context.types().GetInstId(self_type_id)});

  // There's no need to push `Self` into scope here, because we can find it in
  // the parent class scope.
  context.node_stack().Push(node_id, self_inst_id);
  return true;
}

static auto DiagnoseExtendImplOutsideClass(Context& context,
                                           Parse::AnyImplDeclId node_id)
    -> void {
  CARBON_DIAGNOSTIC(ExtendImplOutsideClass, Error,
                    "`extend impl` can only be used in a class");
  context.emitter().Emit(node_id, ExtendImplOutsideClass);
}

// Process an `extend impl` declaration by extending the impl scope with the
// `impl`'s scope.
static auto ExtendImpl(Context& context, Parse::NodeId extend_node,
                       Parse::AnyImplDeclId node_id, SemIR::ImplId impl_id,
                       Parse::NodeId self_type_node, SemIR::TypeId self_type_id,
                       Parse::NodeId params_node,
                       SemIR::InstId constraint_inst_id,
                       SemIR::TypeId constraint_id) -> bool {
  auto parent_scope_id = context.decl_name_stack().PeekParentScopeId();
  if (!parent_scope_id.has_value()) {
    DiagnoseExtendImplOutsideClass(context, node_id);
    return false;
  }
  // TODO: This is also valid in a mixin.
  if (!TryAsClassScope(context, parent_scope_id)) {
    DiagnoseExtendImplOutsideClass(context, node_id);
    return false;
  }

  auto& parent_scope = context.name_scopes().Get(parent_scope_id);

  if (params_node.has_value()) {
    CARBON_DIAGNOSTIC(ExtendImplForall, Error,
                      "cannot `extend` a parameterized `impl`");
    context.emitter().Emit(extend_node, ExtendImplForall);
    parent_scope.set_has_error();
    return false;
  }

  if (context.parse_tree().node_kind(self_type_node) ==
      Parse::NodeKind::TypeImplAs) {
    CARBON_DIAGNOSTIC(ExtendImplSelfAs, Error,
                      "cannot `extend` an `impl` with an explicit self type");
    auto diag = context.emitter().Build(extend_node, ExtendImplSelfAs);

    // If the explicit self type is not the default, just bail out.
    if (self_type_id != GetDefaultSelfType(context)) {
      diag.Emit();
      parent_scope.set_has_error();
      return false;
    }

    // The explicit self type is the same as the default self type, so suggest
    // removing it and recover as if it were not present.
    if (auto self_as =
            context.parse_tree_and_subtrees().ExtractAs<Parse::TypeImplAs>(
                self_type_node)) {
      CARBON_DIAGNOSTIC(ExtendImplSelfAsDefault, Note,
                        "remove the explicit `Self` type here");
      diag.Note(self_as->type_expr, ExtendImplSelfAsDefault);
    }
    diag.Emit();
  }

  if (!context.types().Is<SemIR::FacetType>(constraint_id)) {
    context.TODO(node_id, "extending non-facet-type constraint");
    parent_scope.set_has_error();
    return false;
  }
  const auto& impl = context.impls().Get(impl_id);
  if (impl.witness_id == SemIR::ErrorInst::SingletonInstId) {
    parent_scope.set_has_error();
  }

  parent_scope.AddExtendedScope(constraint_inst_id);
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
                                return inst_id ==
                                       SemIR::ErrorInst::SingletonInstId;
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

static auto MergeImplRedecl(Context& context, SemIR::Impl& new_impl,
                            SemIR::ImplId prev_impl_id) -> bool {
  auto& prev_impl = context.impls().Get(prev_impl_id);

  // If the parameters aren't the same, then this is not a redeclaration of this
  // `impl`. Keep looking for a prior declaration without issuing a diagnostic.
  if (!CheckRedeclParamsMatch(context, DeclParams(new_impl),
                              DeclParams(prev_impl), SemIR::SpecificId::None,
                              /*diagnose=*/false, /*check_syntax=*/true,
                              /*check_self=*/true)) {
    // NOLINTNEXTLINE(readability-simplify-boolean-expr)
    return false;
  }
  return true;
}

static auto IsValidImplRedecl(Context& context, SemIR::Impl& new_impl,
                              SemIR::ImplId prev_impl_id) -> bool {
  auto& prev_impl = context.impls().Get(prev_impl_id);

  // TODO: Following #3763, disallow redeclarations in different scopes.

  // Following #4672, disallowing defining non-extern declarations in another
  // file.
  if (auto import_ref =
          context.insts().TryGetAs<SemIR::AnyImportRef>(prev_impl.self_id)) {
    // TODO: Handle extern.
    CARBON_DIAGNOSTIC(RedeclImportedImpl, Error,
                      "redeclaration of imported impl");
    // TODO: Note imported declaration
    context.emitter().Emit(new_impl.latest_decl_id(), RedeclImportedImpl);
    return false;
  }

  if (prev_impl.has_definition_started()) {
    // Impls aren't merged in order to avoid generic region lookup into a
    // mismatching table.
    CARBON_DIAGNOSTIC(ImplRedefinition, Error,
                      "redefinition of `impl {0} as {1}`", InstIdAsRawType,
                      InstIdAsRawType);
    CARBON_DIAGNOSTIC(ImplPreviousDefinition, Note,
                      "previous definition was here");
    context.emitter()
        .Build(new_impl.latest_decl_id(), ImplRedefinition, new_impl.self_id,
               new_impl.constraint_id)
        .Note(prev_impl.definition_id, ImplPreviousDefinition)
        .Emit();
    return false;
  }

  // TODO: Only allow redeclaration in a match_first/impl_priority block.

  return true;
}

// Checks that the constraint specified for the impl is valid and complete.
// Returns a pointer to the interface that the impl implements. On error,
// issues a diagnostic and returns nullptr.
static auto CheckConstraintIsInterface(Context& context,
                                       const SemIR::Impl& impl)
    -> const SemIR::CompleteFacetType::RequiredInterface* {
  auto facet_type_id =
      context.types().GetTypeIdForTypeInstId(impl.constraint_id);
  if (facet_type_id == SemIR::ErrorInst::SingletonTypeId) {
    return nullptr;
  }
  auto facet_type = context.types().TryGetAs<SemIR::FacetType>(facet_type_id);
  if (!facet_type) {
    CARBON_DIAGNOSTIC(ImplAsNonFacetType, Error, "impl as non-facet type {0}",
                      InstIdAsType);
    context.emitter().Emit(impl.latest_decl_id(), ImplAsNonFacetType,
                           impl.constraint_id);
    return nullptr;
  }

  auto complete_id = RequireCompleteFacetType(
      context, facet_type_id, context.insts().GetLocId(impl.constraint_id),
      *facet_type, [&] {
        CARBON_DIAGNOSTIC(ImplAsIncompleteFacetType, Error,
                          "impl as incomplete facet type {0}", InstIdAsType);
        return context.emitter().Build(impl.latest_decl_id(),
                                       ImplAsIncompleteFacetType,
                                       impl.constraint_id);
      });
  if (!complete_id.has_value()) {
    return nullptr;
  }
  const auto& complete = context.complete_facet_types().Get(complete_id);
  if (complete.num_to_impl != 1) {
    CARBON_DIAGNOSTIC(ImplOfNotOneInterface, Error,
                      "impl as {0} interfaces, expected 1", int);
    context.emitter().Emit(impl.latest_decl_id(), ImplOfNotOneInterface,
                           complete.num_to_impl);
    return nullptr;
  }
  return &complete.required_interfaces.front();
}

// Build an ImplDecl describing the signature of an impl. This handles the
// common logic shared by impl forward declarations and impl definitions.
static auto BuildImplDecl(Context& context, Parse::AnyImplDeclId node_id,
                          bool is_definition)
    -> std::pair<SemIR::ImplId, SemIR::InstId> {
  auto [constraint_node, constraint_id] =
      context.node_stack().PopExprWithNodeId();
  auto [self_type_node, self_inst_id] =
      context.node_stack().PopWithNodeId<Parse::NodeCategory::ImplAs>();
  auto self_type_id = context.types().GetTypeIdForTypeInstId(self_inst_id);
  // Pop the `impl` introducer and any `forall` parameters as a "name".
  auto name = PopImplIntroducerAndParamsAsNameComponent(context, node_id);
  auto decl_block_id = context.inst_block_stack().Pop();

  // Convert the constraint expression to a type.
  auto [constraint_inst_id, constraint_type_id] =
      ExprAsType(context, constraint_node, constraint_id);

  // Process modifiers.
  // TODO: Should we somehow permit access specifiers on `impl`s?
  // TODO: Handle `final` modifier.
  auto introducer =
      context.decl_introducer_state_stack().Pop<Lex::TokenKind::Impl>();
  LimitModifiersOnDecl(context, introducer, KeywordModifierSet::ImplDecl);

  // Finish processing the name, which should be empty, but might have
  // parameters.
  auto name_context = context.decl_name_stack().FinishImplName();
  CARBON_CHECK(name_context.state == DeclNameStack::NameContext::State::Empty);

  // TODO: Check for an orphan `impl`.

  // Add the impl declaration.
  SemIR::ImplDecl impl_decl = {.impl_id = SemIR::ImplId::None,
                               .decl_block_id = decl_block_id};
  auto impl_decl_id = AddPlaceholderInst(context, node_id, impl_decl);

  SemIR::Impl impl_info = {name_context.MakeEntityWithParamsBase(
                               name, impl_decl_id,
                               /*is_extern=*/false, SemIR::LibraryNameId::None),
                           {.self_id = self_inst_id,
                            .constraint_id = constraint_inst_id,
                            .interface = SemIR::SpecificInterface::None}};

  const SemIR::CompleteFacetType::RequiredInterface* required_interface =
      CheckConstraintIsInterface(context, impl_info);
  if (required_interface) {
    impl_info.interface = *required_interface;
  }

  // Add the impl declaration.
  bool invalid_redeclaration = false;
  auto lookup_bucket_ref = context.impls().GetOrAddLookupBucket(impl_info);
  // TODO: Detect two impl declarations with the same self type and interface,
  // and issue an error if they don't match.
  for (auto prev_impl_id : lookup_bucket_ref) {
    if (MergeImplRedecl(context, impl_info, prev_impl_id)) {
      if (IsValidImplRedecl(context, impl_info, prev_impl_id)) {
        impl_decl.impl_id = prev_impl_id;
      } else {
        // IsValidImplRedecl() has issued a diagnostic, avoid generating more
        // diagnostics for this declaration.
        invalid_redeclaration = true;
      }
      break;
    }
  }

  // Create a new impl if this isn't a valid redeclaration.
  if (!impl_decl.impl_id.has_value()) {
    impl_info.generic_id = BuildGeneric(context, impl_decl_id);
    if (required_interface) {
      impl_info.witness_id = ImplWitnessForDeclaration(context, impl_info);
    } else {
      impl_info.witness_id = SemIR::ErrorInst::SingletonInstId;
      // TODO: We might also want to mark that the name scope for the impl has
      // an error -- at least once we start making name lookups within the impl
      // also look into the facet (eg, so you can name associated constants from
      // within the impl).
    }
    FinishGenericDecl(context, impl_decl_id, impl_info.generic_id);
    impl_decl.impl_id = context.impls().Add(impl_info);
    lookup_bucket_ref.push_back(impl_decl.impl_id);
  } else {
    const auto& first_impl = context.impls().Get(impl_decl.impl_id);
    FinishGenericRedecl(context, impl_decl_id, first_impl.generic_id);
  }

  // Write the impl ID into the ImplDecl.
  ReplaceInstBeforeConstantUse(context, impl_decl_id, impl_decl);

  // For an `extend impl` declaration, mark the impl as extending this `impl`.
  if (self_type_id != SemIR::ErrorInst::SingletonTypeId &&
      introducer.modifier_set.HasAnyOf(KeywordModifierSet::Extend)) {
    auto extend_node = introducer.modifier_node_id(ModifierOrder::Decl);
    if (impl_info.generic_id.has_value()) {
      SemIR::TypeId type_id = context.insts().Get(constraint_inst_id).type_id();
      constraint_inst_id = AddInst<SemIR::SpecificConstant>(
          context, context.insts().GetLocId(constraint_inst_id),
          {.type_id = type_id,
           .inst_id = constraint_inst_id,
           .specific_id =
               context.generics().GetSelfSpecific(impl_info.generic_id)});
    }
    if (!ExtendImpl(context, extend_node, node_id, impl_decl.impl_id,
                    self_type_node, self_type_id, name.implicit_params_loc_id,
                    constraint_inst_id, constraint_type_id)) {
      // Don't allow the invalid impl to be used.
      context.impls().Get(impl_decl.impl_id).witness_id =
          SemIR::ErrorInst::SingletonInstId;
    }
  }

  // Impl definitions are required in the same file as the declaration. We skip
  // this requirement if we've already issued an invalid redeclaration error.
  if (!is_definition && !invalid_redeclaration) {
    context.definitions_required().push_back(impl_decl_id);
  }

  return {impl_decl.impl_id, impl_decl_id};
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

  context.scope_stack().Push(
      impl_decl_id, impl_info.scope_id,
      context.generics().GetSelfSpecific(impl_info.generic_id));
  StartGenericDefinition(context);
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

  auto& impl_info = context.impls().Get(impl_id);
  CARBON_CHECK(!impl_info.is_defined());
  FinishImplWitness(context, impl_info);
  impl_info.defined = true;
  FinishGenericDefinition(context, impl_info.generic_id);

  context.inst_block_stack().Pop();
  // The decl_name_stack and scopes are popped by `ProcessNodeIds`.
  return true;
}

}  // namespace Carbon::Check

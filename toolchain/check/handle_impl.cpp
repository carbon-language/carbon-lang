// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/impl.h"
#include "toolchain/check/merge.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/parse/typed_nodes.h"
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

  // Push a pattern block for the signature of the `forall` (if any).
  // TODO: Instead use a separate parse node kinds for `impl` and `impl forall`,
  // and only push a pattern block in `forall` case.
  context.pattern_block_stack().Push();
  return true;
}

auto HandleParseNode(Context& context, Parse::ImplForallId node_id) -> bool {
  auto params_id =
      context.node_stack().Pop<Parse::NodeKind::ImplicitParamList>();
  context.node_stack()
      .PopAndDiscardSoloNodeId<Parse::NodeKind::ImplicitParamListStart>();
  RequireGenericParamsOnType(context, params_id);
  context.node_stack().Push(node_id, params_id);
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
  context.AddNameToLookup(SemIR::NameId::SelfType, self_id);
  return true;
}

// If the specified name scope corresponds to a class, returns the corresponding
// class declaration.
// TODO: Should this be somewhere more central?
static auto TryAsClassScope(Context& context, SemIR::NameScopeId scope_id)
    -> std::optional<SemIR::ClassDecl> {
  if (!scope_id.is_valid()) {
    return std::nullopt;
  }
  auto& scope = context.name_scopes().Get(scope_id);
  if (!scope.inst_id.is_valid()) {
    return std::nullopt;
  }
  return context.insts().TryGetAs<SemIR::ClassDecl>(scope.inst_id);
}

static auto GetDefaultSelfType(Context& context) -> SemIR::TypeId {
  auto parent_scope_id = context.decl_name_stack().PeekParentScopeId();

  if (auto class_decl = TryAsClassScope(context, parent_scope_id)) {
    return context.classes().Get(class_decl->class_id).self_type_id;
  }

  // TODO: This is also valid in a mixin.

  return SemIR::TypeId::Invalid;
}

auto HandleParseNode(Context& context, Parse::DefaultSelfImplAsId node_id)
    -> bool {
  auto self_type_id = GetDefaultSelfType(context);
  if (!self_type_id.is_valid()) {
    CARBON_DIAGNOSTIC(ImplAsOutsideClass, Error,
                      "`impl as` can only be used in a class");
    context.emitter().Emit(node_id, ImplAsOutsideClass);
    self_type_id = SemIR::TypeId::Error;
  }

  // Build the implicit access to the enclosing `Self`.
  // TODO: Consider calling `HandleNameAsExpr` to build this implicit `Self`
  // expression. We've already done the work to check that the enclosing context
  // is a class and found its `Self`, so additionally performing an unqualified
  // name lookup would be redundant work, but would avoid duplicating the
  // handling of the `Self` expression.
  auto self_inst_id = context.AddInst(
      node_id,
      SemIR::NameRef{.type_id = SemIR::TypeId::TypeType,
                     .name_id = SemIR::NameId::SelfType,
                     .value_id = context.types().GetInstId(self_type_id)});

  // There's no need to push `Self` into scope here, because we can find it in
  // the parent class scope.
  context.node_stack().Push(node_id, self_inst_id);
  return true;
}

// Process an `extend impl` declaration by extending the impl scope with the
// `impl`'s scope.
static auto ExtendImpl(Context& context, Parse::NodeId extend_node,
                       Parse::AnyImplDeclId node_id,
                       Parse::NodeId self_type_node, SemIR::TypeId self_type_id,
                       Parse::NodeId params_node, SemIR::TypeId constraint_id)
    -> void {
  auto parent_scope_id = context.decl_name_stack().PeekParentScopeId();
  auto& parent_scope = context.name_scopes().Get(parent_scope_id);

  // TODO: This is also valid in a mixin.
  if (!TryAsClassScope(context, parent_scope_id)) {
    CARBON_DIAGNOSTIC(ExtendImplOutsideClass, Error,
                      "`extend impl` can only be used in a class");
    context.emitter().Emit(node_id, ExtendImplOutsideClass);
    return;
  }

  if (params_node.is_valid()) {
    CARBON_DIAGNOSTIC(ExtendImplForall, Error,
                      "cannot `extend` a parameterized `impl`");
    context.emitter().Emit(extend_node, ExtendImplForall);
    parent_scope.has_error = true;
    return;
  }

  if (context.parse_tree().node_kind(self_type_node) ==
      Parse::NodeKind::TypeImplAs) {
    CARBON_DIAGNOSTIC(ExtendImplSelfAs, Error,
                      "cannot `extend` an `impl` with an explicit self type");
    auto diag = context.emitter().Build(extend_node, ExtendImplSelfAs);

    // If the explicit self type is not the default, just bail out.
    if (self_type_id != GetDefaultSelfType(context)) {
      diag.Emit();
      parent_scope.has_error = true;
      return;
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

  auto interface_type =
      context.types().TryGetAs<SemIR::InterfaceType>(constraint_id);
  if (!interface_type) {
    context.TODO(node_id, "extending non-interface constraint");
    parent_scope.has_error = true;
    return;
  }

  auto& interface = context.interfaces().Get(interface_type->interface_id);
  if (!interface.is_defined()) {
    CARBON_DIAGNOSTIC(ExtendUndefinedInterface, Error,
                      "`extend impl` requires a definition for interface `{0}`",
                      SemIR::TypeId);
    auto diag = context.emitter().Build(node_id, ExtendUndefinedInterface,
                                        constraint_id);
    context.NoteUndefinedInterface(interface_type->interface_id, diag);
    diag.Emit();
    parent_scope.has_error = true;
    return;
  }

  parent_scope.extended_scopes.push_back(interface.scope_id);
}

// Pops the parameters of an `impl`, forming a `NameComponent` with no
// associated name that describes them.
static auto PopImplIntroducerAndParamsAsNameComponent(
    Context& context, Parse::AnyImplDeclId end_of_decl_node_id)
    -> NameComponent {
  auto [implicit_params_loc_id, implicit_params_id] =
      context.node_stack().PopWithNodeIdIf<Parse::NodeKind::ImplForall>();

  Parse::NodeId first_param_node_id =
      context.node_stack().PopForSoloNodeId<Parse::NodeKind::ImplIntroducer>();
  Parse::NodeId last_param_node_id = end_of_decl_node_id;

  return {
      .name_loc_id = Parse::NodeId::Invalid,
      .name_id = SemIR::NameId::Invalid,
      .first_param_node_id = first_param_node_id,
      .last_param_node_id = last_param_node_id,
      .implicit_params_loc_id = implicit_params_loc_id,
      .implicit_params_id =
          implicit_params_id.value_or(SemIR::InstBlockId::Invalid),
      .params_loc_id = Parse::NodeId::Invalid,
      .params_id = SemIR::InstBlockId::Invalid,
      .pattern_block_id = context.pattern_block_stack().Pop(),
  };
}

static auto MergeImplRedecl(Context& context, SemIR::Impl& new_impl,
                            SemIR::ImplId prev_impl_id) -> bool {
  auto& prev_impl = context.impls().Get(prev_impl_id);

  // TODO: Following #3763, disallow redeclarations in different scopes.

  // If the parameters aren't the same, then this is not a redeclaration of this
  // `impl`. Keep looking for a prior declaration without issuing a diagnostic.
  if (!CheckRedeclParamsMatch(context, DeclParams(new_impl),
                              DeclParams(prev_impl), SemIR::SpecificId::Invalid,
                              /*check_syntax=*/true, /*diagnose=*/false)) {
    // NOLINTNEXTLINE(readability-simplify-boolean-expr)
    return false;
  }

  // TODO: CheckIsAllowedRedecl. We don't have a suitable NameId; decide if we
  // need to treat the `T as I` as a kind of name.

  // TODO: Merge information from the new declaration into the old one as
  // needed.

  return true;
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
  auto self_type_id = context.GetTypeIdForTypeInst(self_inst_id);
  // Pop the `impl` introducer and any `forall` parameters as a "name".
  auto name = PopImplIntroducerAndParamsAsNameComponent(context, node_id);
  auto decl_block_id = context.inst_block_stack().Pop();

  // Convert the constraint expression to a type.
  // TODO: Check that its constant value is a constraint.
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
  SemIR::ImplDecl impl_decl = {.impl_id = SemIR::ImplId::Invalid,
                               .decl_block_id = decl_block_id};
  auto impl_decl_id =
      context.AddPlaceholderInst(SemIR::LocIdAndInst(node_id, impl_decl));

  SemIR::Impl impl_info = {
      name_context.MakeEntityWithParamsBase(name, impl_decl_id,
                                            /*is_extern=*/false,
                                            SemIR::LibraryNameId::Invalid),
      {.self_id = self_inst_id, .constraint_id = constraint_inst_id}};

  // Add the impl declaration.
  auto lookup_bucket_ref = context.impls().GetOrAddLookupBucket(impl_info);
  for (auto prev_impl_id : lookup_bucket_ref) {
    if (MergeImplRedecl(context, impl_info, prev_impl_id)) {
      impl_decl.impl_id = prev_impl_id;
      break;
    }
  }

  // Create a new impl if this isn't a valid redeclaration.
  if (!impl_decl.impl_id.is_valid()) {
    impl_info.generic_id = FinishGenericDecl(context, impl_decl_id);
    impl_decl.impl_id = context.impls().Add(impl_info);
    lookup_bucket_ref.push_back(impl_decl.impl_id);
  } else {
    FinishGenericRedecl(context, impl_decl_id,
                        context.impls().Get(impl_decl.impl_id).generic_id);
  }

  // Write the impl ID into the ImplDecl.
  context.ReplaceInstBeforeConstantUse(impl_decl_id, impl_decl);

  // For an `extend impl` declaration, mark the impl as extending this `impl`.
  if (introducer.modifier_set.HasAnyOf(KeywordModifierSet::Extend)) {
    auto extend_node = introducer.modifier_node_id(ModifierOrder::Decl);
    ExtendImpl(context, extend_node, node_id, self_type_node, self_type_id,
               name.implicit_params_loc_id, constraint_type_id);
  }

  if (!is_definition && context.IsImplFile()) {
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

  if (impl_info.is_defined()) {
    CARBON_DIAGNOSTIC(ImplRedefinition, Error,
                      "redefinition of `impl {0} as {1}`", std::string,
                      std::string);
    CARBON_DIAGNOSTIC(ImplPreviousDefinition, Note,
                      "previous definition was here");
    context.emitter()
        .Build(node_id, ImplRedefinition,
               context.sem_ir().StringifyTypeExpr(impl_info.self_id),
               context.sem_ir().StringifyTypeExpr(impl_info.constraint_id))
        .Note(impl_info.definition_id, ImplPreviousDefinition)
        .Emit();
  } else {
    impl_info.definition_id = impl_decl_id;
    impl_info.scope_id = context.name_scopes().Add(
        impl_decl_id, SemIR::NameId::Invalid,
        context.decl_name_stack().PeekParentScopeId());
  }

  context.scope_stack().Push(impl_decl_id, impl_info.scope_id);

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

  if (!context.impls().Get(impl_id).is_defined()) {
    context.impls().Get(impl_id).witness_id =
        BuildImplWitness(context, impl_id);
  }

  context.inst_block_stack().Pop();
  // The decl_name_stack and scopes are popped by `ProcessNodeIds`.
  return true;
}

}  // namespace Carbon::Check

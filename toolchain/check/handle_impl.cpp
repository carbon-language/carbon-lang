// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/parse/typed_nodes.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto HandleImplIntroducer(Context& context, Parse::ImplIntroducerId parse_node)
    -> bool {
  // Create an instruction block to hold the instructions created for the type
  // and interface.
  context.inst_block_stack().Push();

  // Push the bracketing node.
  context.node_stack().Push(parse_node);

  // Optional modifiers follow.
  context.decl_state_stack().Push(DeclState::Impl);

  // An impl doesn't have a name per se, but it makes the processing more
  // consistent to imagine that it does. This also gives us a scope for implicit
  // parameters.
  context.decl_name_stack().PushScopeAndStartName();
  return true;
}

auto HandleImplForall(Context& context, Parse::ImplForallId parse_node)
    -> bool {
  auto params_id =
      context.node_stack().Pop<Parse::NodeKind::ImplicitParamList>();
  context.node_stack().Push(parse_node, params_id);
  return true;
}

auto HandleTypeImplAs(Context& context, Parse::TypeImplAsId parse_node)
    -> bool {
  auto [self_node, self_id] = context.node_stack().PopExprWithParseNode();
  auto self_type_id = ExprAsType(context, self_node, self_id);
  context.node_stack().Push(parse_node, self_type_id);
  // TODO: `Self` should come into scope here, at least if it's not already in
  // scope. Check the design for the latter case.
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
  auto enclosing_scope_id = context.decl_name_stack().PeekTargetScope();

  if (auto class_decl = TryAsClassScope(context, enclosing_scope_id)) {
    return context.classes().Get(class_decl->class_id).self_type_id;
  }

  // TODO: This is also valid in a mixin.

  return SemIR::TypeId::Invalid;
}

auto HandleDefaultSelfImplAs(Context& context,
                             Parse::DefaultSelfImplAsId parse_node) -> bool {
  auto self_type_id = GetDefaultSelfType(context);
  if (!self_type_id.is_valid()) {
    CARBON_DIAGNOSTIC(ImplAsOutsideClass, Error,
                      "`impl as` can only be used in a class.");
    context.emitter().Emit(parse_node, ImplAsOutsideClass);
    self_type_id = SemIR::TypeId::Error;
  }

  context.node_stack().Push(parse_node, self_type_id);
  return true;
}

// Process an `extend impl` declaration by extending the impl scope with the
// `impl`'s scope.
static auto ExtendImpl(Context& context, Parse::NodeId extend_node,
                       Parse::AnyImplDeclId parse_node,
                       Parse::NodeId self_type_node, SemIR::TypeId self_type_id,
                       Parse::NodeId params_node,
                       SemIR::TypeId constraint_id) -> void {
  auto enclosing_scope_id = context.decl_name_stack().PeekTargetScope();
  auto& enclosing_scope = context.name_scopes().Get(enclosing_scope_id);

  // TODO: This is also valid in a mixin.
  if (!TryAsClassScope(context, enclosing_scope_id)) {
    CARBON_DIAGNOSTIC(ExtendImplOutsideClass, Error,
                      "`extend impl` can only be used in a class.");
    context.emitter().Emit(parse_node, ExtendImplOutsideClass);
    return;
  }

  if (params_node.is_valid()) {
    CARBON_DIAGNOSTIC(ExtendImplForall, Error,
                      "Cannot `extend` a parameterized `impl`.");
    context.emitter().Emit(extend_node, ExtendImplForall);
    enclosing_scope.has_error = true;
    return;
  }

  if (context.parse_tree().node_kind(self_type_node) ==
      Parse::NodeKind::TypeImplAs) {
    CARBON_DIAGNOSTIC(ExtendImplSelfAs, Error,
                      "Cannot `extend` an `impl` with an explicit self type.");
    CARBON_DIAGNOSTIC(ExtendImplSelfAsDefault, Note,
                      "Remove the explicit `Self` type here.");
    auto diag = context.emitter().Build(extend_node, ExtendImplSelfAs);
    if (self_type_id == GetDefaultSelfType(context)) {
      // If the explicit self type is the default, suggest removing it and
      // recover.
      if (auto self_as = context.parse_tree().ExtractAs<Parse::TypeImplAs>(
              self_type_node)) {
        diag.Note(self_as->type_expr, ExtendImplSelfAsDefault);
      }
      diag.Emit();
    } else {
      diag.Emit();
      enclosing_scope.has_error = true;
      return;
    }
  }

  auto interface_type =
      context.types().TryGetAs<SemIR::InterfaceType>(constraint_id);
  if (!interface_type) {
    context.TODO(parse_node, "extending non-interface constraint");
    enclosing_scope.has_error = true;
    return;
  }

  auto& interface = context.interfaces().Get(interface_type->interface_id);
  if (!interface.is_defined()) {
    CARBON_DIAGNOSTIC(
        ExtendUndefinedInterface, Error,
        "`extend impl` requires a definition for interface `{0}`.",
        SemIR::TypeId);
    auto diag = context.emitter().Build(parse_node, ExtendUndefinedInterface,
                                        constraint_id);
    context.NoteUndefinedInterface(interface_type->interface_id, diag);
    diag.Emit();
    enclosing_scope.has_error = true;
    return;
  }

  enclosing_scope.extended_scopes.push_back(interface.scope_id);
}

// Build an ImplDecl describing the signature of an impl. This handles the
// common logic shared by impl forward declarations and impl definitions.
static auto BuildImplDecl(Context& context, Parse::AnyImplDeclId parse_node)
    -> std::pair<SemIR::ImplId, SemIR::InstId> {
  auto [constraint_node, constraint_id] =
      context.node_stack().PopExprWithParseNode();
  auto [self_type_node, self_type_id] =
      context.node_stack().PopWithParseNode<Parse::NodeCategory::ImplAs>();
  auto [params_node, params_id] =
      context.node_stack().PopWithParseNodeIf<Parse::NodeKind::ImplForall>();
  auto decl_block_id = context.inst_block_stack().Pop();
  context.node_stack().PopForSoloParseNode<Parse::NodeKind::ImplIntroducer>();

  // Convert the constraint expression to a type.
  // TODO: Check that its constant value is a constraint.
  auto constraint_type_id = ExprAsType(context, constraint_node, constraint_id);

  // Process modifiers.
  // TODO: Should we somehow permit access specifiers on `impl`s?
  // TODO: Handle `final` modifier.
  LimitModifiersOnDecl(context, KeywordModifierSet::ImplDecl,
                       Lex::TokenKind::Impl);

  // Finish processing the name, which should be empty, but might have
  // parameters.
  auto name_context = context.decl_name_stack().FinishImplName();
  CARBON_CHECK(name_context.state == DeclNameStack::NameContext::State::Empty);

  // TODO: Check for an orphan `impl`.

  // TODO: Check parameters. Store them on the `Impl` in some form.
  static_cast<void>(params_id);

  // Add the impl declaration.
  // TODO: Does lookup in an impl file need to look for a prior impl declaration
  // in the api file?
  auto impl_id = context.impls().LookupOrAdd(self_type_id, constraint_type_id);
  auto impl_decl = SemIR::ImplDecl{impl_id, decl_block_id};
  auto impl_decl_id = context.AddInst({parse_node, impl_decl});

  // For an `extend impl` declaration, mark the impl as extending this `impl`.
  if (!!(context.decl_state_stack().innermost().modifier_set &
         KeywordModifierSet::Extend)) {
    auto extend_node = context.decl_state_stack().innermost().saw_decl_modifier;
    ExtendImpl(context, extend_node, parse_node, self_type_node, self_type_id,
               params_node, constraint_type_id);
  }

  context.decl_state_stack().Pop(DeclState::Impl);

  return {impl_decl.impl_id, impl_decl_id};
}

auto HandleImplDecl(Context& context, Parse::ImplDeclId parse_node) -> bool {
  BuildImplDecl(context, parse_node);
  context.decl_name_stack().PopScope();
  return true;
}

auto HandleImplDefinitionStart(Context& context,
                               Parse::ImplDefinitionStartId parse_node)
    -> bool {
  auto [impl_id, impl_decl_id] = BuildImplDecl(context, parse_node);
  auto& impl_info = context.impls().Get(impl_id);

  if (impl_info.definition_id.is_valid()) {
    CARBON_DIAGNOSTIC(ImplRedefinition, Error,
                      "Redefinition of `impl {0} as {1}`.", SemIR::TypeId,
                      SemIR::TypeId);
    CARBON_DIAGNOSTIC(ImplPreviousDefinition, Note,
                      "Previous definition was here.");
    context.emitter()
        .Build(parse_node, ImplRedefinition, impl_info.self_id,
               impl_info.constraint_id)
        .Note(impl_info.definition_id, ImplPreviousDefinition)
        .Emit();
  } else {
    impl_info.definition_id = impl_decl_id;
    impl_info.scope_id =
        context.name_scopes().Add(impl_decl_id, SemIR::NameId::Invalid,
                                  context.decl_name_stack().PeekTargetScope());
  }

  context.scope_stack().Push(impl_decl_id, impl_info.scope_id);

  context.inst_block_stack().Push();
  context.node_stack().Push(parse_node, impl_id);

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

auto HandleImplDefinition(Context& context,
                          Parse::ImplDefinitionId /*parse_node*/) -> bool {
  auto impl_id =
      context.node_stack().Pop<Parse::NodeKind::ImplDefinitionStart>();
  context.inst_block_stack().Pop();
  context.decl_name_stack().PopScope();

  // The impl is now fully defined.
  context.impls().Get(impl_id).defined = true;
  return true;
}

}  // namespace Carbon::Check

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/lex/token_index.h"

namespace Carbon::Check {

auto HandleVariableIntroducer(Context& context,
                              Parse::VariableIntroducerId node_id) -> bool {
  // No action, just a bracketing node.
  context.node_stack().Push(node_id);
  context.decl_introducer_state_stack().Push<Lex::TokenKind::Var>();
  return true;
}

auto HandleReturnedModifier(Context& context, Parse::ReturnedModifierId node_id)
    -> bool {
  // No action, just a bracketing node.
  context.node_stack().Push(node_id);
  return true;
}

auto HandleVariableInitializer(Context& context,
                               Parse::VariableInitializerId node_id) -> bool {
  if (context.scope_stack().PeekIndex() == ScopeIndex::Package) {
    context.inst_block_stack().PushGlobalInit();
  }
  context.node_stack().Push(node_id);
  return true;
}

auto HandleVariableDecl(Context& context, Parse::VariableDeclId node_id)
    -> bool {
  // Handle the optional initializer.
  std::optional<SemIR::InstId> init_id;
  if (context.node_stack().PeekNextIs<Parse::NodeKind::VariableInitializer>()) {
    init_id = context.node_stack().PopExpr();
    context.node_stack()
        .PopAndDiscardSoloNodeId<Parse::NodeKind::VariableInitializer>();
  }

  if (context.node_stack().PeekIs<Parse::NodeKind::TuplePattern>()) {
    if (init_id && context.scope_stack().PeekIndex() == ScopeIndex::Package) {
      context.inst_block_stack().PopGlobalInit();
    }
    return context.TODO(node_id, "tuple pattern in var");
  }

  auto value_id = context.node_stack().PopPattern();

  // Pop the `returned` specifier if present.
  context.node_stack()
      .PopAndDiscardSoloNodeIdIf<Parse::NodeKind::ReturnedModifier>();

  context.node_stack()
      .PopAndDiscardSoloNodeId<Parse::NodeKind::VariableIntroducer>();

  // Process declaration modifiers.
  // TODO: For a qualified `var` declaration, this should use the target scope
  // of the name introduced in the declaration. See #2590.
  auto [_, parent_scope_inst] = context.name_scopes().GetInstIfValid(
      context.scope_stack().PeekNameScopeId());
  auto introducer =
      context.decl_introducer_state_stack().Pop<Lex::TokenKind::Var>();
  CheckAccessModifiersOnDecl(context, introducer, parent_scope_inst);
  LimitModifiersOnDecl(context, introducer, KeywordModifierSet::Access);

  // Extract the name binding.
  if (auto bind_name = context.insts().TryGetAs<SemIR::AnyBindName>(value_id)) {
    // Form a corresponding name in the current context, and bind the name to
    // the variable.
    auto name_context = context.decl_name_stack().MakeUnqualifiedName(
        context.insts().GetLocId(value_id),
        context.bind_names().Get(bind_name->bind_name_id).name_id);
    context.decl_name_stack().AddNameOrDiagnoseDuplicate(
        name_context, value_id, introducer.modifier_set.GetAccessKind());
    value_id = bind_name->value_id;
  } else if (auto field_decl =
                 context.insts().TryGetAs<SemIR::FieldDecl>(value_id)) {
    // Introduce the field name into the class.
    auto name_context = context.decl_name_stack().MakeUnqualifiedName(
        context.insts().GetLocId(value_id), field_decl->name_id);
    context.decl_name_stack().AddNameOrDiagnoseDuplicate(
        name_context, value_id, introducer.modifier_set.GetAccessKind());
  }
  // TODO: Handle other kinds of pattern.

  // If there was an initializer, assign it to the storage.
  if (init_id) {
    if (context.GetCurrentScopeAs<SemIR::ClassDecl>()) {
      // TODO: In a class scope, we should instead save the initializer
      // somewhere so that we can use it as a default.
      context.TODO(node_id, "Field initializer");
    } else {
      init_id = Initialize(context, node_id, value_id, *init_id);
      // TODO: Consider using different instruction kinds for assignment versus
      // initialization.
      context.AddInst<SemIR::Assign>(node_id,
                                     {.lhs_id = value_id, .rhs_id = *init_id});
    }
    if (context.scope_stack().PeekIndex() == ScopeIndex::Package) {
      context.inst_block_stack().PopGlobalInit();
    }
  }

  return true;
}

}  // namespace Carbon::Check

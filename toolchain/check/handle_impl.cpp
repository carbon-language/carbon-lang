// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

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

  // Create a scope for implicit parameters. We may not use it, but it's simpler
  // to create it unconditionally than to track whether it exists.
  context.scope_stack().Push();
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
  auto self_id = context.node_stack().PopExpr();
  context.node_stack().Push(parse_node, self_id);
  return true;
}

auto HandleDefaultSelfImplAs(Context& /*context*/,
                             Parse::DefaultSelfImplAsId /*parse_node*/)
    -> bool {
  // TODO: Determine self_id and push it onto node stack.
  return true;
}

static auto BuildImplDecl(Context& context, Parse::AnyImplDeclId /*parse_node*/)
    -> SemIR::InstId {
  auto interface_id = context.node_stack().PopExpr();
  auto self_id = context.node_stack().PopIf<Parse::NodeKind::TypeImplAs>();
  auto params_id = context.node_stack().PopIf<Parse::NodeKind::ImplForall>();
  auto decl_block_id = context.inst_block_stack().Pop();
  context.node_stack()
      .PopAndDiscardSoloParseNode<Parse::NodeKind::ImplIntroducer>();

  // TODO: Build an `Impl` object.
  static_cast<void>(decl_block_id);
  static_cast<void>(params_id);
  static_cast<void>(self_id);
  static_cast<void>(interface_id);

  return SemIR::InstId::Invalid;
}

auto HandleImplDecl(Context& context, Parse::ImplDeclId parse_node) -> bool {
  BuildImplDecl(context, parse_node);
  context.scope_stack().Pop();
  return true;
}

auto HandleImplDefinitionStart(Context& context,
                               Parse::ImplDefinitionStartId parse_node)
    -> bool {
  auto impl_decl_id = BuildImplDecl(context, parse_node);
  auto enclosing_scope_id = SemIR::NameScopeId::Invalid;
  auto scope_id = context.name_scopes().Add(
      impl_decl_id, SemIR::NameId::Invalid, enclosing_scope_id);
  context.scope_stack().Push(impl_decl_id, scope_id);
  return true;
}

auto HandleImplDefinition(Context& context,
                          Parse::ImplDefinitionId /*parse_node*/) -> bool {
  context.scope_stack().Pop();
  context.scope_stack().Pop();
  return true;
}

}  // namespace Carbon::Check

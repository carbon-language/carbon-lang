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
  context.PushScope();
  return true;
}

auto HandleImplForall(Context& context, Parse::ImplForallId parse_node)
    -> bool {
  auto params_id =
      context.node_stack().Pop<Parse::NodeKind::ImplicitParamList>();
  context.node_stack().Push(parse_node, params_id);
  return true;
}

auto HandleImplAs(Context& /*context*/, Parse::ImplAsId /*parse_node*/)
    -> bool {
  return true;
}

static auto BuildImplDecl(Context& context, Parse::AnyImplDeclId /*parse_node*/)
    -> SemIR::InstId {
  auto interface_id = context.node_stack().PopExpr();
  auto self_id = SemIR::InstId::Invalid;
  if (!context.node_stack().PeekIs<Parse::NodeKind::ImplForall>() &&
      !context.node_stack().PeekIs<Parse::NodeKind::ImplIntroducer>()) {
    self_id = context.node_stack().PopExpr();
  }

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
  context.PopScope();
  return true;
}

auto HandleImplDefinitionStart(Context& context,
                               Parse::ImplDefinitionStartId parse_node)
    -> bool {
  auto impl_decl_id = BuildImplDecl(context, parse_node);
  auto enclosing_scope_id = SemIR::NameScopeId::Invalid;
  auto scope_id = context.name_scopes().Add(
      impl_decl_id, SemIR::NameId::Invalid, enclosing_scope_id);
  context.PushScope(impl_decl_id, scope_id);
  return true;
}

auto HandleImplDefinition(Context& context,
                          Parse::ImplDefinitionId /*parse_node*/) -> bool {
  context.PopScope();
  context.PopScope();
  return true;
}

}  // namespace Carbon::Check

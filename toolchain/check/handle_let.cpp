// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/modifiers_allowed.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

auto HandleLetDecl(Context& context, Parse::Node parse_node) -> bool {
  auto value_id = context.node_stack().PopExpr();
  SemIR::InstId pattern_id =
      context.node_stack().Pop<Parse::NodeKind::PatternBinding>();
  context.node_stack()
      .PopAndDiscardSoloParseNode<Parse::NodeKind::LetIntroducer>();
  // Process declaration modifiers.
  llvm::StringRef decl_name = "`let` declaration";
  CheckAccessModifiersOnDecl(context, decl_name);
  RequireDefaultFinalOnlyInInterfaces(context, decl_name);
  auto modifiers = ModifiersAllowedOnDecl(
      context,
      KeywordModifierSet().SetPrivate().SetProtected().SetDefault().SetFinal(),
      decl_name);

  if (modifiers.HasPrivate()) {
    context.TODO(context.decl_state_stack().innermost().saw_access_modifier,
                 "private");
  }
  if (modifiers.HasProtected()) {
    context.TODO(context.decl_state_stack().innermost().saw_access_modifier,
                 "protected");
  }
  if (modifiers.HasDefault()) {
    context.TODO(context.decl_state_stack().innermost().saw_decl_modifier,
                 "default");
  }
  if (modifiers.HasFinal()) {
    context.TODO(context.decl_state_stack().innermost().saw_decl_modifier,
                 "final");
  }
  context.decl_state_stack().Pop(DeclState::Let);

  // Convert the value to match the type of the pattern.
  auto pattern = context.insts().Get(pattern_id);
  value_id =
      ConvertToValueOfType(context, parse_node, value_id, pattern.type_id());

  // Update the binding with its value and add it to the current block, after
  // the computation of the value.
  // TODO: Support other kinds of pattern here.
  auto bind_name = pattern.As<SemIR::BindName>();
  CARBON_CHECK(!bind_name.value_id.is_valid())
      << "Binding should not already have a value!";
  bind_name.value_id = value_id;
  context.insts().Set(pattern_id, bind_name);
  context.inst_block_stack().AddInstId(pattern_id);

  // Add the name of the binding to the current scope.
  context.AddNameToLookup(pattern.parse_node(), bind_name.name_id, pattern_id);
  return true;
}

auto HandleLetIntroducer(Context& context, Parse::Node parse_node) -> bool {
  context.decl_state_stack().Push(DeclState::Let, parse_node);
  // Push a bracketing node to establish the pattern context.
  context.node_stack().Push(parse_node);
  return true;
}

auto HandleLetInitializer(Context& /*context*/, Parse::Node /*parse_node*/)
    -> bool {
  return true;
}

}  // namespace Carbon::Check

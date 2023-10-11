// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/sem_ir/node.h"

namespace Carbon::Check {

auto HandleLetDeclaration(Context& context, Parse::Node parse_node) -> bool {
  auto value_id = context.node_stack().PopExpression();
  SemIR::NodeId pattern_id =
      context.node_stack().Pop<Parse::NodeKind::PatternBinding>();
  context.node_stack()
      .PopAndDiscardSoloParseNode<Parse::NodeKind::LetIntroducer>();

  // Convert the value to match the type of the pattern.
  auto pattern = context.semantics_ir().GetNode(pattern_id);
  value_id =
      ConvertToValueOfType(context, parse_node, value_id, pattern.type_id());

  // Update the binding with its value and add it to the current block, after
  // the computation of the value.
  // TODO: Support other kinds of pattern here.
  auto bind_name = pattern.As<SemIR::BindName>();
  CARBON_CHECK(!bind_name.value_id.is_valid())
      << "Binding should not already have a value!";
  bind_name.value_id = value_id;
  context.semantics_ir().ReplaceNode(pattern_id, bind_name);
  context.node_block_stack().AddNodeId(pattern_id);

  // Add the name of the binding to the current scope.
  context.AddNameToLookup(pattern.parse_node(), bind_name.name_id, pattern_id);
  return true;
}

auto HandleLetIntroducer(Context& context, Parse::Node parse_node) -> bool {
  // Push a bracketing node to establish the pattern context.
  context.node_stack().Push(parse_node);
  return true;
}

auto HandleLetInitializer(Context& /*context*/, Parse::Node /*parse_node*/)
    -> bool {
  return true;
}

}  // namespace Carbon::Check

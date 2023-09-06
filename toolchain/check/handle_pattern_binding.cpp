// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/node.h"

namespace Carbon::Check {

auto HandleAddress(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleAddress");
}

auto HandleGenericPatternBinding(Context& context, Parse::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "GenericPatternBinding");
}

auto HandlePatternBinding(Context& context, Parse::Node parse_node) -> bool {
  auto [type_node, parsed_type_id] =
      context.node_stack().PopExpressionWithParseNode();
  auto cast_type_id = context.ExpressionAsType(type_node, parsed_type_id);

  // Get the name.
  auto [name_node, name_id] =
      context.node_stack().PopWithParseNode<Parse::NodeKind::Name>();

  // Allocate a node of the appropriate kind, linked to the name for error
  // locations.
  switch (auto context_parse_node_kind = context.parse_tree().node_kind(
              context.node_stack().PeekParseNode())) {
    case Parse::NodeKind::VariableIntroducer:
      context.AddNodeAndPush(parse_node, SemIR::Node::VarStorage::Make(
                                             name_node, cast_type_id, name_id));
      break;
    case Parse::NodeKind::ParameterListStart:
      context.AddNodeAndPush(parse_node, SemIR::Node::Parameter::Make(
                                             name_node, cast_type_id, name_id));
      break;
    default:
      CARBON_FATAL() << "Found a pattern binding in unexpected context "
                     << context_parse_node_kind;
  }
  return true;
}

auto HandleTemplate(Context& context, Parse::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleTemplate");
}

}  // namespace Carbon::Check

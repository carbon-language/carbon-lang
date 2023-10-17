// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
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
  auto type_node_copy = type_node;
  auto cast_type_id = ExpressionAsType(context, type_node, parsed_type_id);

  // Get the name.
  auto [name_node, name_id] =
      context.node_stack().PopWithParseNode<Parse::NodeKind::Name>();

  // Allocate a node of the appropriate kind, linked to the name for error
  // locations.
  // TODO: Each of these cases should create a `BindName` node.
  switch (auto context_parse_node_kind = context.parse_tree().node_kind(
              context.node_stack().PeekParseNode())) {
    case Parse::NodeKind::VariableIntroducer:
      if (!context.TryToCompleteType(cast_type_id, [&] {
            CARBON_DIAGNOSTIC(IncompleteTypeInVarDeclaration, Error,
                              "Variable has incomplete type `{0}`.",
                              std::string);
            return context.emitter().Build(
                type_node_copy, IncompleteTypeInVarDeclaration,
                context.semantics_ir().StringifyType(cast_type_id, true));
          })) {
        cast_type_id = SemIR::TypeId::Error;
      }
      context.AddNodeAndPush(
          parse_node, SemIR::VarStorage(name_node, cast_type_id, name_id));
      break;

    case Parse::NodeKind::ParameterListStart:
      // Parameters can have incomplete types in a function declaration, but not
      // in a function definition. We don't know which kind we have here.
      context.AddNodeAndPush(
          parse_node, SemIR::Parameter(name_node, cast_type_id, name_id));
      break;

    case Parse::NodeKind::LetIntroducer:
      if (!context.TryToCompleteType(cast_type_id, [&] {
            CARBON_DIAGNOSTIC(IncompleteTypeInLetDeclaration, Error,
                              "`let` binding has incomplete type `{0}`.",
                              std::string);
            return context.emitter().Build(
                type_node_copy, IncompleteTypeInLetDeclaration,
                context.semantics_ir().StringifyType(cast_type_id, true));
          })) {
        cast_type_id = SemIR::TypeId::Error;
      }
      // Create the node, but don't add it to a block until after we've formed
      // its initializer.
      // TODO: For general pattern parsing, we'll need to create a block to hold
      // the `let` pattern before we see the initializer.
      context.node_stack().Push(
          parse_node,
          context.semantics_ir().AddNodeInNoBlock(SemIR::BindName(
              name_node, cast_type_id, name_id, SemIR::NodeId::Invalid)));
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

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/sem_ir/node.h"

namespace Carbon::Check {

auto HandleVariableDeclaration(Context& context, Parse::Node parse_node)
    -> bool {
  // Handle the optional initializer.
  auto init_id = SemIR::NodeId::Invalid;
  bool has_init =
      context.parse_tree().node_kind(context.node_stack().PeekParseNode()) !=
      Parse::NodeKind::PatternBinding;
  if (has_init) {
    init_id = context.node_stack().PopExpression();
    context.node_stack()
        .PopAndDiscardSoloParseNode<Parse::NodeKind::VariableInitializer>();
  }

  // Extract the name binding.
  SemIR::NodeId var_id =
      context.node_stack().Pop<Parse::NodeKind::PatternBinding>();
  auto var = context.semantics_ir().GetNodeAs<SemIR::VarStorage>(var_id);

  // Form a corresponding name in the current context, and bind the name to the
  // variable.
  context.declaration_name_stack().AddNameToLookup(
      context.declaration_name_stack().MakeUnqualifiedName(var.parse_node,
                                                           var.name_id),
      var_id);

  // If there was an initializer, assign it to the storage.
  //
  // TODO: In a class scope, we should instead save the initializer somewhere
  // so that we can use it as a default.
  if (has_init) {
    init_id = Initialize(context, parse_node, var_id, init_id);
    // TODO: Consider using different node kinds for assignment versus
    // initialization.
    context.AddNode(SemIR::Assign(parse_node, var_id, init_id));
  }

  context.node_stack()
      .PopAndDiscardSoloParseNode<Parse::NodeKind::VariableIntroducer>();

  return true;
}

auto HandleVariableIntroducer(Context& context, Parse::Node parse_node)
    -> bool {
  // No action, just a bracketing node.
  context.node_stack().Push(parse_node);
  return true;
}

auto HandleVariableInitializer(Context& context, Parse::Node parse_node)
    -> bool {
  // No action, just a bracketing node.
  context.node_stack().Push(parse_node);
  return true;
}

}  // namespace Carbon::Check

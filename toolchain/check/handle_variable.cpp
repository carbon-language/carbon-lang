// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

auto HandleVariableDeclaration(Context& context, Parse::Node parse_node)
    -> bool {
  // Handle the optional initializer.
  auto init_id = SemIR::InstId::Invalid;
  bool has_init =
      context.parse_tree().node_kind(context.node_stack().PeekParseNode()) !=
      Parse::NodeKind::PatternBinding;
  if (has_init) {
    init_id = context.node_stack().PopExpression();
    context.node_stack()
        .PopAndDiscardSoloParseNode<Parse::NodeKind::VariableInitializer>();
  }

  // Extract the name binding.
  auto value_id = context.node_stack().Pop<Parse::NodeKind::PatternBinding>();
  if (auto bind_name = context.insts().Get(value_id).TryAs<SemIR::BindName>()) {
    // Form a corresponding name in the current context, and bind the name to
    // the variable.
    context.declaration_name_stack().AddNameToLookup(
        context.declaration_name_stack().MakeUnqualifiedName(
            bind_name->parse_node, bind_name->name_id),
        value_id);
    value_id = bind_name->value_id;
  }

  // If there was an initializer, assign it to the storage.
  if (has_init) {
    if (context.insts().Get(value_id).Is<SemIR::VarStorage>()) {
      init_id = Initialize(context, parse_node, value_id, init_id);
      // TODO: Consider using different instruction kinds for assignment versus
      // initialization.
      context.AddInst(SemIR::Assign{parse_node, value_id, init_id});
    } else {
      // TODO: In a class scope, we should instead save the initializer
      // somewhere so that we can use it as a default.
      context.TODO(parse_node, "Field initializer");
    }
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

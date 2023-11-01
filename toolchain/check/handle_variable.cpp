// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

auto HandleVariableDeclaration(Context& context, Parse::Lamp parse_lamp)
    -> bool {
  // Handle the optional initializer.
  auto init_id = SemIR::InstId::Invalid;
  bool has_init =
      context.parse_tree().node_kind(context.lamp_stack().PeekParseNode()) !=
      Parse::LampKind::PatternBinding;
  if (has_init) {
    init_id = context.lamp_stack().PopExpression();
    context.lamp_stack()
        .PopAndDiscardSoloParseNode<Parse::LampKind::VariableInitializer>();
  }

  // Extract the name binding.
  auto value_id = context.lamp_stack().Pop<Parse::LampKind::PatternBinding>();
  if (auto bind_name = context.insts().Get(value_id).TryAs<SemIR::BindName>()) {
    // Form a corresponding name in the current context, and bind the name to
    // the variable.
    context.declaration_name_stack().AddNameToLookup(
        context.declaration_name_stack().MakeUnqualifiedName(
            bind_name->parse_lamp, bind_name->name_id),
        value_id);
    value_id = bind_name->value_id;
  }

  // If there was an initializer, assign it to the storage.
  if (has_init) {
    if (context.insts().Get(value_id).Is<SemIR::VarStorage>()) {
      init_id = Initialize(context, parse_lamp, value_id, init_id);
      // TODO: Consider using different node kinds for assignment versus
      // initialization.
      context.AddInst(SemIR::Assign{parse_lamp, value_id, init_id});
    } else {
      // TODO: In a class scope, we should instead save the initializer
      // somewhere so that we can use it as a default.
      context.TODO(parse_lamp, "Field initializer");
    }
  }

  context.lamp_stack()
      .PopAndDiscardSoloParseNode<Parse::LampKind::VariableIntroducer>();

  return true;
}

auto HandleVariableIntroducer(Context& context, Parse::Lamp parse_lamp)
    -> bool {
  // No action, just a bracketing node.
  context.lamp_stack().Push(parse_lamp);
  return true;
}

auto HandleVariableInitializer(Context& context, Parse::Lamp parse_lamp)
    -> bool {
  // No action, just a bracketing node.
  context.lamp_stack().Push(parse_lamp);
  return true;
}

}  // namespace Carbon::Check

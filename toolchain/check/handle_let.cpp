// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

auto HandleLetDeclaration(Context& context, Parse::Lamp parse_lamp) -> bool {
  auto value_id = context.lamp_stack().PopExpression();
  SemIR::InstId pattern_id =
      context.lamp_stack().Pop<Parse::LampKind::PatternBinding>();
  context.lamp_stack()
      .PopAndDiscardSoloParseNode<Parse::LampKind::LetIntroducer>();

  // Convert the value to match the type of the pattern.
  auto pattern = context.insts().Get(pattern_id);
  value_id =
      ConvertToValueOfType(context, parse_lamp, value_id, pattern.type_id());

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
  context.AddNameToLookup(pattern.parse_lamp(), bind_name.name_id, pattern_id);
  return true;
}

auto HandleLetIntroducer(Context& context, Parse::Lamp parse_lamp) -> bool {
  // Push a bracketing node to establish the pattern context.
  context.lamp_stack().Push(parse_lamp);
  return true;
}

auto HandleLetInitializer(Context& /*context*/, Parse::Lamp /*parse_lamp*/)
    -> bool {
  return true;
}

}  // namespace Carbon::Check

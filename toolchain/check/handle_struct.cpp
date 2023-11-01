// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"

namespace Carbon::Check {

auto HandleStructComma(Context& context, Parse::Lamp /*parse_node*/) -> bool {
  context.ParamOrArgComma();
  return true;
}

auto HandleStructFieldDesignator(Context& context, Parse::Lamp /*parse_node*/)
    -> bool {
  // This leaves the designated name on top because the `.` isn't interesting.
  CARBON_CHECK(
      context.parse_tree().node_kind(context.lamp_stack().PeekParseNode()) ==
      Parse::LampKind::Name);
  return true;
}

auto HandleStructFieldType(Context& context, Parse::Lamp parse_node) -> bool {
  auto [type_node, type_id] = context.lamp_stack().PopExpressionWithParseNode();
  SemIR::TypeId cast_type_id = ExpressionAsType(context, type_node, type_id);

  auto [name_node, name_id] =
      context.lamp_stack().PopWithParseNode<Parse::LampKind::Name>();

  context.AddInstAndPush(
      parse_node, SemIR::StructTypeField{name_node, name_id, cast_type_id});
  return true;
}

auto HandleStructFieldUnknown(Context& context, Parse::Lamp parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleStructFieldUnknown");
}

auto HandleStructFieldValue(Context& context, Parse::Lamp parse_node) -> bool {
  auto [value_parse_node, value_inst_id] =
      context.lamp_stack().PopExpressionWithParseNode();
  StringId name_id = context.lamp_stack().Pop<Parse::LampKind::Name>();

  // Store the name for the type.
  context.args_type_info_stack().AddInst(SemIR::StructTypeField{
      parse_node, name_id, context.insts().Get(value_inst_id).type_id()});

  // Push the value back on the stack as an argument.
  context.lamp_stack().Push(parse_node, value_inst_id);
  return true;
}

auto HandleStructLiteral(Context& context, Parse::Lamp parse_node) -> bool {
  auto refs_id = context.ParamOrArgEnd(
      Parse::LampKind::StructLiteralOrStructTypeLiteralStart);

  context.PopScope();
  context.lamp_stack()
      .PopAndDiscardSoloParseNode<
          Parse::LampKind::StructLiteralOrStructTypeLiteralStart>();
  auto type_block_id = context.args_type_info_stack().Pop();

  auto type_id = context.CanonicalizeStructType(parse_node, type_block_id);

  auto value_id =
      context.AddInst(SemIR::StructLiteral{parse_node, type_id, refs_id});
  context.lamp_stack().Push(parse_node, value_id);
  return true;
}

auto HandleStructLiteralOrStructTypeLiteralStart(Context& context,
                                                 Parse::Lamp parse_node)
    -> bool {
  context.PushScope();
  context.lamp_stack().Push(parse_node);
  // At this point we aren't sure whether this will be a value or type literal,
  // so we push onto args irrespective. It just won't be used for a type
  // literal.
  context.args_type_info_stack().Push();
  context.ParamOrArgStart();
  return true;
}

auto HandleStructTypeLiteral(Context& context, Parse::Lamp parse_node) -> bool {
  auto refs_id = context.ParamOrArgEnd(
      Parse::LampKind::StructLiteralOrStructTypeLiteralStart);

  context.PopScope();
  context.lamp_stack()
      .PopAndDiscardSoloParseNode<
          Parse::LampKind::StructLiteralOrStructTypeLiteralStart>();
  // This is only used for value literals.
  context.args_type_info_stack().Pop();

  CARBON_CHECK(refs_id != SemIR::InstBlockId::Empty)
      << "{} is handled by StructLiteral.";

  context.AddInstAndPush(
      parse_node,
      SemIR::StructType{parse_node, SemIR::TypeId::TypeType, refs_id});
  return true;
}

}  // namespace Carbon::Check

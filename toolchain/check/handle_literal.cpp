// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cmath>

#include "toolchain/check/call.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/literal.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/type.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::BoolLiteralFalseId node_id)
    -> bool {
  context.node_stack().Push(
      node_id, MakeBoolLiteral(context, node_id, SemIR::BoolValue::False));
  return true;
}

auto HandleParseNode(Context& context, Parse::BoolLiteralTrueId node_id)
    -> bool {
  context.node_stack().Push(
      node_id, MakeBoolLiteral(context, node_id, SemIR::BoolValue::True));
  return true;
}

auto HandleParseNode(Context& context, Parse::CharLiteralId node_id) -> bool {
  auto value = context.tokens().GetCharLiteralValue(
      context.parse_tree().node_token(node_id));
  auto char_id = SemIR::CharId::ForCodePoint(llvm::APInt(32, value.value));
  CARBON_CHECK(char_id.has_value(), "Invalid character literal parsed: {0}",
               value.value);
  auto inst_id = AddInst<SemIR::CharLiteralValue>(
      context, node_id,
      {.type_id = GetSingletonType(context, SemIR::CharLiteralType::TypeInstId),
       .value = *char_id});
  context.node_stack().Push(node_id, inst_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::IntLiteralId node_id) -> bool {
  auto int_literal_id = MakeIntLiteral(
      context, node_id,
      context.tokens().GetIntLiteral(context.parse_tree().node_token(node_id)));
  context.node_stack().Push(node_id, int_literal_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::RealLiteralId node_id) -> bool {
  auto real_id =
      context.tokens().GetRealLiteral(context.parse_tree().node_token(node_id));
  AddInstAndPush<SemIR::FloatLiteralValue>(
      context, node_id,
      {.type_id =
           GetSingletonType(context, SemIR::FloatLiteralType::TypeInstId),
       .real_id = real_id});
  return true;
}

auto HandleParseNode(Context& context, Parse::StringLiteralId node_id) -> bool {
  auto str_literal_id =
      MakeStringLiteral(context, node_id,
                        context.tokens().GetStringLiteralValue(
                            context.parse_tree().node_token(node_id)));
  context.node_stack().Push(node_id, str_literal_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::BoolTypeLiteralId node_id)
    -> bool {
  auto type_inst_id = MakeBoolTypeLiteral(context, node_id);
  context.node_stack().Push(node_id, type_inst_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::CharTypeLiteralId node_id)
    -> bool {
  auto type_inst_id = MakeCharTypeLiteral(context, node_id);
  context.node_stack().Push(node_id, type_inst_id);
  return true;
}

// Shared implementation for handling `iN` and `uN` literals.
static auto HandleIntOrUnsignedIntTypeLiteral(Context& context,
                                              Parse::NodeId node_id,
                                              SemIR::IntKind int_kind,
                                              IntId size_id) -> bool {
  if (!(context.ints().Get(size_id) & 7).isZero()) {
    CARBON_DIAGNOSTIC(IntWidthNotMultipleOf8, Error,
                      "bit width of integer type literal must be a multiple of "
                      "8; use `Core.{0:Int|UInt}({1})` instead",
                      Diagnostics::BoolAsSelect, llvm::APSInt);
    context.emitter().Emit(
        node_id, IntWidthNotMultipleOf8, int_kind.is_signed(),
        llvm::APSInt(context.ints().Get(size_id), /*isUnsigned=*/true));
  }
  auto type_inst_id = MakeIntTypeLiteral(context, node_id, int_kind, size_id);
  context.node_stack().Push(node_id, type_inst_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::IntTypeLiteralId node_id)
    -> bool {
  auto tok_id = context.parse_tree().node_token(node_id);
  auto size_id = context.tokens().GetTypeLiteralSize(tok_id);
  return HandleIntOrUnsignedIntTypeLiteral(context, node_id,
                                           SemIR::IntKind::Signed, size_id);
}

auto HandleParseNode(Context& context, Parse::UnsignedIntTypeLiteralId node_id)
    -> bool {
  auto tok_id = context.parse_tree().node_token(node_id);
  auto size_id = context.tokens().GetTypeLiteralSize(tok_id);
  return HandleIntOrUnsignedIntTypeLiteral(context, node_id,
                                           SemIR::IntKind::Unsigned, size_id);
}

auto HandleParseNode(Context& context, Parse::FloatTypeLiteralId node_id)
    -> bool {
  auto tok_id = context.parse_tree().node_token(node_id);
  auto size_id = context.tokens().GetTypeLiteralSize(tok_id);
  auto type_inst_id = MakeFloatTypeLiteral(context, node_id, size_id);
  context.node_stack().Push(node_id, type_inst_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::StringTypeLiteralId node_id)
    -> bool {
  auto type_inst_id = MakeStringTypeLiteral(context, node_id);
  context.node_stack().Push(node_id, type_inst_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::TypeTypeLiteralId node_id)
    -> bool {
  auto type_inst_id = MakeTypeTypeLiteral(context, node_id);
  context.node_stack().Push(node_id, type_inst_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::AutoTypeLiteralId node_id)
    -> bool {
  // `auto` is currently only supported as the entire return type of a
  // function, as in `fn F() -> auto`, or as the type in a primitive return
  // form, as in `fn F() -> ref auto`. In those cases, `auto` is the only child
  // of a `ReturnType` node or a primitive form node, so is immediately followed
  // by it in postorder. There is always a following node, because the tree ends
  // with `FileEnd`.
  //
  // `HandleReturnSpecifier` recognizes the `auto`, so the value we push here is
  // just a placeholder. For a primitive form, the primitive form handler will
  // use the placeholder as its type component.
  //
  // TODO: Using parse tree lookahead is a hack; stop doing this. Ideally we'd
  // fix this as part of a change to accept and handle `auto` more broadly.
  auto next_node_kind =
      context.parse_tree().node_kind(Parse::NodeId(node_id.index + 1));
  bool is_return_type = next_node_kind == Parse::NodeKind::ReturnType;
  bool is_return_primitive_form =
      (next_node_kind == Parse::NodeKind::RefPrimitiveForm ||
       next_node_kind == Parse::NodeKind::ValPrimitiveForm ||
       next_node_kind == Parse::NodeKind::VarPrimitiveForm) &&
      context.parse_tree().node_kind(Parse::NodeId(node_id.index + 2)) ==
          Parse::NodeKind::ReturnType;
  if (is_return_type || is_return_primitive_form) {
    context.node_stack().Push(node_id, SemIR::AutoType::TypeInstId);
    return true;
  }
  return context.TODO(node_id, "HandleAutoTypeLiteral");
}

}  // namespace Carbon::Check

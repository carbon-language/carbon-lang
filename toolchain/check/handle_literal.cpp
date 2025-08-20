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
  AddInstAndPush<SemIR::BoolLiteral>(
      context, node_id,
      {.type_id = GetSingletonType(context, SemIR::BoolType::TypeInstId),
       .value = SemIR::BoolValue::False});
  return true;
}

auto HandleParseNode(Context& context, Parse::BoolLiteralTrueId node_id)
    -> bool {
  AddInstAndPush<SemIR::BoolLiteral>(
      context, node_id,
      {.type_id = GetSingletonType(context, SemIR::BoolType::TypeInstId),
       .value = SemIR::BoolValue::True});
  return true;
}

auto HandleParseNode(Context& context, Parse::CharLiteralId node_id) -> bool {
  auto value = context.tokens().GetCharLiteralValue(
      context.parse_tree().node_token(node_id));
  auto inst_id = AddInst<SemIR::CharLiteralValue>(
      context, node_id,
      {.type_id = GetSingletonType(context, SemIR::CharLiteralType::TypeInstId),
       .value = SemIR::CharId(value.value)});
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
  auto fn_inst_id = LookupNameInCore(context, node_id, "Bool");
  auto type_inst_id = PerformCall(context, node_id, fn_inst_id, {});
  context.node_stack().Push(node_id, type_inst_id);
  return true;
}

// Shared implementation for handling `iN` and `uN` literals.
static auto HandleIntOrUnsignedIntTypeLiteral(Context& context,
                                              Parse::NodeId node_id,
                                              SemIR::IntKind int_kind,
                                              IntId size_id) -> bool {
  if (!(context.ints().Get(size_id) & 3).isZero()) {
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
  context.node_stack().Push(node_id, SemIR::TypeType::TypeInstId);
  return true;
}

auto HandleParseNode(Context& context, Parse::AutoTypeLiteralId node_id)
    -> bool {
  return context.TODO(node_id, "HandleAutoTypeLiteral");
}

}  // namespace Carbon::Check

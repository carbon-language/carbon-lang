// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/constant.h"

#include "toolchain/check/cpp/import.h"
#include "toolchain/check/eval.h"

namespace Carbon::Check {

// TODO: dedup with code in `MapConstant` and `TryEvaluateMacroToConstant`.
auto MapAPValueToConstant(Context& context, SemIR::LocId loc_id,
                          const clang::APValue& ap_value, clang::QualType type)
    -> SemIR::ConstantId {
  SemIR::TypeId type_id = ImportCppType(context, loc_id, type).type_id;
  if (!type_id.has_value()) {
    return SemIR::ConstantId::NotConstant;
  }

  if (ap_value.isInt()) {
    if (type->isBooleanType()) {
      auto value = SemIR::BoolValue::From(!ap_value.getInt().isZero());
      return TryEvalInst(
          context, SemIR::BoolLiteral{.type_id = type_id, .value = value});
    } else {
      CARBON_CHECK(type->isIntegralOrEnumerationType());

      IntId int_id = context.ints().Add(ap_value.getInt());
      return TryEvalInst(context,
                         SemIR::IntValue{.type_id = type_id, .int_id = int_id});
    }
  } else {
    // TODO: support other types.
    return SemIR::ConstantId::NotConstant;
  }
}

}  // namespace Carbon::Check

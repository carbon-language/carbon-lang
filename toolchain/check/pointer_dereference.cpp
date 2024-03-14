// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

auto PerformPointerDereference(Context& context,
                               Parse::AnyPointerDeferenceExprId node_id,
                               SemIR::InstId base_id) -> SemIR::InstId {
  base_id = ConvertToValueExpr(context, base_id);
  auto type_id =
      context.GetUnqualifiedType(context.insts().Get(base_id).type_id());
  auto result_type_id = SemIR::TypeId::Error;
  if (auto pointer_type =
          context.types().TryGetAs<SemIR::PointerType>(type_id)) {
    result_type_id = pointer_type->pointee_id;
  } else if (type_id != SemIR::TypeId::Error) {
    CARBON_DIAGNOSTIC(DerefOfNonPointer, Error,
                      "Cannot dereference operand of non-pointer type `{0}`.",
                      SemIR::TypeId);
    auto builder =
        context.emitter().Build(TokenOnly(node_id), DerefOfNonPointer, type_id);
    builder.Emit();
  }
  return context.AddInst({node_id, SemIR::Deref{result_type_id, base_id}});
}

}  // namespace Carbon::Check

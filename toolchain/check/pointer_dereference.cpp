// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/STLFunctionalExtras.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

auto PerformPointerDereference(
    Context& context, Parse::AnyPointerDeferenceExprId node_id,
    SemIR::InstId base_id,
    llvm::function_ref<auto(SemIR::TypeId not_pointer_type_id)->void>
        diagnose_not_pointer) -> SemIR::InstId {
  // TODO: Once we have a finalized design for a pointer interface, use
  //
  //   HandleUnaryOperator(context, node_id, {"Pointer", "Dereference"});
  //
  // to convert to a pointer value.
  base_id = ConvertToValueExpr(context, base_id);
  auto type_id =
      context.GetUnqualifiedType(context.insts().Get(base_id).type_id());
  auto result_type_id = SemIR::TypeId::Error;
  if (auto pointer_type =
          context.types().TryGetAs<SemIR::PointerType>(type_id)) {
    result_type_id = pointer_type->pointee_id;
  } else if (type_id != SemIR::TypeId::Error) {
    diagnose_not_pointer(type_id);
  }
  return context.AddInst({node_id, SemIR::Deref{result_type_id, base_id}});
}

}  // namespace Carbon::Check

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/operator.h"

#include "toolchain/check/call.h"
#include "toolchain/check/context.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/member_access.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Returns the `Op` function for the specified operator.
static auto GetOperatorOpFunction(Context& context, SemIR::LocId loc_id,
                                  Operator op) -> SemIR::InstId {
  // Look up the interface, and pass it any generic arguments.
  auto interface_id = LookupNameInCore(context, loc_id, op.interface_name);
  if (!op.interface_args_ref.empty()) {
    interface_id =
        PerformCall(context, loc_id, interface_id, op.interface_args_ref);
  }

  // Look up the interface member.
  auto op_name_id =
      SemIR::NameId::ForIdentifier(context.identifiers().Add(op.op_name));
  return PerformMemberAccess(context, loc_id, interface_id, op_name_id);
}

auto BuildUnaryOperator(Context& context, SemIR::LocId loc_id, Operator op,
                        SemIR::InstId operand_id,
                        Context::BuildDiagnosticFn missing_impl_diagnoser)
    -> SemIR::InstId {
  // Look up the operator function.
  auto op_fn = GetOperatorOpFunction(context, loc_id.ToImplicit(), op);

  // Form `operand.(Op)`.
  auto bound_op_id = PerformCompoundMemberAccess(context, loc_id, operand_id,
                                                 op_fn, missing_impl_diagnoser);
  if (bound_op_id == SemIR::ErrorInst::SingletonInstId) {
    return SemIR::ErrorInst::SingletonInstId;
  }

  // Form `bound_op()`.
  return PerformCall(context, loc_id, bound_op_id, {});
}

auto BuildBinaryOperator(Context& context, SemIR::LocId loc_id, Operator op,
                         SemIR::InstId lhs_id, SemIR::InstId rhs_id,
                         Context::BuildDiagnosticFn missing_impl_diagnoser)
    -> SemIR::InstId {
  // Look up the operator function.
  auto op_fn = GetOperatorOpFunction(context, loc_id.ToImplicit(), op);

  // Form `lhs.(Op)`.
  auto bound_op_id = PerformCompoundMemberAccess(context, loc_id, lhs_id, op_fn,
                                                 missing_impl_diagnoser);
  if (bound_op_id == SemIR::ErrorInst::SingletonInstId) {
    return SemIR::ErrorInst::SingletonInstId;
  }

  // Form `bound_op(rhs)`.
  return PerformCall(context, loc_id, bound_op_id, {rhs_id});
}

}  // namespace Carbon::Check

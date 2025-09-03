// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/operator.h"

#include "toolchain/check/call.h"
#include "toolchain/check/context.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/import_cpp.h"
#include "toolchain/check/member_access.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Returns the `Op` function for the specified operator.
static auto GetOperatorOpFunction(Context& context, SemIR::LocId loc_id,
                                  Operator op) -> SemIR::InstId {
  auto implicit_loc_id = context.insts().GetLocIdForDesugaring(loc_id);

  // Look up the interface, and pass it any generic arguments.
  auto interface_id =
      LookupNameInCore(context, implicit_loc_id, op.interface_name);
  if (!op.interface_args_ref.empty()) {
    interface_id = PerformCall(context, implicit_loc_id, interface_id,
                               op.interface_args_ref);
  }

  // Look up the interface member.
  auto op_name_id =
      SemIR::NameId::ForIdentifier(context.identifiers().Add(op.op_name));
  return PerformMemberAccess(context, implicit_loc_id, interface_id,
                             op_name_id);
}

auto BuildUnaryOperator(Context& context, SemIR::LocId loc_id, Operator op,
                        SemIR::InstId operand_id,
                        MakeDiagnosticBuilderFn missing_impl_diagnoser)
    -> SemIR::InstId {
  // Look up the operator function.
  auto op_fn = GetOperatorOpFunction(context, loc_id, op);

  // Form `operand.(Op)`.
  auto bound_op_id = PerformCompoundMemberAccess(context, loc_id, operand_id,
                                                 op_fn, missing_impl_diagnoser);
  if (bound_op_id == SemIR::ErrorInst::InstId) {
    return SemIR::ErrorInst::InstId;
  }

  // Form `bound_op()`.
  return PerformCall(context, loc_id, bound_op_id, {});
}

// Returns whether the type of the instruction is a C++ class.
static auto IsOfCppClassType(Context& context, SemIR::InstId inst_id) -> bool {
  auto class_type = context.insts().TryGetAs<SemIR::ClassType>(
      context.types().GetInstId(context.insts().Get(inst_id).type_id()));
  if (!class_type) {
    // Not a class.
    return false;
  }

  return context.name_scopes()
      .Get(context.classes().Get(class_type->class_id).scope_id)
      .is_cpp_scope();
}

auto BuildBinaryOperator(Context& context, SemIR::LocId loc_id, Operator op,
                         SemIR::InstId lhs_id, SemIR::InstId rhs_id,
                         MakeDiagnosticBuilderFn missing_impl_diagnoser)
    -> SemIR::InstId {
  // For binary operators with a C++ class as at least one of the operands, try
  // to import and call the C++ operator.
  // TODO: Instead of hooking this here, change impl lookup, so that a generic
  // constraint such as `T:! Core.Add` is satisfied by C++ class types that are
  // addable. See
  // https://github.com/carbon-language/carbon-lang/pull/5996/files/5d01fa69511b76f87efbc0387f5e40abcf4c911a#r2308666348
  // and
  // https://github.com/carbon-language/carbon-lang/pull/5996/files/5d01fa69511b76f87efbc0387f5e40abcf4c911a#r2308664536
  if (IsOfCppClassType(context, lhs_id) || IsOfCppClassType(context, rhs_id)) {
    SemIR::ScopeLookupResult cpp_lookup_result =
        ImportOperatorFromCpp(context, loc_id, op);
    if (cpp_lookup_result.is_found()) {
      return PerformCall(context, loc_id, cpp_lookup_result.target_inst_id(),
                         {lhs_id, rhs_id});
    }
  }

  // Look up the operator function.
  auto op_fn = GetOperatorOpFunction(context, loc_id, op);

  // Form `lhs.(Op)`.
  auto bound_op_id = PerformCompoundMemberAccess(context, loc_id, lhs_id, op_fn,
                                                 missing_impl_diagnoser);
  if (bound_op_id == SemIR::ErrorInst::InstId) {
    return SemIR::ErrorInst::InstId;
  }

  // Form `bound_op(rhs)`.
  return PerformCall(context, loc_id, bound_op_id, {rhs_id});
}

}  // namespace Carbon::Check

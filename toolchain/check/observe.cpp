// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/observe.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/type.h"

namespace Carbon::Check {
auto GetObserveIds(Context& context) -> llvm::SmallVector<SemIR::ObserveId> {
  llvm::SmallVector<SemIR::ObserveId> ids;
  if (context.scope_stack().IsInFunctionScope()) {
    llvm::append_range(ids, context.observe_stack().PeekAllValues());
  }
  for (auto interface : context.interfaces().values()) {
    if (interface.observe_block_id.has_value()) {
      llvm::append_range(
          ids, context.observe_blocks().Get(interface.observe_block_id));
    }
  }
  return ids;
}

auto UnpackObserveDecl(Context& context, SemIR::ObserveId id)
    -> std::pair<llvm::SmallVector<SemIR::InstId>, SemIR::InstId> {
  llvm::SmallVector<SemIR::InstId> canonical_value_ids;
  auto constraint_canonical_value_id = SemIR::InstId::None;
  const auto& observe = context.observes().Get(id);
  for (auto operator_id :
       context.inst_blocks().GetOrEmpty(observe.operations_id)) {
    CARBON_KIND_SWITCH(context.insts().Get(operator_id)) {
      case CARBON_KIND(SemIR::ObserveEquivalent observe_equivalent): {
        if (canonical_value_ids.empty()) {
          canonical_value_ids.push_back(
              GetCanonicalFacetOrTypeValue(context, observe_equivalent.lhs_id));
        }
        canonical_value_ids.push_back(
            GetCanonicalFacetOrTypeValue(context, observe_equivalent.rhs_id));
        break;
      }
      case CARBON_KIND(SemIR::ObserveImpls observe_impls): {
        if (canonical_value_ids.empty()) {
          canonical_value_ids.push_back(
              GetCanonicalFacetOrTypeValue(context, observe_impls.lhs_id));
        }
        constraint_canonical_value_id =
            GetCanonicalFacetOrTypeValue(context, observe_impls.rhs_id);
        break;
      }
      default: {
        break;
      }
    }
  }
  return {std::move(canonical_value_ids), constraint_canonical_value_id};
}

auto CheckObserveOperandsForEquivalance(
    Context& context, llvm::ArrayRef<SemIR::InstId> observe_operand_ids,
    SemIR::TypeId lhs_type_id, SemIR::TypeId rhs_type_id) -> bool {
  auto lhs_found = false;
  auto rhs_found = false;
  for (auto operand_id : observe_operand_ids) {
    auto operand_type_id = context.insts().GetAttachedType(operand_id);
    lhs_found = lhs_found || lhs_type_id == operand_type_id;
    rhs_found = rhs_found || rhs_type_id == operand_type_id;
  }
  return lhs_found && rhs_found;
}
}  // namespace Carbon::Check

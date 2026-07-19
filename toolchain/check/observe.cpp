// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/observe.h"

#include "common/check.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/type.h"

namespace Carbon::Check {
auto GetObserveIds(Context& context, SemIR::InstId expr_id)
    -> llvm::SmallVector<SemIR::ObserveId> {
  llvm::SmallVector<SemIR::ObserveId> ids;

  // Search must preserve declaration order since in some cases later
  // observations may depend on earlier ones. For instance, in an interface
  // requirement chain of `D` -> `C` -> `B` -> `A`, we must first observe that
  // `T impls C` before we can observe `T impls B` when resolving from `D` to
  // `A`.

  if (auto access =
          context.insts().Get(expr_id).TryAs<SemIR::ImplWitnessAccess>()) {
    if (auto witness = context.insts()
                           .Get(access->witness_id)
                           .TryAs<SemIR::LookupImplWitness>()) {
      auto specific_interface = context.specific_interfaces().Get(
          witness->query_specific_interface_id);
      auto interface =
          context.interfaces().Get(specific_interface.interface_id);
      if (interface.observe_block_id.has_value()) {
        llvm::append_range(
            ids, context.observe_blocks().Get(interface.observe_block_id));
      }
    }
  }

  if (context.scope_stack().IsInFunctionScope()) {
    llvm::append_range(ids, context.observe_stack().PeekAllValues());
  }

  return ids;
}

auto UnpackObserve(Context& context, const SemIR::Observe& observe)
    -> std::pair<llvm::SmallVector<SemIR::InstId>, SemIR::InstId> {
  llvm::SmallVector<SemIR::InstId> operand_ids;
  auto impls_constraint_id = SemIR::InstId::None;
  for (auto operation_id :
       context.inst_blocks().GetOrEmpty(observe.operations_id)) {
    CARBON_KIND_SWITCH(context.insts().Get(operation_id)) {
      case CARBON_KIND(SemIR::ObserveEquivalent observe_equivalent): {
        if (operand_ids.empty()) {
          operand_ids.push_back(observe_equivalent.lhs_id);
        }
        operand_ids.push_back(observe_equivalent.rhs_id);
        break;
      }
      case CARBON_KIND(SemIR::ObserveImpls observe_impls): {
        if (operand_ids.empty()) {
          operand_ids.push_back(observe_impls.lhs_id);
        }
        impls_constraint_id = observe_impls.rhs_id;
        break;
      }
      default: {
        CARBON_FATAL("Unexpected inst kind: {0}", operation_id);
        break;
      }
    }
  }
  return {std::move(operand_ids), impls_constraint_id};
}

auto CheckObserveEquivalence(Context& context,
                             llvm::ArrayRef<SemIR::InstId> observe_operand_ids,
                             SemIR::TypeId lhs_type_id,
                             SemIR::TypeId rhs_type_id) -> bool {
  auto lhs_found = false;
  auto rhs_found = false;
  for (auto operand_id : observe_operand_ids) {
    auto operand_type_id = context.insts().GetAttachedType(operand_id);
    lhs_found |= lhs_type_id == operand_type_id;
    rhs_found |= rhs_type_id == operand_type_id;
    if (lhs_found && rhs_found) {
      return true;
    }
  }
  return false;
}
}  // namespace Carbon::Check

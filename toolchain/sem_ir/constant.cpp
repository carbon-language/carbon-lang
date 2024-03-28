// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/constant.h"

#include "toolchain/sem_ir/inst_profile.h"

namespace Carbon::SemIR {

auto ConstantStore::GetOrAdd(Inst inst, bool is_symbolic) -> ConstantId {
  // Compute the instruction's profile.
  ConstantNode node = {.inst = inst, .constant_id = ConstantId::NotConstant};
  llvm::FoldingSetNodeID id;
  node.Profile(id, constants_.getContext());

  // Check if we have already created this constant.
  void* insert_pos;
  if (ConstantNode* found = constants_.FindNodeOrInsertPos(id, insert_pos)) {
    CARBON_CHECK(found->constant_id.is_constant())
        << "Found non-constant in constant store for " << inst;
    CARBON_CHECK(found->constant_id.is_symbolic() == is_symbolic)
        << "Mismatch in phase for constant " << inst;
    return found->constant_id;
  }

  // Create the new inst and insert the new node.
  auto inst_id = constants_.getContext()->insts().AddInNoBlock(
      LocIdAndInst::Untyped(Parse::NodeId::Invalid, inst));
  auto constant_id = is_symbolic
                         ? SemIR::ConstantId::ForSymbolicConstant(inst_id)
                         : SemIR::ConstantId::ForTemplateConstant(inst_id);
  node.constant_id = constant_id;
  constants_.InsertNode(new (*allocator_) ConstantNode(node), insert_pos);

  // The constant value of any constant instruction is that instruction itself.
  constants_.getContext()->constant_values().Set(inst_id, constant_id);
  return constant_id;
}

auto ConstantStore::GetAsVector() const -> llvm::SmallVector<InstId, 0> {
  llvm::SmallVector<InstId, 0> result;
  result.reserve(constants_.size());
  for (const ConstantNode& node : constants_) {
    result.push_back(node.constant_id.inst_id());
  }
  // For stability, put the results into index order. This happens to also be
  // insertion order.
  std::sort(result.begin(), result.end(),
            [](InstId a, InstId b) { return a.index < b.index; });
  return result;
}

auto ConstantStore::ConstantNode::Profile(llvm::FoldingSetNodeID& id,
                                          File* sem_ir) -> void {
  ProfileConstant(id, *sem_ir, inst);
}

}  // namespace Carbon::SemIR

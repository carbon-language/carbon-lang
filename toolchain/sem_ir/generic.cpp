// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/constant.h"

#include "toolchain/sem_ir/inst_profile.h"

namespace Carbon::SemIR {

// Profile a GenericInstance that has already been created. This must match
// ProfilePotentialNewGenericInstance.
auto GenericInstanceStore::Node::Profile(llvm::FoldingSetNodeID& id,
                                         GenericInstanceStore* store) -> void {
  auto& generic = store->generic_instances_.Get(generic_instance_id);
  id.AddInteger(generic.generic_id.index);
  for (auto inst_id : store->inst_block_store_.Get(generic.args_id)) {
    id.AddInteger(inst_id.index);
  }
}

// Profile a potentially new generic instance, for which an InstBlock has not
// yet been created. This must match Node::Profile.
static auto ProfilePotentialNewGenericInstance(
    llvm::FoldingSetNodeID& id, GenericId generic_id,
    llvm::ArrayRef<ConstantId> arg_ids) -> void {
  id.AddInteger(generic_id.index);
  for (auto const_id : arg_ids) {
    id.AddInteger(const_id.inst_id().index);
  }
}

auto GenericInstanceStore::GetOrAdd(
    GenericId generic_id, llvm::ArrayRef<ConstantId> arg_ids) -> GenericInstanceId {
  // Compute the generic instance's profile.
  llvm::FoldingSetNodeID id;
  ProfilePotentialNewGenericInstance(id, generic_id, arg_ids);

  // Check if we have already created this generic instance.
  void* insert_pos;
  if (Node* found = lookup_table_.FindNodeOrInsertPos(id, insert_pos)) {
    return found->generic_instance_id;
  }

  // Create the new instance and insert the new node.
  llvm::SmallVector<InstId> arg_inst_ids;
  arg_inst_ids.reserve(arg_ids.size());
  for (auto arg_const_id : arg_ids) {
    arg_inst_ids.push_back(arg_const_id.inst_id());
  }
  auto generic_instance_id =
      generic_instances_.Add({.generic_id = generic_id,
                              .args_id = inst_block_store_.Add(arg_inst_ids)});
  lookup_table_.InsertNode(new (*allocator_) Node(generic_instance_id),
                           insert_pos);
  return generic_instance_id;
}

}  // namespace Carbon::SemIR

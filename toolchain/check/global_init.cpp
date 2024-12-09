// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/global_init.h"

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto GlobalInit::Resume() -> void {
  context_->inst_block_stack().Push(block_id_, block_);
}

auto GlobalInit::Suspend() -> void {
  // TODO: Consider splicing together blocks in order to avoid sizable copies
  // here.
  auto contents = context_->inst_block_stack().PeekCurrentBlockContents();
  block_.assign(contents.begin(), contents.end());

  block_id_ = context_->inst_block_stack().PeekOrAdd();
  context_->inst_block_stack().PopAndDiscard();
}

auto GlobalInit::Finalize() -> void {
  // __global_init is only added if there are initialization instructions.
  if (block_.empty() && block_id_ == SemIR::InstBlockId::GlobalInit) {
    return;
  }

  Resume();
  context_->AddInst<SemIR::Return>(Parse::NodeId::Invalid, {});
  // Pop the GlobalInit block here to finalize it.
  context_->inst_block_stack().Pop();

  auto name_id = context_->sem_ir().identifiers().Add("__global_init");
  context_->sem_ir().set_global_ctor_id(context_->sem_ir().functions().Add(
      {{.name_id = SemIR::NameId::ForIdentifier(name_id),
        .parent_scope_id = SemIR::NameScopeId::Package,
        .generic_id = SemIR::GenericId::Invalid,
        .first_param_node_id = Parse::NodeId::Invalid,
        .last_param_node_id = Parse::NodeId::Invalid,
        .pattern_block_id = SemIR::InstBlockId::Empty,
        .implicit_param_patterns_id = SemIR::InstBlockId::Invalid,
        .param_patterns_id = SemIR::InstBlockId::Empty,
        .call_params_id = SemIR::InstBlockId::Empty,
        .is_extern = false,
        .extern_library_id = SemIR::LibraryNameId::Invalid,
        .non_owning_decl_id = SemIR::InstId::Invalid,
        .first_owning_decl_id = SemIR::InstId::Invalid},
       {.return_slot_pattern_id = SemIR::InstId::Invalid,
        .body_block_ids = {SemIR::InstBlockId::GlobalInit}}}));
}

}  // namespace Carbon::Check

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

#include "common/check.h"

namespace Carbon::Check {

Context::Context(DiagnosticEmitter<SemIRLoc>* emitter,
                 Parse::GetTreeAndSubtreesFn tree_and_subtrees_getter,
                 SemIR::File* sem_ir, int imported_ir_count, int total_ir_count,
                 llvm::raw_ostream* vlog_stream)
    : emitter_(emitter),
      tree_and_subtrees_getter_(tree_and_subtrees_getter),
      sem_ir_(sem_ir),
      vlog_stream_(vlog_stream),
      node_stack_(sem_ir->parse_tree(), vlog_stream),
      inst_block_stack_("inst_block_stack_", *sem_ir, vlog_stream),
      pattern_block_stack_("pattern_block_stack_", *sem_ir, vlog_stream),
      param_and_arg_refs_stack_(*sem_ir, vlog_stream, node_stack_),
      args_type_info_stack_("args_type_info_stack_", *sem_ir, vlog_stream),
      decl_name_stack_(this),
      scope_stack_(sem_ir_),
      vtable_stack_("vtable_stack_", *sem_ir, vlog_stream),
      global_init_(this),
      region_stack_(
          [this](SemIRLoc loc, std::string label) { TODO(loc, label); }) {
  // Prepare fields which relate to the number of IRs available for import.
  import_irs().Reserve(imported_ir_count);
  import_ir_constant_values_.reserve(imported_ir_count);
  check_ir_map_.resize(total_ir_count, SemIR::ImportIRId::None);

  // TODO: Remove this and add a `VerifyOnFinish` once we properly push and pop
  // in the right places.
  generic_region_stack().Push();
}

auto Context::TODO(SemIRLoc loc, std::string label) -> bool {
  CARBON_DIAGNOSTIC(SemanticsTodo, Error, "semantics TODO: `{0}`", std::string);
  emitter_->Emit(loc, SemanticsTodo, std::move(label));
  return false;
}

auto Context::VerifyOnFinish() const -> void {
  // Information in all the various context objects should be cleaned up as
  // various pieces of context go out of scope. At this point, nothing should
  // remain.
  // node_stack_ will still contain top-level entities.
  inst_block_stack_.VerifyOnFinish();
  pattern_block_stack_.VerifyOnFinish();
  param_and_arg_refs_stack_.VerifyOnFinish();
  args_type_info_stack_.VerifyOnFinish();
  CARBON_CHECK(struct_type_fields_stack_.empty());
  // TODO: Add verification for decl_name_stack_ and
  // decl_introducer_state_stack_.
  scope_stack_.VerifyOnFinish();
  // TODO: Add verification for generic_region_stack_.

#ifndef NDEBUG
  if (auto verify = sem_ir_->Verify(); !verify.ok()) {
    CARBON_FATAL("{0}Built invalid semantics IR: {1}\n", sem_ir_,
                 verify.error());
  }
#endif
}

auto Context::PrintForStackDump(llvm::raw_ostream& output) const -> void {
  output << "Check::Context\n";

  // In a stack dump, this is probably indented by a tab. We treat that as 8
  // spaces then add a couple to indent past the Context label.
  constexpr int Indent = 10;

  node_stack_.PrintForStackDump(Indent, output);
  inst_block_stack_.PrintForStackDump(Indent, output);
  pattern_block_stack_.PrintForStackDump(Indent, output);
  param_and_arg_refs_stack_.PrintForStackDump(Indent, output);
  args_type_info_stack_.PrintForStackDump(Indent, output);
}

}  // namespace Carbon::Check

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

#include <string>
#include <utility>

#include "common/check.h"
#include "toolchain/check/deferred_definition_worklist.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

Context::Context(DiagnosticEmitterBase* emitter,
                 Parse::GetTreeAndSubtreesFn tree_and_subtrees_getter,
                 SemIR::File* sem_ir, int imported_ir_count, int total_ir_count,
                 bool gen_implicit_type_impls, llvm::raw_ostream* vlog_stream)
    : emitter_(emitter),
      tree_and_subtrees_getter_(tree_and_subtrees_getter),
      sem_ir_(sem_ir),
      total_ir_count_(total_ir_count),
      gen_implicit_type_impls_(gen_implicit_type_impls),
      vlog_stream_(vlog_stream),
      node_stack_(sem_ir->parse_tree(), vlog_stream),
      inst_block_stack_("inst_block_stack_", *sem_ir, vlog_stream),
      pattern_block_stack_("pattern_block_stack_", *sem_ir, vlog_stream),
      param_and_arg_refs_stack_(*sem_ir, vlog_stream, node_stack_),
      args_type_info_stack_("args_type_info_stack_", *sem_ir, vlog_stream),
      decl_name_stack_(this),
      scope_stack_(sem_ir_),
      deferred_definition_worklist_(vlog_stream),
      vtable_stack_("vtable_stack_", *sem_ir, vlog_stream),
      check_ir_map_(
          FixedSizeValueStore<SemIR::CheckIRId, SemIR::ImportIRId>::
              MakeWithExplicitSize(total_ir_count_, SemIR::ImportIRId::None)),
      global_init_(this),
      region_stack_([this](SemIR::LocId loc_id, std::string label) {
        TODO(loc_id, label);
      }) {
  // Prepare fields which relate to the number of IRs available for import.
  import_irs().Reserve(imported_ir_count);
  import_ir_constant_values_.reserve(imported_ir_count);
}

auto Context::TODO(SemIR::LocId loc_id, std::string label) -> bool {
  CARBON_DIAGNOSTIC(SemanticsTodo, Error, "semantics TODO: `{0}`", std::string);
  emitter_->Emit(loc_id, SemanticsTodo, std::move(label));
  return false;
}

auto Context::TODO(SemIR::InstId loc_inst_id, std::string label) -> bool {
  return TODO(SemIR::LocId(loc_inst_id), label);
}

auto Context::VerifyOnFinish() const -> void {
  // Information in all the various context objects should be cleaned up as
  // various pieces of context go out of scope. At this point, nothing should
  // remain, so we verify stacks are empty. `node_stack_` is an exception
  // because it ends containing all top-level entities.
  inst_block_stack_.VerifyOnFinish();
  pattern_block_stack_.VerifyOnFinish();
  param_and_arg_refs_stack_.VerifyOnFinish();
  args_type_info_stack_.VerifyOnFinish();
  CARBON_CHECK(struct_type_fields_stack_.empty());
  CARBON_CHECK(field_decls_stack_.empty());
  decl_name_stack_.VerifyOnFinish();
  decl_introducer_state_stack_.VerifyOnFinish();
  scope_stack_.VerifyOnFinish();
  generic_region_stack_.VerifyOnFinish();
  vtable_stack_.VerifyOnFinish();
  region_stack_.VerifyOnFinish();
  CARBON_CHECK(impl_lookup_stack_.empty());

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

  output.indent(Indent);
  output << "filename: " << tokens().source().filename() << "\n";

  node_stack_.PrintForStackDump(Indent, output);
  inst_block_stack_.PrintForStackDump(Indent, output);
  pattern_block_stack_.PrintForStackDump(Indent, output);
  param_and_arg_refs_stack_.PrintForStackDump(Indent, output);
  args_type_info_stack_.PrintForStackDump(Indent, output);
}

}  // namespace Carbon::Check

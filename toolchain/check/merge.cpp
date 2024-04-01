// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/merge.h"

#include "toolchain/check/function.h"
#include "toolchain/check/import_ref.h"

namespace Carbon::Check {

// Returns the instruction to consider when merging the given inst_id. Returns
// nullopt if merging is infeasible and no diagnostic should be printed.
static auto ResolveMergeableInst(Context& context, SemIR::InstId inst_id)
    -> std::optional<SemIR::Inst> {
  auto inst = context.insts().Get(inst_id);
  switch (inst.kind()) {
    case SemIR::ImportRefUnused::Kind:
      // Resolve before merging.
      TryResolveImportRefUnused(context, inst_id);
      break;

    case SemIR::ImportRefUsed::Kind:
      // Already resolved.
      break;

    case SemIR::Namespace::Kind:
      // Return back the namespace directly.
      return inst;

    default:
      CARBON_FATAL() << "Unexpected inst kind passed to ResolveMergeableInst: "
                     << inst;
  }

  auto const_id = context.constant_values().Get(inst_id);
  // TODO: Function and type declarations are constant, but `var` declarations
  // are non-constant and should still merge.
  if (!const_id.is_constant()) {
    return std::nullopt;
  }
  return context.insts().Get(const_id.inst_id());
}

auto MergeImportRef(Context& context, SemIR::InstId new_inst_id,
                    SemIR::InstId prev_inst_id) -> void {
  auto new_inst = ResolveMergeableInst(context, new_inst_id);
  auto prev_inst = ResolveMergeableInst(context, prev_inst_id);
  if (!new_inst || !prev_inst) {
    // TODO: Once `var` declarations get an associated instruction for handling,
    // it might be more appropriate to return without diagnosing here, to handle
    // invalid declarations.
    context.DiagnoseDuplicateName(new_inst_id, prev_inst_id);
    return;
  }

  if (new_inst->kind() != prev_inst->kind()) {
    context.DiagnoseDuplicateName(new_inst_id, prev_inst_id);
    return;
  }

  switch (new_inst->kind()) {
    case SemIR::FunctionDecl::Kind: {
      auto new_fn = context.functions().Get(
          new_inst->As<SemIR::FunctionDecl>().function_id);
      auto prev_fn_id = prev_inst->As<SemIR::FunctionDecl>().function_id;
      // TODO: May need to "spoil" the new function to prevent it from being
      // emitted, since it will already be added.
      MergeFunctionRedecl(context, context.insts().GetLocId(new_inst_id),
                          new_fn,
                          /*new_is_definition=*/false, prev_fn_id,
                          /*prev_is_imported=*/true);
      return;
    }
    default:
      context.TODO(new_inst_id, "Merging not yet supported.");
      return;
  }
}

}  // namespace Carbon::Check

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_MERGE_H_
#define CARBON_TOOLCHAIN_CHECK_MERGE_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/import_ir.h"

namespace Carbon::Check {

struct InstForMerge {
  // The resolved instruction.
  SemIR::Inst inst;
  // The imported instruction, or invalid if not an import. This should
  // typically only be used for the ImportIRId, but we only resolve that if
  // needed.
  SemIR::ImportIRInstId import_ir_inst_id;
};

// Resolves prev_inst_id for merging (or name conflicts). This handles imports
// to return the instruction relevant for a merge. If an import is found and was
// previously used, it notes it, although an invalid redeclaration may diagnose
// for other reasons too.
auto ResolvePrevInstForMerge(Context& context, Parse::NodeId node_id,
                             SemIR::InstId prev_inst_id) -> InstForMerge;

// When the prior name lookup result is an import and we are successfully
// merging, replace the name lookup result with the reference in the current
// file.
auto ReplacePrevInstForMerge(Context& context, SemIR::NameScopeId scope_id,
                             SemIR::NameId name_id, SemIR::InstId new_inst_id)
    -> void;

// Merges an import ref at new_inst_id another at prev_inst_id. May print a
// diagnostic if merging is invalid.
auto MergeImportRef(Context& context, SemIR::InstId new_inst_id,
                    SemIR::InstId prev_inst_id) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_MERGE_H_

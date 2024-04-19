// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CLASS_H_
#define CARBON_TOOLCHAIN_CHECK_CLASS_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/class.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Tries to merge new_class into prev_class_id. Since new_class won't have a
// definition even if one is upcoming, set is_definition to indicate the planned
// result.
//
// If merging is successful, returns true and may update the previous class.
// Otherwise, returns false. Prints a diagnostic when appropriate.
auto MergeClassRedecl(Context& context, SemIRLoc new_loc,
                      SemIR::Class& new_class, bool new_is_import,
                      bool new_is_definition, bool new_is_extern,
                      SemIR::ClassId prev_class_id, bool prev_is_extern,
                      SemIR::ImportIRInstId prev_import_ir_inst_id) -> bool;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CLASS_H_

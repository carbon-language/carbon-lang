// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_IMPORT_REF_H_
#define CARBON_TOOLCHAIN_CHECK_IMPORT_REF_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Sets the IR for ImportIRId::ApiForImpl. Should be called before AddImportIR
// in order to ensure the correct ID is assigned.
auto SetApiImportIR(Context& context, SemIR::ImportIR import_ir) -> void;

// Adds an ImportIR, returning the ID. May use an existing ID if already added.
auto AddImportIR(Context& context, SemIR::ImportIR import_ir)
    -> SemIR::ImportIRId;

// Adds an import_ref instruction for the specified instruction in the
// specified IR. The import_ref is initially marked as unused.
auto AddImportRef(Context& context, SemIR::ImportIRInst import_ir_inst,
                  SemIR::BindNameId bind_name_id) -> SemIR::InstId;

// If the passed in instruction ID is an ImportRefUnloaded, turns it into an
// ImportRefLoaded for use.
auto LoadImportRef(Context& context, SemIR::InstId inst_id) -> void;

// Load all impls declared in IRs imported into this context.
auto ImportImpls(Context& context) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_IMPORT_REF_H_

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
                  SemIR::EntityNameId entity_name_id) -> SemIR::InstId;

// Returns the canonical IR inst for an entity. Returns an invalid ir_id for the
// current IR.
auto GetCanonicalImportIRInst(Context& context, const SemIR::File* cursor_ir,
                              SemIR::InstId cursor_inst_id)
    -> SemIR::ImportIRInst;

// Verifies a new instruction is the same as a previous instruction.
// prev_import_ir_inst should come from GetCanonicalImportIRInst.
auto VerifySameCanonicalImportIRInst(Context& context, SemIR::InstId prev_id,
                                     SemIR::ImportIRInst prev_import_ir_inst,
                                     SemIR::ImportIRId new_ir_id,
                                     const SemIR::File* new_import_ir,
                                     SemIR::InstId new_inst_id) -> void;

// If the passed in instruction ID is an ImportRefUnloaded, turns it into an
// ImportRefLoaded for use.
auto LoadImportRef(Context& context, SemIR::InstId inst_id) -> void;

// Load all impls declared in the api file corresponding to this impl file.
auto ImportImplsFromApiFile(Context& context) -> void;

// Load a specific impl declared in an imported IR.
auto ImportImpl(Context& context, SemIR::ImportIRId import_ir_id,
                SemIR::ImplId impl_id) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_IMPORT_REF_H_

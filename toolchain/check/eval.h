// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_EVAL_H_
#define CARBON_TOOLCHAIN_CHECK_EVAL_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

// Adds a `ConstantId` for a constant that has been imported from another IR.
// Does not evaluate the instruction, instead trusting that it is already in a
// suitable form, but does canonicalize the operands if necessary.
// TODO: Rely on import to canonicalize the operands to avoid this work.
auto AddImportedConstant(Context& context, SemIR::Inst inst)
    -> SemIR::ConstantId;

// Determines the phase of the instruction `inst`, and returns its constant
// value if it has constant phase. If it has runtime phase, returns
// `SemIR::ConstantId::NotConstant`.
auto TryEvalInst(Context& context, SemIR::InstId inst_id, SemIR::Inst inst)
    -> SemIR::ConstantId;
// Like the above but specific a LocId instead of deriving it from the
// `inst_id`. This is most useful when passing `None` as the `inst_id`.
auto TryEvalInst(Context& context, SemIR::LocId loc_id, SemIR::InstId inst_id,
                 SemIR::Inst inst) -> SemIR::ConstantId;

// Evaluates the eval block for a region of a specific. Produces a block
// containing the evaluated constant values of the instructions in the eval
// block.
auto TryEvalBlockForSpecific(Context& context, SemIRLoc loc,
                             SemIR::SpecificId specific_id,
                             SemIR::GenericInstIndex::Region region)
    -> SemIR::InstBlockId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_EVAL_H_

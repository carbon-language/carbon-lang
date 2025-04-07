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

// Evaluates the instruction `inst`. If `inst_id` is specified, it is the ID of
// the instruction; otherwise, evaluation of the instruction must not require an
// `InstId` to be provided.
auto TryEvalInstUnsafe(Context& context, SemIR::InstId inst_id,
                       SemIR::Inst inst) -> SemIR::ConstantId;

// Determines the phase of the instruction `inst_id`, and returns its constant
// value if it has constant phase. If it has runtime phase, returns
// `SemIR::ConstantId::NotConstant`.
inline auto TryEvalInst(Context& context, SemIR::InstId inst_id)
    -> SemIR::ConstantId {
  return TryEvalInstUnsafe(context, inst_id, context.insts().Get(inst_id));
}

// Same, but for a typed instruction that doesn't have an InstId assigned yet,
// in the case where evaluation doesn't need an InstId. This can be used to
// avoid allocating an instruction in the case where you just want a constant
// value and the instruction is known to not matter. However, even then care
// should be taken: if the produced constant is symbolic, you may still need an
// instruction to associate the constant with the enclosing generic.
template <typename InstT>
  requires(!InstT::Kind.constant_needs_inst_id())
auto TryEvalInst(Context& context, InstT inst) -> SemIR::ConstantId {
  return TryEvalInstUnsafe(context, SemIR::InstId::None, inst);
}

// Evaluates the eval block for a region of a specific. Produces a block
// containing the evaluated constant values of the instructions in the eval
// block.
auto TryEvalBlockForSpecific(Context& context, SemIRLoc loc,
                             SemIR::SpecificId specific_id,
                             SemIR::GenericInstIndex::Region region)
    -> SemIR::InstBlockId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_EVAL_H_

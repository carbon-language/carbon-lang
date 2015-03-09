//===---- MipsModuleISelDAGToDAG.h -  Change Subtarget             --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a pass used to change the subtarget for the
// Mips Instruction selector.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MIPS_MIPSMODULEISELDAGTODAG_H
#define LLVM_LIB_TARGET_MIPS_MIPSMODULEISELDAGTODAG_H

//===----------------------------------------------------------------------===//
// Instruction Selector Implementation
//===----------------------------------------------------------------------===//

namespace llvm {
class FunctionPass;
class MipsTargetMachine;

/// createMipsISelDag - This pass converts a legalized DAG into a
/// MIPS-specific DAG, ready for instruction scheduling.
FunctionPass *createMipsModuleISelDag(MipsTargetMachine &TM);
}

#endif

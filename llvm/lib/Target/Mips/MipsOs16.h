//===---- MipsOs16.h for Mips Option -Os16                         --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an optimization phase for the MIPS target.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MIPS_MIPSOS16_H
#define LLVM_LIB_TARGET_MIPS_MIPSOS16_H

namespace llvm {
class MipsTargetMachine;
class ModulePass;

ModulePass *createMipsOs16(MipsTargetMachine &TM);
}

#endif

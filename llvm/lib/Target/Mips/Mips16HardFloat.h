//===---- Mips16HardFloat.h for Mips16 Hard Float                  --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a phase which implements part of the floating point
// interoperability between Mips16 and Mips32 code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MIPS_MIPS16HARDFLOAT_H
#define LLVM_LIB_TARGET_MIPS_MIPS16HARDFLOAT_H

namespace llvm {
class MipsTargetMachine;
class ModulePass;

ModulePass *createMips16HardFloat(MipsTargetMachine &TM);
}

#endif

//==-- AArch64.h - Top-level interface for AArch64 representation -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// AArch64 back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_AARCH64_H
#define LLVM_TARGET_AARCH64_H

#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class AArch64AsmPrinter;
class FunctionPass;
class AArch64TargetMachine;
class MachineInstr;
class MCInst;

FunctionPass *createAArch64ISelDAG(AArch64TargetMachine &TM,
                                   CodeGenOpt::Level OptLevel);

FunctionPass *createAArch64CleanupLocalDynamicTLSPass();

FunctionPass *createAArch64BranchFixupPass();

void LowerAArch64MachineInstrToMCInst(const MachineInstr *MI, MCInst &OutMI,
                                      AArch64AsmPrinter &AP);


}

#endif

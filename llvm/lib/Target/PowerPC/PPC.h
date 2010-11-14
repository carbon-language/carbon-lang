//===-- PPC.h - Top-level interface for PowerPC Target ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// PowerPC back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_POWERPC_H
#define LLVM_TARGET_POWERPC_H

// GCC #defines PPC on Linux but we use it as our namespace name
#undef PPC

namespace llvm {
  class PPCTargetMachine;
  class FunctionPass;
  class formatted_raw_ostream;
  class JITCodeEmitter;
  class Target;
  class MachineInstr;
  class MCInst;
  class AsmPrinter;
  
FunctionPass *createPPCBranchSelectionPass();
FunctionPass *createPPCISelDag(PPCTargetMachine &TM);
FunctionPass *createPPCJITCodeEmitterPass(PPCTargetMachine &TM,
                                          JITCodeEmitter &MCE);

void LowerPPCMachineInstrToMCInst(const MachineInstr *MI, MCInst &OutMI,
                                  AsmPrinter &AP);

extern Target ThePPC32Target;
extern Target ThePPC64Target;

} // end namespace llvm;

// Defines symbolic names for PowerPC registers.  This defines a mapping from
// register name to register number.
//
#include "PPCGenRegisterNames.inc"

// Defines symbolic names for the PowerPC instructions.
//
#include "PPCGenInstrNames.inc"

#endif

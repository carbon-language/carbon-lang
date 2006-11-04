//===-- PPC.h - Top-level interface for PowerPC Target ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// PowerPC back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_POWERPC_H
#define LLVM_TARGET_POWERPC_H

#include <iosfwd>


// GCC #defines PPC on Linux but we use it as our namespace name
#undef PPC

namespace llvm {
  class PPCTargetMachine;
  class FunctionPassManager;
  class FunctionPass;
  class MachineCodeEmitter;
  
  namespace PPC {
    /// Predicate - These are "(BO << 5) | BI"  for various predicates.
    enum Predicate {
      PRED_ALWAYS = (20 << 5) | 0,
      PRED_LT     = (12 << 5) | 0,
      PRED_LE     = ( 4 << 5) | 1,
      PRED_EQ     = (12 << 5) | 2,
      PRED_GE     = ( 4 << 5) | 0,
      PRED_GT     = (12 << 5) | 1,
      PRED_NE     = ( 4 << 5) | 2,
      PRED_UN     = (12 << 5) | 3,
      PRED_NU     = ( 4 << 5) | 3
    };
  }
  
FunctionPass *createPPCBranchSelectionPass();
FunctionPass *createPPCISelDag(PPCTargetMachine &TM);
FunctionPass *createPPCAsmPrinterPass(std::ostream &OS,
                                      PPCTargetMachine &TM);
FunctionPass *createPPCCodeEmitterPass(PPCTargetMachine &TM,
                                       MachineCodeEmitter &MCE);
void addPPCMachOObjectWriterPass(FunctionPassManager &FPM, std::ostream &o, 
                                 PPCTargetMachine &tm);
} // end namespace llvm;

// Defines symbolic names for PowerPC registers.  This defines a mapping from
// register name to register number.
//
#include "PPCGenRegisterNames.inc"

// Defines symbolic names for the PowerPC instructions.
//
#include "PPCGenInstrNames.inc"

#endif

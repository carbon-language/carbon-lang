//===-- SPU.h - Top-level interface for Cell SPU Target ----------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// Cell SPU back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_IBMCELLSPU_H
#define LLVM_TARGET_IBMCELLSPU_H

#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class SPUTargetMachine;
  class FunctionPass;
  class formatted_raw_ostream;

  FunctionPass *createSPUISelDag(SPUTargetMachine &TM);

  extern Target TheCellSPUTarget;
}

// Defines symbolic names for the SPU instructions.
//
#include "SPUGenInstrNames.inc"

#endif /* LLVM_TARGET_IBMCELLSPU_H */

//=== Blackfin.h - Top-level interface for Blackfin backend -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// Blackfin back-end.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_BLACKFIN_H
#define TARGET_BLACKFIN_H

#include "llvm/Target/TargetMachine.h"

namespace llvm {

  class FunctionPass;
  class BlackfinTargetMachine;

  FunctionPass *createBlackfinISelDag(BlackfinTargetMachine &TM,
                                      CodeGenOpt::Level OptLevel);
  extern Target TheBlackfinTarget;

} // end namespace llvm

// Defines symbolic names for Blackfin registers.  This defines a mapping from
// register name to register number.
#include "BlackfinGenRegisterNames.inc"

// Defines symbolic names for the Blackfin instructions.
#include "BlackfinGenInstrNames.inc"

#endif

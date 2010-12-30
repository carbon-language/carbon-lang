//===-- PTX.h - Top-level interface for PTX representation ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// PTX back-end.
//
//===----------------------------------------------------------------------===//

#ifndef PTX_H
#define PTX_H

#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class PTXTargetMachine;
  class FunctionPass;

  namespace PTX {
    enum StateSpace {
      GLOBAL = 0, // default to global state space
      CONSTANT = 1,
      LOCAL = 2,
      PARAMETER = 3,
      SHARED = 4
    };
  } // namespace PTX

  FunctionPass *createPTXISelDag(PTXTargetMachine &TM,
                                 CodeGenOpt::Level OptLevel);

  FunctionPass *createPTXMFInfoExtract(PTXTargetMachine &TM,
                                       CodeGenOpt::Level OptLevel);

  extern Target ThePTXTarget;
} // namespace llvm;

// Defines symbolic names for PTX registers.
#include "PTXGenRegisterNames.inc"

// Defines symbolic names for the PTX instructions.
#include "PTXGenInstrNames.inc"

#endif // PTX_H

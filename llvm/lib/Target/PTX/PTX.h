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

#include "MCTargetDesc/PTXMCTargetDesc.h"
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

    enum Predicate {
      PRED_NORMAL = 0,
      PRED_NEGATE = 1
    };
  } // namespace PTX

  FunctionPass *createPTXISelDag(PTXTargetMachine &TM,
                                 CodeGenOpt::Level OptLevel);

  FunctionPass *createPTXMFInfoExtract(PTXTargetMachine &TM,
                                       CodeGenOpt::Level OptLevel);

  FunctionPass *createPTXRegisterAllocator();

} // namespace llvm;

#endif // PTX_H

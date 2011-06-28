//===-- Alpha.h - Top-level interface for Alpha representation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// Alpha back-end.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_ALPHA_H
#define TARGET_ALPHA_H

#include "llvm/Target/TargetMachine.h"

namespace llvm {
  namespace Alpha {
    // These describe LDAx

    static const int IMM_LOW  = -32768;
    static const int IMM_HIGH = 32767;
    static const int IMM_MULT = 65536;
  }

  class AlphaTargetMachine;
  class FunctionPass;
  class formatted_raw_ostream;

  FunctionPass *createAlphaISelDag(AlphaTargetMachine &TM);
  FunctionPass *createAlphaPatternInstructionSelector(TargetMachine &TM);
  FunctionPass *createAlphaJITCodeEmitterPass(AlphaTargetMachine &TM,
                                              JITCodeEmitter &JCE);
  FunctionPass *createAlphaLLRPPass(AlphaTargetMachine &tm);
  FunctionPass *createAlphaBranchSelectionPass();

  extern Target TheAlphaTarget;

} // end namespace llvm;

// Defines symbolic names for Alpha registers.  This defines a mapping from
// register name to register number.
//

#define GET_REGINFO_ENUM
#include "AlphaGenRegisterInfo.inc"

// Defines symbolic names for the Alpha instructions.
//
#define GET_INSTRINFO_ENUM
#include "AlphaGenInstrInfo.inc"

#endif

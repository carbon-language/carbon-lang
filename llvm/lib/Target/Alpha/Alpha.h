//===-- Alpha.h - Top-level interface for Alpha representation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// Alpha back-end.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_ALPHA_H
#define TARGET_ALPHA_H

#include <iosfwd>

namespace llvm {

  class AlphaTargetMachine;
  class FunctionPass;
  class TargetMachine;
  class MachineCodeEmitter;

  FunctionPass *createAlphaSimpleInstructionSelector(TargetMachine &TM);
  FunctionPass *createAlphaISelDag(TargetMachine &TM);
  FunctionPass *createAlphaCodePrinterPass(std::ostream &OS,
                                             TargetMachine &TM);
  FunctionPass *createAlphaPatternInstructionSelector(TargetMachine &TM);
  FunctionPass *createAlphaCodeEmitterPass(AlphaTargetMachine &TM,
                                           MachineCodeEmitter &MCE);
} // end namespace llvm;

// Defines symbolic names for Alpha registers.  This defines a mapping from
// register name to register number.
//
#include "AlphaGenRegisterNames.inc"

// Defines symbolic names for the Alpha instructions.
//
#include "AlphaGenInstrNames.inc"

#endif

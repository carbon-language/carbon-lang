//===-- Mips.h - Top-level interface for Mips representation ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bruno Cardoso Lopes and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in 
// the LLVM Mips back-end.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_MIPS_H
#define TARGET_MIPS_H

#include <iosfwd>

namespace llvm {
  class MipsTargetMachine;
  class FunctionPassManager;
  class FunctionPass;
  class MachineCodeEmitter;

  FunctionPass *createMipsISelDag(MipsTargetMachine &TM);
  FunctionPass *createMipsDelaySlotFillerPass(MipsTargetMachine &TM);
  FunctionPass *createMipsCodePrinterPass(std::ostream &OS, 
                                          MipsTargetMachine &TM);
} // end namespace llvm;

// Defines symbolic names for Mips registers.  This defines a mapping from
// register name to register number.
#include "MipsGenRegisterNames.inc"

// Defines symbolic names for the Mips instructions.
#include "MipsGenInstrNames.inc"

#endif

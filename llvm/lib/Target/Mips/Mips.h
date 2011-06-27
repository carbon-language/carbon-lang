//===-- Mips.h - Top-level interface for Mips representation ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in
// the LLVM Mips back-end.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_MIPS_H
#define TARGET_MIPS_H

#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class MipsTargetMachine;
  class FunctionPass;
  class MachineCodeEmitter;
  class formatted_raw_ostream;

  FunctionPass *createMipsISelDag(MipsTargetMachine &TM);
  FunctionPass *createMipsDelaySlotFillerPass(MipsTargetMachine &TM);
  FunctionPass *createMipsExpandPseudoPass(MipsTargetMachine &TM);
  FunctionPass *createMipsEmitGPRestorePass(MipsTargetMachine &TM);

  extern Target TheMipsTarget;
  extern Target TheMipselTarget;

} // end namespace llvm;

// Defines symbolic names for Mips registers.  This defines a mapping from
// register name to register number.
#define GET_REGINFO_ENUM
#include "MipsGenRegisterInfo.inc"

// Defines symbolic names for the Mips instructions.
#include "MipsGenInstrNames.inc"

#endif

//==-- MSP430.h - Top-level interface for MSP430 representation --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in
// the LLVM MSP430 backend.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_MSP430_H
#define LLVM_TARGET_MSP430_H

#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class MSP430TargetMachine;
  class FunctionPass;
  class formatted_raw_ostream;

  FunctionPass *createMSP430ISelDag(MSP430TargetMachine &TM,
                                    CodeGenOpt::Level OptLevel);

  extern Target TheMSP430Target;

} // end namespace llvm;

// Defines symbolic names for MSP430 registers.
// This defines a mapping from register name to register number.
#include "MSP430GenRegisterNames.inc"

// Defines symbolic names for the MSP430 instructions.
#include "MSP430GenInstrNames.inc"

#endif

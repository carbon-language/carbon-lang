//===-- ARM.h - Top-level interface for ARM representation---- --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the "Instituto Nokia de Tecnologia" and
// is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// ARM back-end.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_ARM_H
#define TARGET_ARM_H

#include <iosfwd>
#include <cassert>

namespace llvm {
  // Enums corresponding to ARM condition codes
  namespace ARMCC {
    enum CondCodes {
      NE,
      EQ
    };
  }

  static const char *ARMCondCodeToString(ARMCC::CondCodes CC) {
    switch (CC) {
    default: assert(0 && "Unknown condition code");
    case ARMCC::NE:  return "ne";
    case ARMCC::EQ:  return "eq";
    }
  }

  class FunctionPass;
  class TargetMachine;

  FunctionPass *createARMISelDag(TargetMachine &TM);
  FunctionPass *createARMCodePrinterPass(std::ostream &OS, TargetMachine &TM);
} // end namespace llvm;

// Defines symbolic names for ARM registers.  This defines a mapping from
// register name to register number.
//
#include "ARMGenRegisterNames.inc"

// Defines symbolic names for the ARM instructions.
//
#include "ARMGenInstrNames.inc"


#endif

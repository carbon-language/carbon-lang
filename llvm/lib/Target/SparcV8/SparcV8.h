//===-- SparcV8.h - Top-level interface for SparcV8 representation -*- C++ -*-//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// SparcV8 back-end.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_SPARCV8_H
#define TARGET_SPARCV8_H

#include <iosfwd>

namespace llvm {

  class FunctionPass;
  class TargetMachine;

  FunctionPass *createSparcV8SimpleInstructionSelector(TargetMachine &TM);
  FunctionPass *createSparcV8CodePrinterPass(std::ostream &OS,
                                             TargetMachine &TM);
  FunctionPass *createSparcV8DelaySlotFillerPass(TargetMachine &TM);

} // end namespace llvm;

// Defines symbolic names for SparcV8 registers.  This defines a mapping from
// register name to register number.
//
#include "SparcV8GenRegisterNames.inc"

// Defines symbolic names for the SparcV8 instructions.
//
#include "SparcV8GenInstrNames.inc"

#endif

//===-- SPU.h - Top-level interface for Cell SPU Target ----------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by a team from the Computer Systems Research
// Department at The Aerospace Corporation and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// Cell SPU back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_IBMCELLSPU_H
#define LLVM_TARGET_IBMCELLSPU_H

#include <iosfwd>

namespace llvm {
  class SPUTargetMachine;
  class FunctionPass;

  FunctionPass *createSPUISelDag(SPUTargetMachine &TM);
  FunctionPass *createSPUAsmPrinterPass(std::ostream &o, SPUTargetMachine &tm);

  /* Utility functions/predicates/etc used all over the place: */
  //! Predicate test for a signed 10-bit value
  /*!
    \param Value The input value to be tested

    This predicate tests for a signed 10-bit value, returning the 10-bit value
    as a short if true.
   */
  inline bool isS10Constant(short Value) {
    int SExtValue = ((int) Value << (32 - 10)) >> (32 - 10);
    return ((Value > 0 && Value <= (1 << 9) - 1)
	    || (Value < 0 && (short) SExtValue == Value));
  }

  inline bool isS10Constant(int Value) {
    return (Value >= -(1 << 9) && Value <= (1 << 9) - 1);
  }

  inline bool isS10Constant(uint32_t Value) {
    return (Value <= ((1 << 9) - 1));
  }

  inline bool isS10Constant(int64_t Value) {
    return (Value >= -(1 << 9) && Value <= (1 << 9) - 1);
  }

  inline bool isS10Constant(uint64_t Value) {
    return (Value <= ((1 << 9) - 1));
  }
}

// Defines symbolic names for the SPU instructions.
//
#include "SPUGenInstrNames.inc"

#endif /* LLVM_TARGET_IBMCELLSPU_H */

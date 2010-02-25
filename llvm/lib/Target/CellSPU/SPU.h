//===-- SPU.h - Top-level interface for Cell SPU Target ----------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// Cell SPU back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_IBMCELLSPU_H
#define LLVM_TARGET_IBMCELLSPU_H

#include "llvm/System/DataTypes.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class SPUTargetMachine;
  class FunctionPass;
  class formatted_raw_ostream;

  FunctionPass *createSPUISelDag(SPUTargetMachine &TM);

  /*--== Utility functions/predicates/etc used all over the place: --==*/
  //! Predicate test for a signed 10-bit value
  /*!
    \param Value The input value to be tested

    This predicate tests for a signed 10-bit value, returning the 10-bit value
    as a short if true.
   */
  template<typename T>
  inline bool isS10Constant(T Value);

  template<>
  inline bool isS10Constant<short>(short Value) {
    int SExtValue = ((int) Value << (32 - 10)) >> (32 - 10);
    return ((Value > 0 && Value <= (1 << 9) - 1)
            || (Value < 0 && (short) SExtValue == Value));
  }

  template<>
  inline bool isS10Constant<int>(int Value) {
    return (Value >= -(1 << 9) && Value <= (1 << 9) - 1);
  }

  template<>
  inline bool isS10Constant<uint32_t>(uint32_t Value) {
    return (Value <= ((1 << 9) - 1));
  }

  template<>
  inline bool isS10Constant<int64_t>(int64_t Value) {
    return (Value >= -(1 << 9) && Value <= (1 << 9) - 1);
  }

  template<>
  inline bool isS10Constant<uint64_t>(uint64_t Value) {
    return (Value <= ((1 << 9) - 1));
  }

  //! Predicate test for an unsigned 10-bit value
  /*!
    \param Value The input value to be tested
   */
  inline bool isU10Constant(short Value) {
    return (Value == (Value & 0x3ff));
  }

  inline bool isU10Constant(int Value) {
    return (Value == (Value & 0x3ff));
  }

  inline bool isU10Constant(uint32_t Value) {
    return (Value == (Value & 0x3ff));
  }

  inline bool isU10Constant(int64_t Value) {
    return (Value == (Value & 0x3ff));
  }

  inline bool isU10Constant(uint64_t Value) {
    return (Value == (Value & 0x3ff));
  }

  //! Predicate test for a signed 14-bit value
  /*!
    \param Value The input value to be tested
   */
  template<typename T>
  inline bool isS14Constant(T Value);

  template<>
  inline bool isS14Constant<short>(short Value) {
    return (Value >= -(1 << 13) && Value <= (1 << 13) - 1);
  }

  template<>
  inline bool isS14Constant<int>(int Value) {
    return (Value >= -(1 << 13) && Value <= (1 << 13) - 1);
  }

  template<>
  inline bool isS14Constant<uint32_t>(uint32_t Value) {
    return (Value <= ((1 << 13) - 1));
  }

  template<>
  inline bool isS14Constant<int64_t>(int64_t Value) {
    return (Value >= -(1 << 13) && Value <= (1 << 13) - 1);
  }

  template<>
  inline bool isS14Constant<uint64_t>(uint64_t Value) {
    return (Value <= ((1 << 13) - 1));
  }

  //! Predicate test for a signed 16-bit value
  /*!
    \param Value The input value to be tested
   */
  template<typename T>
  inline bool isS16Constant(T Value);

  template<>
  inline bool isS16Constant<short>(short Value) {
    return true;
  }

  template<>
  inline bool isS16Constant<int>(int Value) {
    return (Value >= -(1 << 15) && Value <= (1 << 15) - 1);
  }

  template<>
  inline bool isS16Constant<uint32_t>(uint32_t Value) {
    return (Value <= ((1 << 15) - 1));
  }

  template<>
  inline bool isS16Constant<int64_t>(int64_t Value) {
    return (Value >= -(1 << 15) && Value <= (1 << 15) - 1);
  }

  template<>
  inline bool isS16Constant<uint64_t>(uint64_t Value) {
    return (Value <= ((1 << 15) - 1));
  }

  extern Target TheCellSPUTarget;

}

// Defines symbolic names for the SPU instructions.
//
#include "SPUGenInstrNames.inc"

#endif /* LLVM_TARGET_IBMCELLSPU_H */

//===-- PTXBaseInfo.h - Top level definitions for PTX -------- --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone helper functions and enum definitions for
// the PTX target useful for the compiler back-end and the MC libraries.
// As such, it deliberately does not include references to LLVM core
// code gen types, passes, etc..
//
//===----------------------------------------------------------------------===//

#ifndef PTXBASEINFO_H
#define PTXBASEINFO_H

#include "PTXMCTargetDesc.h"

namespace llvm {
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
      PRED_NEGATE = 1,
      PRED_NONE   = 2
    };
  } // namespace PTX

  /// Namespace to hold all target-specific flags.
  namespace PTXRoundingMode {
    // Instruction Flags
    enum {
      // Rounding Mode Flags
      RndMask             = 15,
      RndDefault          =  0, // ---
      RndNone             =  1, // <NONE>
      RndNearestEven      =  2, // .rn
      RndTowardsZero      =  3, // .rz
      RndNegInf           =  4, // .rm
      RndPosInf           =  5, // .rp
      RndApprox           =  6, // .approx
      RndNearestEvenInt   =  7, // .rni
      RndTowardsZeroInt   =  8, // .rzi
      RndNegInfInt        =  9, // .rmi
      RndPosInfInt        = 10  // .rpi
    };
  } // namespace PTXII
} // namespace llvm

#endif


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
  namespace PTXStateSpace {
    enum {
      Global    = 0, // default to global state space
      Constant  = 1,
      Local     = 2,
      Parameter = 3,
      Shared    = 4
    };
  } // namespace PTXStateSpace

  namespace PTXPredicate {
    enum {
      Normal = 0,
      Negate = 1,
      None   = 2
    };
  } // namespace PTXPredicate

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

  namespace PTXRegisterType {
    // Register type encoded in MCOperands
    enum {
      Pred  = 0,
      B16,
      B32,
      B64,
      F32,
      F64
    };
  } // namespace PTXRegisterType
} // namespace llvm

#endif


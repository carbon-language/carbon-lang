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

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
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

  namespace PTXRegisterSpace {
    // Register space encoded in MCOperands
    enum {
      Reg = 0,
      Local,
      Param,
      Argument,
      Return
    };
  }

  inline static void decodeRegisterName(raw_ostream &OS,
                                        unsigned EncodedReg) {
    OS << "%";

    unsigned RegSpace  = EncodedReg & 0x7;
    unsigned RegType   = (EncodedReg >> 3) & 0x7;
    unsigned RegOffset = EncodedReg >> 6;

    switch (RegSpace) {
    default:
      llvm_unreachable("Unknown register space!");
    case PTXRegisterSpace::Reg:
      switch (RegType) {
      default:
        llvm_unreachable("Unknown register type!");
      case PTXRegisterType::Pred:
        OS << "p";
        break;
      case PTXRegisterType::B16:
        OS << "rh";
        break;
      case PTXRegisterType::B32:
        OS << "r";
        break;
      case PTXRegisterType::B64:
        OS << "rd";
        break;
      case PTXRegisterType::F32:
        OS << "f";
        break;
      case PTXRegisterType::F64:
        OS << "fd";
        break;
      }
      break;
    case PTXRegisterSpace::Return:
      OS << "ret";
      break;
    case PTXRegisterSpace::Argument:
      OS << "arg";
      break;
    }

    OS << RegOffset;
  }
} // namespace llvm

#endif


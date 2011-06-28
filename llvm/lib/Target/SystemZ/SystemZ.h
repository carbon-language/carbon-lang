//=-- SystemZ.h - Top-level interface for SystemZ representation -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in
// the LLVM SystemZ backend.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_SystemZ_H
#define LLVM_TARGET_SystemZ_H

#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class SystemZTargetMachine;
  class FunctionPass;
  class formatted_raw_ostream;

  namespace SystemZCC {
    // SystemZ specific condition code. These correspond to SYSTEMZ_*_COND in
    // SystemZInstrInfo.td. They must be kept in synch.
    enum CondCodes {
      O   = 0,
      H   = 1,
      NLE = 2,
      L   = 3,
      NHE = 4,
      LH  = 5,
      NE  = 6,
      E   = 7,
      NLH = 8,
      HE  = 9,
      NL  = 10,
      LE  = 11,
      NH  = 12,
      NO  = 13,
      INVALID = -1
    };
  }

  FunctionPass *createSystemZISelDag(SystemZTargetMachine &TM,
                                    CodeGenOpt::Level OptLevel);

  extern Target TheSystemZTarget;

} // end namespace llvm;

// Defines symbolic names for SystemZ registers.
// This defines a mapping from register name to register number.
#define GET_REGINFO_ENUM
#include "SystemZGenRegisterInfo.inc"

// Defines symbolic names for the SystemZ instructions.
#define GET_INSTRINFO_ENUM
#include "SystemZGenInstrInfo.inc"

#endif

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
  class raw_ostream;

  namespace SystemZCC {
    // SystemZ specific condition code. These correspond to SYSTEMZ_*_COND in
    // SystemZInstrInfo.td. They must be kept in synch.
    enum CondCodes {
      E  = 0,
      NE = 1,
      H  = 2,
      L  = 3,
      HE = 4,
      LE = 5
    };
  }

  FunctionPass *createSystemZISelDag(SystemZTargetMachine &TM,
                                    CodeGenOpt::Level OptLevel);
  FunctionPass *createSystemZCodePrinterPass(raw_ostream &o,
                                            SystemZTargetMachine &tm,
                                            CodeGenOpt::Level OptLevel,
                                            bool verbose);

} // end namespace llvm;

// Defines symbolic names for SystemZ registers.
// This defines a mapping from register name to register number.
#include "SystemZGenRegisterNames.inc"

// Defines symbolic names for the SystemZ instructions.
#include "SystemZGenInstrNames.inc"

#endif

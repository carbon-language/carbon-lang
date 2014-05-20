//===- lib/MC/MCTargetOptions.cpp - MC Target Options --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCTargetOptions.h"

namespace llvm {

MCTargetOptions::MCTargetOptions()
    : SanitizeAddress(false), MCRelaxAll(false), MCNoExecStack(false),
      MCSaveTempLabels(false), MCUseDwarfDirectory(false),
      ShowMCEncoding(false), ShowMCInst(false), AsmVerbose(false) {}

} // end namespace llvm

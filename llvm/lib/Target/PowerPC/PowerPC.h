//===-- PowerPC.h - Top-level interface for PowerPC representation -*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// PowerPC back-end.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_POWERPC_H
#define TARGET_POWERPC_H

#include <iosfwd>

namespace llvm {

class FunctionPass;
class TargetMachine;

enum PPCTargetEnum {
  TargetDefault, TargetAIX, TargetDarwin
};

FunctionPass *createPPCBranchSelectionPass();
FunctionPass *createPPC32ISelPattern(TargetMachine &TM);
FunctionPass *createPPC32ISelDag(TargetMachine &TM);
FunctionPass *createDarwinAsmPrinter(std::ostream &OS, TargetMachine &TM);
FunctionPass *createAIXAsmPrinter(std::ostream &OS, TargetMachine &TM);

extern bool PICEnabled;
extern PPCTargetEnum PPCTarget;
} // end namespace llvm;

// GCC #defines PPC on Linux but we use it as our namespace name
#undef PPC

// Defines symbolic names for PowerPC registers.  This defines a mapping from
// register name to register number.
//
#include "PowerPCGenRegisterNames.inc"

// Defines symbolic names for the PowerPC instructions.
//
#include "PowerPCGenInstrNames.inc"

#endif

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

FunctionPass *createPowerPCPEI();
FunctionPass *createPPCBranchSelectionPass();
FunctionPass *createPPC32ISelSimple(TargetMachine &TM);
FunctionPass *createPPC32AsmPrinter(std::ostream &OS,TargetMachine &TM);
FunctionPass *createPPC64ISelSimple(TargetMachine &TM);
FunctionPass *createPPC64AsmPrinter(std::ostream &OS,TargetMachine &TM);

} // end namespace llvm;

// Defines symbolic names for PowerPC registers.  This defines a mapping from
// register name to register number.
//
#include "PowerPCGenRegisterNames.inc"

// Defines symbolic names for the PowerPC instructions.
//
#include "PowerPCGenInstrNames.inc"

#endif

//===-- PPC32.h - Top-level interface for 32-bit PowerPC -----------*- C++ -*-//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// Darwin/PowerPC back-end.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_POWERPC32_H
#define TARGET_POWERPC32_H

#include "PowerPC.h"
#include <iosfwd>

namespace llvm {

class FunctionPass;
class TargetMachine;

FunctionPass *createPPC32ISelSimple(TargetMachine &TM);
FunctionPass *createPPC32AsmPrinter(std::ostream &OS,TargetMachine &TM);

} // end namespace llvm;

#endif

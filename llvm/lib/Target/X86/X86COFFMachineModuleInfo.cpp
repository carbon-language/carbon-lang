//===-- llvm/CodeGen/X86COFFMachineModuleInfo.cpp -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is an MMI implementation for X86 COFF (windows) targets.
//
//===----------------------------------------------------------------------===//

#include "X86COFFMachineModuleInfo.h"
#include "X86MachineFunctionInfo.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;


X86COFFMachineModuleInfo::~X86COFFMachineModuleInfo() {
}


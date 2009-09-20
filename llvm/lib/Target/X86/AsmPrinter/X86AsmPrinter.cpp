//===-- X86AsmPrinter.cpp - Convert X86 LLVM IR to X86 assembly -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file the shared super class printer that converts from our internal
// representation of machine-dependent LLVM code to Intel and AT&T format
// assembly language.
// This printer is the output mechanism used by `llc'.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86ATTAsmPrinter.h"
#include "X86ATTInstPrinter.h"
#include "X86IntelInstPrinter.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

/// createX86CodePrinterPass - Returns a pass that prints the X86 assembly code
/// for a MachineFunction to the given output stream, using the given target
/// machine description.
///
static AsmPrinter *createX86CodePrinterPass(formatted_raw_ostream &o,
                                            TargetMachine &tm,
                                            const MCAsmInfo *tai,
                                            bool verbose) {
  return new X86ATTAsmPrinter(o, tm, tai, verbose);
}


static MCInstPrinter *createX86MCInstPrinter(const Target &T,
                                             unsigned SyntaxVariant,
                                             const MCAsmInfo &MAI,
                                             raw_ostream &O) {
  if (SyntaxVariant == 0)
    return new X86ATTInstPrinter(O, MAI);
  if (SyntaxVariant == 1)
    return new X86IntelInstPrinter(O, MAI);
  return 0;
}

// Force static initialization.
extern "C" void LLVMInitializeX86AsmPrinter() { 
  TargetRegistry::RegisterAsmPrinter(TheX86_32Target, createX86CodePrinterPass);
  TargetRegistry::RegisterAsmPrinter(TheX86_64Target, createX86CodePrinterPass);
  
  TargetRegistry::RegisterMCInstPrinter(TheX86_32Target,createX86MCInstPrinter);
  TargetRegistry::RegisterMCInstPrinter(TheX86_64Target,createX86MCInstPrinter);
}

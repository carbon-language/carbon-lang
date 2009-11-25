//===- X86Disassembler.cpp - Disassembler for x86 and x86_64 ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCDisassembler.h"
#include "llvm/Target/TargetRegistry.h"
#include "X86.h"
using namespace llvm;

static const MCDisassembler *createX86_32Disassembler(const Target &T) {
  return 0;
}

static const MCDisassembler *createX86_64Disassembler(const Target &T) {
  return 0; 
}

extern "C" void LLVMInitializeX86Disassembler() { 
  // Register the disassembler.
  TargetRegistry::RegisterMCDisassembler(TheX86_32Target, 
                                         createX86_32Disassembler);
  TargetRegistry::RegisterMCDisassembler(TheX86_64Target,
                                         createX86_64Disassembler);
}

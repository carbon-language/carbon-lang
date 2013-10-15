//===-- AllTargetsBindings.cpp --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the llvm-c functions for initialization of
// different aspects of all configured targets.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Target.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm;

void LLVMInitializeAllTargetInfos(void) {
  InitializeAllTargetInfos();
}

void LLVMInitializeAllTargets(void) {
  InitializeAllTargets();
}

void LLVMInitializeAllTargetMCs(void) {
  InitializeAllTargetMCs();
}

void LLVMInitializeAllAsmPrinters(void) {
  InitializeAllAsmPrinters();
}

void LLVMInitializeAllAsmParsers(void) {
  InitializeAllAsmParsers();
}

void LLVMInitializeAllDisassemblers(void) {
  InitializeAllDisassemblers();
}

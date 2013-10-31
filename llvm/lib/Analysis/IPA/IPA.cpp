//===-- IPA.cpp -----------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the common initialization routines for the IPA library.
//
//===----------------------------------------------------------------------===//

#include "llvm/InitializePasses.h"
#include "llvm/PassRegistry.h"
#include "llvm-c/Initialization.h"

using namespace llvm;

/// initializeIPA - Initialize all passes linked into the IPA library.
void llvm::initializeIPA(PassRegistry &Registry) {
  initializeCallGraphPass(Registry);
  initializeCallGraphPrinterPass(Registry);
  initializeCallGraphViewerPass(Registry);
  initializeFindUsedTypesPass(Registry);
  initializeGlobalsModRefPass(Registry);
}

void LLVMInitializeIPA(LLVMPassRegistryRef R) {
  initializeIPA(*unwrap(R));
}

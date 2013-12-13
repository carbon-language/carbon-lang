//===-- CppBackendTargetInfo.cpp - CppBackend Target Implementation -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CPPTargetMachine.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

Target llvm::TheCppBackendTarget;

static bool CppBackend_TripleMatchQuality(Triple::ArchType Arch) {
  // This backend doesn't correspond to any architecture. It must be explicitly
  // selected with -march.
  return false;
}

extern "C" void LLVMInitializeCppBackendTargetInfo() { 
  TargetRegistry::RegisterTarget(TheCppBackendTarget, "cpp",    
                                  "C++ backend",
                                  &CppBackend_TripleMatchQuality);
}

extern "C" void LLVMInitializeCppBackendTargetMC() {}

//===-- CppBackendTargetInfo.cpp - CppBackend Target Implementation -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

Target llvm::TheCppBackendTarget;

static unsigned CppBackend_JITMatchQuality() {
  return 0;
}

static unsigned CppBackend_TripleMatchQuality(const std::string &TT) {
  // This class always works, but shouldn't be the default in most cases.
  return 1;
}

static unsigned CppBackend_ModuleMatchQuality(const Module &M) {
  // This class always works, but shouldn't be the default in most cases.
  return 1;
}

extern "C" void LLVMInitializeCppBackendTargetInfo() { 
  TargetRegistry::RegisterTarget(TheCppBackendTarget, "cpp",    
                                  "C++ backend",
                                  &CppBackend_TripleMatchQuality,
                                  &CppBackend_ModuleMatchQuality,
                                  &CppBackend_JITMatchQuality);
}

//===-- llvm/CodeGen/MachineDebugInfo.cpp -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineDebugInfo.h"

using namespace llvm;

// Handle the Pass registration stuff necessary to use TargetData's.
namespace {
  RegisterPass<MachineDebugInfo> X("machinedebuginfo", "Debug Information");
}

  
//===----------------------------------------------------------------------===//
/// doInitialization - Initialize the debug state for a new module.
///
bool MachineDebugInfo::doInitialization() {
  return false;
}

/// doFinalization - Tear down the debug state after completion of a module.
///
bool MachineDebugInfo::doFinalization() {
  return false;
}

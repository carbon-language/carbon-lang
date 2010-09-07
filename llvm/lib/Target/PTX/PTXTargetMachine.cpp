//===-- PTXTargetMachine.cpp - Define TargetMachine for PTX ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the PTX target.
//
//===----------------------------------------------------------------------===//

#include "PTX.h"
#include "PTXTargetMachine.h"
#include "llvm/Target/TargetRegistry.h"

using namespace llvm;

extern "C" void LLVMInitializePTXTarget()
{
  // Register the target
  RegisterTargetMachine<PTXTargetMachine> X(ThePTXTarget);
}

PTXTargetMachine::PTXTargetMachine(const Target &T,
                                   const std::string &TT,
                                   const std::string &FS) :
  LLVMTargetMachine(T, TT)
{
}

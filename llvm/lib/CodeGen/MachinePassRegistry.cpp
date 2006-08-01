//===-- MachineInstr.cpp --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachinePassRegistry.h"
#include <iostream>

using namespace llvm;

  
//===---------------------------------------------------------------------===//
///
/// RegisterRegAlloc class - Track the registration of register allocators.
///
//===---------------------------------------------------------------------===//
MachinePassRegistry<RegisterRegAlloc::FunctionPassCtor>
RegisterRegAlloc::Registry;


//===---------------------------------------------------------------------===//
///
/// RegisterScheduler class - Track the registration of instruction schedulers.
///
//===---------------------------------------------------------------------===//
MachinePassRegistry<RegisterScheduler::FunctionPassCtor>
RegisterScheduler::Registry;

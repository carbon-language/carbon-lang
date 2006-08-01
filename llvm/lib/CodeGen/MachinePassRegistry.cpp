//===-- CodeGen/MachineInstr.cpp ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the machine function pass registry for register allocators
// and instruction schedulers.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachinePassRegistry.h"

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

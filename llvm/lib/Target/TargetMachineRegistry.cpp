//===-- TargetMachineRegistry.cpp - Target Auto Registration Impl ---------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file exposes the RegisterTarget class, which TargetMachine
// implementations should use to register themselves with the system.  This file
// also exposes the TargetMachineRegistry class, which allows tools to inspect
// all of registered targets.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetMachineRegistry.h"
using namespace llvm;

const TargetMachineRegistry::Entry *TargetMachineRegistry::List = 0;


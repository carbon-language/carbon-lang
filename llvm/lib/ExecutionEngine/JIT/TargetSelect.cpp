//===-- TargetSelect.cpp - Target Chooser Code ----------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains the hideously gross code that is currently used to select
// a particular TargetMachine for the JIT to use.  This should obviously be
// improved in the future, probably by having the TargetMachines register
// themselves with the runtime, and then have them choose themselves if they
// match the current machine.
//
//===----------------------------------------------------------------------===//

#include "JIT.h"
#include "llvm/Module.h"
#include "llvm/ModuleProvider.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include <iostream>
using namespace llvm;

static cl::opt<const TargetMachineRegistry::Entry*, false, TargetNameParser>
MArch("march", cl::desc("Architecture to generate assembly for:"));

/// create - Create an return a new JIT compiler if there is one available
/// for the current target.  Otherwise, return null.
///
ExecutionEngine *JIT::create(ModuleProvider *MP, IntrinsicLowering *IL) {
  if (MArch == 0) {
    std::string Error;
    MArch = TargetMachineRegistry::getClosestTargetForJIT(Error);
    if (MArch == 0) return 0;
  } else if (MArch->JITMatchQualityFn() == 0) {
    std::cerr << "WARNING: This target JIT is not designed for the host you are"
              << " running.  If bad things happen, please choose a different "
              << "-march switch.\n";
  }

  // Allocate a target...
  TargetMachine *Target = MArch->CtorFn(*MP->getModule(), IL);
  assert(Target && "Could not allocate target machine!");

  // If the target supports JIT code generation, return a new JIT now.
  if (TargetJITInfo *TJ = Target->getJITInfo())
    return new JIT(MP, *Target, *TJ);
  return 0;
}

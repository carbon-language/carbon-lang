//===-- TargetSelect.cpp - Target Chooser Code ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This just asks the TargetMachineRegistry for the appropriate JIT to use, and
// allows the user to specify a specific one on the commandline with -march=x.
//
//===----------------------------------------------------------------------===//

#include "JIT.h"
#include "llvm/Module.h"
#include "llvm/ModuleProvider.h"
#include "llvm/Target/SubtargetFeature.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include <iostream>
using namespace llvm;

static cl::opt<const TargetMachineRegistry::Entry*, false, TargetNameParser>
MArch("march", cl::desc("Architecture to generate assembly for:"));

static cl::opt<std::string>
MCPU("mcpu", 
  cl::desc("Target a specific cpu type (-mcpu=help for details)"),
  cl::value_desc("cpu-name"),
  cl::init(""));

static cl::list<std::string>
MAttrs("mattr", 
  cl::CommaSeparated,
  cl::desc("Target specific attributes (-mattr=help for details)"),
  cl::value_desc("a1,+a2,-a3,..."));

/// create - Create an return a new JIT compiler if there is one available
/// for the current target.  Otherwise, return null.
///
ExecutionEngine *JIT::create(ModuleProvider *MP) {
  if (MArch == 0) {
    std::string Error;
    MArch = TargetMachineRegistry::getClosestTargetForJIT(Error);
    if (MArch == 0) return 0;
  } else if (MArch->JITMatchQualityFn() == 0) {
    std::cerr << "WARNING: This target JIT is not designed for the host you are"
              << " running.  If bad things happen, please choose a different "
              << "-march switch.\n";
  }

  // Package up features to be passed to target/subtarget
  std::string FeaturesStr;
  if (MCPU.size() || MAttrs.size()) {
    SubtargetFeatures Features;
    Features.setCPU(MCPU);
    for (unsigned i = 0; i != MAttrs.size(); ++i)
      Features.AddFeature(MAttrs[i]);
    FeaturesStr = Features.getString();
  }

  // Allocate a target...
  TargetMachine *Target = MArch->CtorFn(*MP->getModule(), 0, FeaturesStr);
  assert(Target && "Could not allocate target machine!");

  // If the target supports JIT code generation, return a new JIT now.
  if (TargetJITInfo *TJ = Target->getJITInfo())
    return new JIT(MP, *Target, *TJ);
  return 0;
}

//===-- TargetSelect.cpp - Target Chooser Code ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This just asks the TargetRegistry for the appropriate JIT to use, and allows
// the user to specify a specific one on the commandline with -march=x. Clients
// should initialize targets prior to calling createJIT.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/Module.h"
#include "llvm/ADT/Triple.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Host.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

/// selectTarget - Pick a target either via -march or by guessing the native
/// arch.  Add any CPU features specified via -mcpu or -mattr.
TargetMachine *EngineBuilder::selectTarget(Module *Mod,
                              StringRef MArch,
                              StringRef MCPU,
                              const SmallVectorImpl<std::string>& MAttrs,
                              Reloc::Model RM,
                              std::string *ErrorStr) {
  Triple TheTriple(Mod->getTargetTriple());
  if (TheTriple.getTriple().empty())
    TheTriple.setTriple(sys::getHostTriple());

  // Adjust the triple to match what the user requested.
  const Target *TheTarget = 0;
  if (!MArch.empty()) {
    for (TargetRegistry::iterator it = TargetRegistry::begin(),
           ie = TargetRegistry::end(); it != ie; ++it) {
      if (MArch == it->getName()) {
        TheTarget = &*it;
        break;
      }
    }

    if (!TheTarget) {
      *ErrorStr = "No available targets are compatible with this -march, "
        "see -version for the available targets.\n";
      return 0;
    }

    // Adjust the triple to match (if known), otherwise stick with the
    // module/host triple.
    Triple::ArchType Type = Triple::getArchTypeForLLVMName(MArch);
    if (Type != Triple::UnknownArch)
      TheTriple.setArch(Type);
  } else {
    std::string Error;
    TheTarget = TargetRegistry::lookupTarget(TheTriple.getTriple(), Error);
    if (TheTarget == 0) {
      if (ErrorStr)
        *ErrorStr = Error;
      return 0;
    }
  }

  if (!TheTarget->hasJIT()) {
    errs() << "WARNING: This target JIT is not designed for the host you are"
           << " running.  If bad things happen, please choose a different "
           << "-march switch.\n";
  }

  // Package up features to be passed to target/subtarget
  std::string FeaturesStr;
  if (!MAttrs.empty()) {
    SubtargetFeatures Features;
    for (unsigned i = 0; i != MAttrs.size(); ++i)
      Features.AddFeature(MAttrs[i]);
    FeaturesStr = Features.getString();
  }

  // Allocate a target...
  TargetMachine *Target = TheTarget->createTargetMachine(TheTriple.getTriple(),
                                                         MCPU, FeaturesStr, RM);
  assert(Target && "Could not allocate target machine!");
  return Target;
}

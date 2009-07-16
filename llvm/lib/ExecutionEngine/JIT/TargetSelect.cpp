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

#include "JIT.h"
#include "llvm/Module.h"
#include "llvm/ModuleProvider.h"
#include "llvm/Support/RegistryParser.h"
#include "llvm/Support/Streams.h"
#include "llvm/Target/SubtargetFeature.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

static cl::opt<std::string>
MArch("march", cl::desc("Architecture to generate assembly for (see --version)"));

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

/// createInternal - Create an return a new JIT compiler if there is one
/// available for the current target.  Otherwise, return null.
///
ExecutionEngine *JIT::createJIT(ModuleProvider *MP, std::string *ErrorStr,
                                JITMemoryManager *JMM,
                                CodeGenOpt::Level OptLevel,
                                bool AllocateGVsWithCode) {
  const Target *TheTarget = 0;
  if (MArch.empty()) {
    std::string Error;
    TheTarget = TargetRegistry::getClosestTargetForJIT(Error);
    if (TheTarget == 0) {
      if (ErrorStr)
        *ErrorStr = Error;
      return 0;
    }
  } else {
    for (TargetRegistry::iterator it = TargetRegistry::begin(),
           ie = TargetRegistry::end(); it != ie; ++it) {
      if (MArch == it->getName()) {
        TheTarget = &*it;
        break;
      }
    }
    
    if (TheTarget == 0) {
      if (ErrorStr)
        *ErrorStr = std::string("invalid target '" + MArch + "'.\n");
      return 0;
    }        

    if (TheTarget->getJITMatchQuality() == 0) {
      cerr << "WARNING: This target JIT is not designed for the host you are"
           << " running.  If bad things happen, please choose a different "
           << "-march switch.\n";
    }
  }

  // Package up features to be passed to target/subtarget
  std::string FeaturesStr;
  if (!MCPU.empty() || !MAttrs.empty()) {
    SubtargetFeatures Features;
    Features.setCPU(MCPU);
    for (unsigned i = 0; i != MAttrs.size(); ++i)
      Features.AddFeature(MAttrs[i]);
    FeaturesStr = Features.getString();
  }

  // Allocate a target...
  TargetMachine *Target = 
    TheTarget->createTargetMachine(*MP->getModule(), FeaturesStr);
  assert(Target && "Could not allocate target machine!");

  // If the target supports JIT code generation, return a new JIT now.
  if (TargetJITInfo *TJ = Target->getJITInfo())
    return new JIT(MP, *Target, *TJ, JMM, OptLevel, AllocateGVsWithCode);

  if (ErrorStr)
    *ErrorStr = "target does not support JIT code generation";
  return 0;
}

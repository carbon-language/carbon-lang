//===--- Options.cpp - clang-cc Option Handling ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// This file contains "pure" option handling, it is only responsible for turning
// the options into internal *Option classes, but shouldn't have any other
// logic.

#include "Options.h"
#include "clang/Frontend/CompileOptions.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CommandLine.h"
#include <stdio.h>

using namespace clang;

//===----------------------------------------------------------------------===//
// Code Generation Options
//===----------------------------------------------------------------------===//

namespace codegenoptions {

static llvm::cl::opt<bool>
DisableLLVMOptimizations("disable-llvm-optzns",
                         llvm::cl::desc("Don't run LLVM optimization passes"));

static llvm::cl::opt<bool>
DisableRedZone("disable-red-zone",
               llvm::cl::desc("Do not emit code that uses the red zone."),
               llvm::cl::init(false));

static llvm::cl::opt<bool>
GenerateDebugInfo("g",
                  llvm::cl::desc("Generate source level debug information"));

static llvm::cl::opt<bool>
NoCommon("fno-common",
         llvm::cl::desc("Compile common globals like normal definitions"),
         llvm::cl::ValueDisallowed);

static llvm::cl::opt<bool>
NoImplicitFloat("no-implicit-float",
  llvm::cl::desc("Don't generate implicit floating point instructions (x86-only)"),
  llvm::cl::init(false));

static llvm::cl::opt<bool>
NoMergeConstants("fno-merge-all-constants",
                       llvm::cl::desc("Disallow merging of constants."));

// It might be nice to add bounds to the CommandLine library directly.
struct OptLevelParser : public llvm::cl::parser<unsigned> {
  bool parse(llvm::cl::Option &O, llvm::StringRef ArgName,
             llvm::StringRef Arg, unsigned &Val) {
    if (llvm::cl::parser<unsigned>::parse(O, ArgName, Arg, Val))
      return true;
    if (Val > 3)
      return O.error("'" + Arg + "' invalid optimization level!");
    return false;
  }
};
static llvm::cl::opt<unsigned, false, OptLevelParser>
OptLevel("O", llvm::cl::Prefix,
         llvm::cl::desc("Optimization level"),
         llvm::cl::init(0));

static llvm::cl::opt<bool>
OptSize("Os", llvm::cl::desc("Optimize for size"));

static llvm::cl::opt<std::string>
TargetCPU("mcpu",
         llvm::cl::desc("Target a specific cpu type (-mcpu=help for details)"));

static llvm::cl::list<std::string>
TargetFeatures("target-feature", llvm::cl::desc("Target specific attributes"));

}

//===----------------------------------------------------------------------===//
// Option Object Construction
//===----------------------------------------------------------------------===//

/// ComputeTargetFeatures - Recompute the target feature list to only
/// be the list of things that are enabled, based on the target cpu
/// and feature list.
void clang::ComputeFeatureMap(TargetInfo &Target,
                              llvm::StringMap<bool> &Features) {
  using namespace codegenoptions;
  assert(Features.empty() && "invalid map");

  // Initialize the feature map based on the target.
  Target.getDefaultFeatures(TargetCPU, Features);

  // Apply the user specified deltas.
  for (llvm::cl::list<std::string>::iterator it = TargetFeatures.begin(),
         ie = TargetFeatures.end(); it != ie; ++it) {
    const char *Name = it->c_str();

    // FIXME: Don't handle errors like this.
    if (Name[0] != '-' && Name[0] != '+') {
      fprintf(stderr, "error: clang-cc: invalid target feature string: %s\n",
              Name);
      exit(1);
    }
    if (!Target.setFeatureEnabled(Features, Name + 1, (Name[0] == '+'))) {
      fprintf(stderr, "error: clang-cc: invalid target feature name: %s\n",
              Name + 1);
      exit(1);
    }
  }
}

void clang::InitializeCompileOptions(CompileOptions &Opts,
                                     const llvm::StringMap<bool> &Features) {
  using namespace codegenoptions;
  Opts.OptimizeSize = OptSize;
  Opts.DebugInfo = GenerateDebugInfo;
  Opts.DisableLLVMOpts = DisableLLVMOptimizations;

  // -Os implies -O2
  Opts.OptimizationLevel = OptSize ? 2 : OptLevel;

  // We must always run at least the always inlining pass.
  Opts.Inlining = (Opts.OptimizationLevel > 1) ? CompileOptions::NormalInlining
    : CompileOptions::OnlyAlwaysInlining;

  Opts.UnrollLoops = (Opts.OptimizationLevel > 1 && !OptSize);
  Opts.SimplifyLibCalls = 1;

#ifdef NDEBUG
  Opts.VerifyModule = 0;
#endif

  Opts.CPU = TargetCPU;
  Opts.Features.clear();
  for (llvm::StringMap<bool>::const_iterator it = Features.begin(),
         ie = Features.end(); it != ie; ++it) {
    // FIXME: If we are completely confident that we have the right set, we only
    // need to pass the minuses.
    std::string Name(it->second ? "+" : "-");
    Name += it->first();
    Opts.Features.push_back(Name);
  }

  Opts.NoCommon = NoCommon;

  Opts.DisableRedZone = DisableRedZone;
  Opts.NoImplicitFloat = NoImplicitFloat;

  Opts.MergeAllConstants = !NoMergeConstants;
}

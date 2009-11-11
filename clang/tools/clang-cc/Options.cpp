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
#include "clang/Frontend/PCHReader.h"
#include "clang/Frontend/PreprocessorOptions.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/STLExtras.h"
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
// General Preprocessor Options
//===----------------------------------------------------------------------===//

namespace preprocessoroptions {

static llvm::cl::list<std::string>
D_macros("D", llvm::cl::value_desc("macro"), llvm::cl::Prefix,
       llvm::cl::desc("Predefine the specified macro"));

static llvm::cl::list<std::string>
ImplicitIncludes("include", llvm::cl::value_desc("file"),
                 llvm::cl::desc("Include file before parsing"));
static llvm::cl::list<std::string>
ImplicitMacroIncludes("imacros", llvm::cl::value_desc("file"),
                      llvm::cl::desc("Include macros from file before parsing"));

static llvm::cl::opt<std::string>
ImplicitIncludePCH("include-pch", llvm::cl::value_desc("file"),
                   llvm::cl::desc("Include precompiled header file"));

static llvm::cl::opt<std::string>
ImplicitIncludePTH("include-pth", llvm::cl::value_desc("file"),
                   llvm::cl::desc("Include file before parsing"));

static llvm::cl::list<std::string>
U_macros("U", llvm::cl::value_desc("macro"), llvm::cl::Prefix,
         llvm::cl::desc("Undefine the specified macro"));

static llvm::cl::opt<bool>
UndefMacros("undef", llvm::cl::value_desc("macro"),
            llvm::cl::desc("undef all system defines"));

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

void clang::InitializePreprocessorOptions(PreprocessorOptions &Opts) {
  using namespace preprocessoroptions;

  Opts.setImplicitPCHInclude(ImplicitIncludePCH);
  Opts.setImplicitPTHInclude(ImplicitIncludePTH);

  // Use predefines?
  Opts.setUsePredefines(!UndefMacros);

  // Add macros from the command line.
  unsigned d = 0, D = D_macros.size();
  unsigned u = 0, U = U_macros.size();
  while (d < D || u < U) {
    if (u == U || (d < D && D_macros.getPosition(d) < U_macros.getPosition(u)))
      Opts.addMacroDef(D_macros[d++]);
    else
      Opts.addMacroUndef(U_macros[u++]);
  }

  // If -imacros are specified, include them now.  These are processed before
  // any -include directives.
  for (unsigned i = 0, e = ImplicitMacroIncludes.size(); i != e; ++i)
    Opts.addMacroInclude(ImplicitMacroIncludes[i]);

  // Add the ordered list of -includes, sorting in the implicit include options
  // at the appropriate location.
  llvm::SmallVector<std::pair<unsigned, std::string*>, 8> OrderedPaths;
  std::string OriginalFile;

  if (!ImplicitIncludePTH.empty())
    OrderedPaths.push_back(std::make_pair(ImplicitIncludePTH.getPosition(),
                                          &ImplicitIncludePTH));
  if (!ImplicitIncludePCH.empty()) {
    OriginalFile = PCHReader::getOriginalSourceFile(ImplicitIncludePCH);
    // FIXME: Don't fail like this.
    if (OriginalFile.empty())
      exit(1);
    OrderedPaths.push_back(std::make_pair(ImplicitIncludePCH.getPosition(),
                                          &OriginalFile));
  }
  for (unsigned i = 0, e = ImplicitIncludes.size(); i != e; ++i)
    OrderedPaths.push_back(std::make_pair(ImplicitIncludes.getPosition(i),
                                          &ImplicitIncludes[i]));
  llvm::array_pod_sort(OrderedPaths.begin(), OrderedPaths.end());

  for (unsigned i = 0, e = OrderedPaths.size(); i != e; ++i)
    Opts.addInclude(*OrderedPaths[i].second);
}

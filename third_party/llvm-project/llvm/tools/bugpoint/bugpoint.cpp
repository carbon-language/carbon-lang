//===- bugpoint.cpp - The LLVM Bugpoint utility ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This program is an automated compiler debugger tool.  It is used to narrow
// down miscompilations and crash problems to a specific pass in the compiler,
// and the specific Module or Function input that is causing the problem.
//
//===----------------------------------------------------------------------===//

#include "BugDriver.h"
#include "ToolRunner.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/LegacyPassNameParser.h"
#include "llvm/InitializePasses.h"
#include "llvm/LinkAllIR.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Valgrind.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

// Enable this macro to debug bugpoint itself.
//#define DEBUG_BUGPOINT 1

using namespace llvm;

static cl::opt<bool>
    FindBugs("find-bugs", cl::desc("Run many different optimization sequences "
                                   "on program to find bugs"),
             cl::init(false));

static cl::list<std::string>
    InputFilenames(cl::Positional, cl::OneOrMore,
                   cl::desc("<input llvm ll/bc files>"));

static cl::opt<unsigned> TimeoutValue(
    "timeout", cl::init(300), cl::value_desc("seconds"),
    cl::desc("Number of seconds program is allowed to run before it "
             "is killed (default is 300s), 0 disables timeout"));

static cl::opt<int> MemoryLimit(
    "mlimit", cl::init(-1), cl::value_desc("MBytes"),
    cl::desc("Maximum amount of memory to use. 0 disables check. Defaults to "
             "400MB (800MB under valgrind, 0 with sanitizers)."));

static cl::opt<bool>
    UseValgrind("enable-valgrind",
                cl::desc("Run optimizations through valgrind"));

// The AnalysesList is automatically populated with registered Passes by the
// PassNameParser.
//
static cl::list<const PassInfo *, bool, PassNameParser>
    PassList(cl::desc("Passes available:"), cl::ZeroOrMore);

static cl::opt<bool>
    StandardLinkOpts("std-link-opts",
                     cl::desc("Include the standard link time optimizations"));

static cl::opt<bool>
    OptLevelO1("O1", cl::desc("Optimization level 1. Identical to 'opt -O1'"));

static cl::opt<bool>
    OptLevelO2("O2", cl::desc("Optimization level 2. Identical to 'opt -O2'"));

static cl::opt<bool> OptLevelOs(
    "Os",
    cl::desc(
        "Like -O2 with extra optimizations for size. Similar to clang -Os"));

static cl::opt<bool>
OptLevelOz("Oz",
           cl::desc("Like -Os but reduces code size further. Similar to clang -Oz"));

static cl::opt<bool>
    OptLevelO3("O3", cl::desc("Optimization level 3. Identical to 'opt -O3'"));

static cl::opt<std::string>
    OverrideTriple("mtriple", cl::desc("Override target triple for module"));

/// BugpointIsInterrupted - Set to true when the user presses ctrl-c.
bool llvm::BugpointIsInterrupted = false;

#ifndef DEBUG_BUGPOINT
static void BugpointInterruptFunction() { BugpointIsInterrupted = true; }
#endif

// Hack to capture a pass list.
namespace {
class AddToDriver : public legacy::FunctionPassManager {
  BugDriver &D;

public:
  AddToDriver(BugDriver &_D) : FunctionPassManager(nullptr), D(_D) {}

  void add(Pass *P) override {
    const void *ID = P->getPassID();
    const PassInfo *PI = PassRegistry::getPassRegistry()->getPassInfo(ID);
    D.addPass(std::string(PI->getPassArgument()));
  }
};
}

// This routine adds optimization passes based on selected optimization level,
// OptLevel.
//
// OptLevel - Optimization Level
static void AddOptimizationPasses(legacy::FunctionPassManager &FPM,
                                  unsigned OptLevel,
                                  unsigned SizeLevel) {
  PassManagerBuilder Builder;
  Builder.OptLevel = OptLevel;
  Builder.SizeLevel = SizeLevel;

  if (OptLevel > 1)
    Builder.Inliner = createFunctionInliningPass(OptLevel, SizeLevel, false);
  else
    Builder.Inliner = createAlwaysInlinerLegacyPass();

  Builder.populateFunctionPassManager(FPM);
  Builder.populateModulePassManager(FPM);
}

#define HANDLE_EXTENSION(Ext)                                                  \
  llvm::PassPluginLibraryInfo get##Ext##PluginInfo();
#include "llvm/Support/Extension.def"

int main(int argc, char **argv) {
#ifndef DEBUG_BUGPOINT
  InitLLVM X(argc, argv);
#endif

  // Initialize passes
  PassRegistry &Registry = *PassRegistry::getPassRegistry();
  initializeCore(Registry);
  initializeScalarOpts(Registry);
  initializeObjCARCOpts(Registry);
  initializeVectorization(Registry);
  initializeIPO(Registry);
  initializeAnalysis(Registry);
  initializeTransformUtils(Registry);
  initializeInstCombine(Registry);
  initializeAggressiveInstCombine(Registry);
  initializeInstrumentation(Registry);
  initializeTarget(Registry);

  if (std::getenv("bar") == (char*) -1) {
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmPrinters();
    InitializeAllAsmParsers();
  }

  cl::ParseCommandLineOptions(argc, argv,
                              "LLVM automatic testcase reducer. See\nhttp://"
                              "llvm.org/cmds/bugpoint.html"
                              " for more information.\n");
#ifndef DEBUG_BUGPOINT
  sys::SetInterruptFunction(BugpointInterruptFunction);
#endif

  LLVMContext Context;
  // If we have an override, set it and then track the triple we want Modules
  // to use.
  if (!OverrideTriple.empty()) {
    TargetTriple.setTriple(Triple::normalize(OverrideTriple));
    outs() << "Override triple set to '" << TargetTriple.getTriple() << "'\n";
  }

  if (MemoryLimit < 0) {
    // Set the default MemoryLimit.  Be sure to update the flag's description if
    // you change this.
    if (sys::RunningOnValgrind() || UseValgrind)
      MemoryLimit = 800;
    else
      MemoryLimit = 400;
#if (LLVM_ADDRESS_SANITIZER_BUILD || LLVM_MEMORY_SANITIZER_BUILD ||            \
     LLVM_THREAD_SANITIZER_BUILD)
    // Starting from kernel 4.9 memory allocated with mmap is counted against
    // RLIMIT_DATA. Sanitizers need to allocate tens of terabytes for shadow.
    MemoryLimit = 0;
#endif
  }

  BugDriver D(argv[0], FindBugs, TimeoutValue, MemoryLimit, UseValgrind,
              Context);
  if (D.addSources(InputFilenames))
    return 1;

  AddToDriver PM(D);

  if (StandardLinkOpts) {
    PassManagerBuilder Builder;
    Builder.Inliner = createFunctionInliningPass();
    Builder.populateLTOPassManager(PM);
  }

  if (OptLevelO1)
    AddOptimizationPasses(PM, 1, 0);
  else if (OptLevelO2)
    AddOptimizationPasses(PM, 2, 0);
  else if (OptLevelO3)
    AddOptimizationPasses(PM, 3, 0);
  else if (OptLevelOs)
    AddOptimizationPasses(PM, 2, 1);
  else if (OptLevelOz)
    AddOptimizationPasses(PM, 2, 2);

  for (const PassInfo *PI : PassList)
    D.addPass(std::string(PI->getPassArgument()));

// Bugpoint has the ability of generating a plethora of core files, so to
// avoid filling up the disk, we prevent it
#ifndef DEBUG_BUGPOINT
  sys::Process::PreventCoreFiles();
#endif

// Needed to pull in symbols from statically linked extensions, including static
// registration. It is unused otherwise because bugpoint has no support for
// NewPM.
#define HANDLE_EXTENSION(Ext)                                                  \
  (void)get##Ext##PluginInfo();
#include "llvm/Support/Extension.def"

  if (Error E = D.run()) {
    errs() << toString(std::move(E));
    return 1;
  }
  return 0;
}

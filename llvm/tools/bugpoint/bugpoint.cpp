//===- bugpoint.cpp - The LLVM Bugpoint utility ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/LinkAllPasses.h"
#include "llvm/LLVMContext.h"
#include "llvm/Support/PassNameParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/StandardPasses.h"
#include "llvm/System/Process.h"
#include "llvm/System/Signals.h"
#include "llvm/System/Valgrind.h"
#include "llvm/LinkAllVMCore.h"
using namespace llvm;

static cl::opt<bool> 
FindBugs("find-bugs", cl::desc("Run many different optimization sequences "
                               "on program to find bugs"), cl::init(false));

static cl::list<std::string>
InputFilenames(cl::Positional, cl::OneOrMore,
               cl::desc("<input llvm ll/bc files>"));

static cl::opt<unsigned>
TimeoutValue("timeout", cl::init(300), cl::value_desc("seconds"),
             cl::desc("Number of seconds program is allowed to run before it "
                      "is killed (default is 300s), 0 disables timeout"));

static cl::opt<int>
MemoryLimit("mlimit", cl::init(-1), cl::value_desc("MBytes"),
             cl::desc("Maximum amount of memory to use. 0 disables check."
                      " Defaults to 100MB (800MB under valgrind)."));

static cl::opt<bool>
UseValgrind("enable-valgrind",
            cl::desc("Run optimizations through valgrind"));

// The AnalysesList is automatically populated with registered Passes by the
// PassNameParser.
//
static cl::list<const PassInfo*, bool, PassNameParser>
PassList(cl::desc("Passes available:"), cl::ZeroOrMore);

static cl::opt<bool>
StandardCompileOpts("std-compile-opts", 
                   cl::desc("Include the standard compile time optimizations"));

static cl::opt<bool>
StandardLinkOpts("std-link-opts", 
                 cl::desc("Include the standard link time optimizations"));

static cl::opt<std::string>
OverrideTriple("mtriple", cl::desc("Override target triple for module"));

/// BugpointIsInterrupted - Set to true when the user presses ctrl-c.
bool llvm::BugpointIsInterrupted = false;

static void BugpointInterruptFunction() {
  BugpointIsInterrupted = true;
}

// Hack to capture a pass list.
namespace {
  class AddToDriver : public PassManager {
    BugDriver &D;
  public:
    AddToDriver(BugDriver &_D) : D(_D) {}
    
    virtual void add(Pass *P) {
      const void *ID = P->getPassID();
      const PassInfo *PI = PassRegistry::getPassRegistry()->getPassInfo(ID);
      D.addPass(PI->getPassArgument());
    }
  };
}

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal();
  llvm::PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv,
                              "LLVM automatic testcase reducer. See\nhttp://"
                              "llvm.org/cmds/bugpoint.html"
                              " for more information.\n");
  sys::SetInterruptFunction(BugpointInterruptFunction);

  LLVMContext& Context = getGlobalContext();
  // If we have an override, set it and then track the triple we want Modules
  // to use.
  if (!OverrideTriple.empty()) {
    TargetTriple.setTriple(OverrideTriple);
    outs() << "Override triple set to '" << OverrideTriple << "'\n";
  }

  if (MemoryLimit < 0) {
    // Set the default MemoryLimit.  Be sure to update the flag's description if
    // you change this.
    if (sys::RunningOnValgrind() || UseValgrind)
      MemoryLimit = 800;
    else
      MemoryLimit = 100;
  }

  BugDriver D(argv[0], FindBugs, TimeoutValue, MemoryLimit,
              UseValgrind, Context);
  if (D.addSources(InputFilenames)) return 1;
  
  AddToDriver PM(D);
  if (StandardCompileOpts) {
    createStandardModulePasses(&PM, 3,
                               /*OptimizeSize=*/ false,
                               /*UnitAtATime=*/ true,
                               /*UnrollLoops=*/ true,
                               /*SimplifyLibCalls=*/ true,
                               /*HaveExceptions=*/ true,
                               createFunctionInliningPass());
  }
      
  if (StandardLinkOpts)
    createStandardLTOPasses(&PM, /*Internalize=*/true,
                            /*RunInliner=*/true,
                            /*VerifyEach=*/false);


  for (std::vector<const PassInfo*>::iterator I = PassList.begin(),
         E = PassList.end();
       I != E; ++I) {
    const PassInfo* PI = *I;
    D.addPass(PI->getPassArgument());
  }

  // Bugpoint has the ability of generating a plethora of core files, so to
  // avoid filling up the disk, we prevent it
  sys::Process::PreventCoreFiles();

  std::string Error;
  bool Failure = D.run(Error);
  if (!Error.empty()) {
    errs() << Error;
    return 1;
  }
  return Failure;
}

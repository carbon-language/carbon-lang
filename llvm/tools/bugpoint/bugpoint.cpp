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
#include "llvm/LinkAllVMCore.h"
using namespace llvm;

// AsChild - Specifies that this invocation of bugpoint is being generated
// from a parent process. It is not intended to be used by users so the 
// option is hidden.
static cl::opt<bool> 
AsChild("as-child", cl::desc("Run bugpoint as child process"), 
        cl::ReallyHidden);
          
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

static cl::opt<unsigned>
MemoryLimit("mlimit", cl::init(100), cl::value_desc("MBytes"),
             cl::desc("Maximum amount of memory to use. 0 disables check."));

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
      const PassInfo *PI = P->getPassInfo();
      D.addPasses(&PI, &PI + 1);
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
  BugDriver D(argv[0], AsChild, FindBugs, TimeoutValue, MemoryLimit, Context);
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

  D.addPasses(PassList.begin(), PassList.end());

  // Bugpoint has the ability of generating a plethora of core files, so to
  // avoid filling up the disk, we prevent it
  sys::Process::PreventCoreFiles();

  try {
    return D.run();
  } catch (ToolExecutionError &TEE) {
    errs() << "Tool execution error: " << TEE.what() << '\n';
  } catch (const std::string& msg) {
    errs() << argv[0] << ": " << msg << "\n";
  } catch (const std::bad_alloc&) {
    errs() << "Oh no, a bugpoint process ran out of memory!\n"
              "To increase the allocation limits for bugpoint child\n"
              "processes, use the -mlimit option.\n";
  } catch (const std::exception &e) {
    errs() << "Whoops, a std::exception leaked out of bugpoint: "
           << e.what() << "\n"
           << "This is a bug in bugpoint!\n";
  } catch (...) {
    errs() << "Whoops, an exception leaked out of bugpoint.  "
           << "This is a bug in bugpoint!\n";
  }
  return 1;
}

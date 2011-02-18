//===- opt.cpp - The LLVM Modular Optimizer -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Optimizations may be specified an arbitrary number of times on the command
// line, They are run in the order specified.
//
//===----------------------------------------------------------------------===//

#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/CallGraphSCCPass.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/RegionPass.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/PassNameParser.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/IRReader.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/StandardPasses.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/LinkAllVMCore.h"
#include <memory>
#include <algorithm>
using namespace llvm;

// The OptimizationList is automatically populated with registered Passes by the
// PassNameParser.
//
static cl::list<const PassInfo*, bool, PassNameParser>
PassList(cl::desc("Optimizations available:"));

// Other command line options...
//
static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bitcode file>"),
    cl::init("-"), cl::value_desc("filename"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Override output filename"),
               cl::value_desc("filename"));

static cl::opt<bool>
Force("f", cl::desc("Enable binary output on terminals"));

static cl::opt<bool>
PrintEachXForm("p", cl::desc("Print module after each transformation"));

static cl::opt<bool>
NoOutput("disable-output",
         cl::desc("Do not write result bitcode file"), cl::Hidden);

static cl::opt<bool>
OutputAssembly("S", cl::desc("Write output as LLVM assembly"));

static cl::opt<bool>
NoVerify("disable-verify", cl::desc("Do not verify result module"), cl::Hidden);

static cl::opt<bool>
VerifyEach("verify-each", cl::desc("Verify after each transform"));

static cl::opt<bool>
StripDebug("strip-debug",
           cl::desc("Strip debugger symbol info from translation unit"));

static cl::opt<bool>
DisableInline("disable-inlining", cl::desc("Do not run the inliner pass"));

static cl::opt<bool>
DisableOptimizations("disable-opt",
                     cl::desc("Do not run any optimization passes"));

static cl::opt<bool>
DisableInternalize("disable-internalize",
                   cl::desc("Do not mark all symbols as internal"));

static cl::opt<bool>
StandardCompileOpts("std-compile-opts",
                   cl::desc("Include the standard compile time optimizations"));

static cl::opt<bool>
StandardLinkOpts("std-link-opts",
                 cl::desc("Include the standard link time optimizations"));

static cl::opt<bool>
OptLevelO1("O1",
           cl::desc("Optimization level 1. Similar to llvm-gcc -O1"));

static cl::opt<bool>
OptLevelO2("O2",
           cl::desc("Optimization level 2. Similar to llvm-gcc -O2"));

static cl::opt<bool>
OptLevelO3("O3",
           cl::desc("Optimization level 3. Similar to llvm-gcc -O3"));

static cl::opt<bool>
UnitAtATime("funit-at-a-time",
            cl::desc("Enable IPO. This is same as llvm-gcc's -funit-at-a-time"),
            cl::init(true));

static cl::opt<bool>
DisableSimplifyLibCalls("disable-simplify-libcalls",
                        cl::desc("Disable simplify-libcalls"));

static cl::opt<bool>
Quiet("q", cl::desc("Obsolete option"), cl::Hidden);

static cl::alias
QuietA("quiet", cl::desc("Alias for -q"), cl::aliasopt(Quiet));

static cl::opt<bool>
AnalyzeOnly("analyze", cl::desc("Only perform analysis, no optimization"));

static cl::opt<bool>
PrintBreakpoints("print-breakpoints-for-testing", 
                 cl::desc("Print select breakpoints location for testing"));

static cl::opt<std::string>
DefaultDataLayout("default-data-layout", 
          cl::desc("data layout string to use if not specified by module"),
          cl::value_desc("layout-string"), cl::init(""));

// ---------- Define Printers for module and function passes ------------
namespace {

struct CallGraphSCCPassPrinter : public CallGraphSCCPass {
  static char ID;
  const PassInfo *PassToPrint;
  raw_ostream &Out;
  std::string PassName;

  CallGraphSCCPassPrinter(const PassInfo *PI, raw_ostream &out) :
    CallGraphSCCPass(ID), PassToPrint(PI), Out(out) {
      std::string PassToPrintName =  PassToPrint->getPassName();
      PassName = "CallGraphSCCPass Printer: " + PassToPrintName;
    }

  virtual bool runOnSCC(CallGraphSCC &SCC) {
    if (!Quiet)
      Out << "Printing analysis '" << PassToPrint->getPassName() << "':\n";

    // Get and print pass...
    for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); I != E; ++I) {
      Function *F = (*I)->getFunction();
      if (F)
        getAnalysisID<Pass>(PassToPrint->getTypeInfo()).print(Out,
                                                              F->getParent());
    }
    return false;
  }

  virtual const char *getPassName() const { return PassName.c_str(); }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequiredID(PassToPrint->getTypeInfo());
    AU.setPreservesAll();
  }
};

char CallGraphSCCPassPrinter::ID = 0;

struct ModulePassPrinter : public ModulePass {
  static char ID;
  const PassInfo *PassToPrint;
  raw_ostream &Out;
  std::string PassName;

  ModulePassPrinter(const PassInfo *PI, raw_ostream &out)
    : ModulePass(ID), PassToPrint(PI), Out(out) {
      std::string PassToPrintName =  PassToPrint->getPassName();
      PassName = "ModulePass Printer: " + PassToPrintName;
    }

  virtual bool runOnModule(Module &M) {
    if (!Quiet)
      Out << "Printing analysis '" << PassToPrint->getPassName() << "':\n";

    // Get and print pass...
    getAnalysisID<Pass>(PassToPrint->getTypeInfo()).print(Out, &M);
    return false;
  }

  virtual const char *getPassName() const { return PassName.c_str(); }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequiredID(PassToPrint->getTypeInfo());
    AU.setPreservesAll();
  }
};

char ModulePassPrinter::ID = 0;
struct FunctionPassPrinter : public FunctionPass {
  const PassInfo *PassToPrint;
  raw_ostream &Out;
  static char ID;
  std::string PassName;

  FunctionPassPrinter(const PassInfo *PI, raw_ostream &out)
    : FunctionPass(ID), PassToPrint(PI), Out(out) {
      std::string PassToPrintName =  PassToPrint->getPassName();
      PassName = "FunctionPass Printer: " + PassToPrintName;
    }

  virtual bool runOnFunction(Function &F) {
    if (!Quiet)
      Out << "Printing analysis '" << PassToPrint->getPassName()
          << "' for function '" << F.getName() << "':\n";

    // Get and print pass...
    getAnalysisID<Pass>(PassToPrint->getTypeInfo()).print(Out,
            F.getParent());
    return false;
  }

  virtual const char *getPassName() const { return PassName.c_str(); }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequiredID(PassToPrint->getTypeInfo());
    AU.setPreservesAll();
  }
};

char FunctionPassPrinter::ID = 0;

struct LoopPassPrinter : public LoopPass {
  static char ID;
  const PassInfo *PassToPrint;
  raw_ostream &Out;
  std::string PassName;

  LoopPassPrinter(const PassInfo *PI, raw_ostream &out) :
    LoopPass(ID), PassToPrint(PI), Out(out) {
      std::string PassToPrintName =  PassToPrint->getPassName();
      PassName = "LoopPass Printer: " + PassToPrintName;
    }


  virtual bool runOnLoop(Loop *L, LPPassManager &LPM) {
    if (!Quiet)
      Out << "Printing analysis '" << PassToPrint->getPassName() << "':\n";

    // Get and print pass...
    getAnalysisID<Pass>(PassToPrint->getTypeInfo()).print(Out,
                        L->getHeader()->getParent()->getParent());
    return false;
  }

  virtual const char *getPassName() const { return PassName.c_str(); }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequiredID(PassToPrint->getTypeInfo());
    AU.setPreservesAll();
  }
};

char LoopPassPrinter::ID = 0;

struct RegionPassPrinter : public RegionPass {
  static char ID;
  const PassInfo *PassToPrint;
  raw_ostream &Out;
  std::string PassName;

  RegionPassPrinter(const PassInfo *PI, raw_ostream &out) : RegionPass(ID),
    PassToPrint(PI), Out(out) {
    std::string PassToPrintName =  PassToPrint->getPassName();
    PassName = "RegionPass Printer: " + PassToPrintName;
  }

  virtual bool runOnRegion(Region *R, RGPassManager &RGM) {
    if (!Quiet) {
      Out << "Printing analysis '" << PassToPrint->getPassName() << "' for "
        << "region: '" << R->getNameStr() << "' in function '"
        << R->getEntry()->getParent()->getNameStr() << "':\n";
    }
    // Get and print pass...
   getAnalysisID<Pass>(PassToPrint->getTypeInfo()).print(Out,
                       R->getEntry()->getParent()->getParent());
    return false;
  }

  virtual const char *getPassName() const { return PassName.c_str(); }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequiredID(PassToPrint->getTypeInfo());
    AU.setPreservesAll();
  }
};

char RegionPassPrinter::ID = 0;

struct BasicBlockPassPrinter : public BasicBlockPass {
  const PassInfo *PassToPrint;
  raw_ostream &Out;
  static char ID;
  std::string PassName;

  BasicBlockPassPrinter(const PassInfo *PI, raw_ostream &out)
    : BasicBlockPass(ID), PassToPrint(PI), Out(out) {
      std::string PassToPrintName =  PassToPrint->getPassName();
      PassName = "BasicBlockPass Printer: " + PassToPrintName;
    }

  virtual bool runOnBasicBlock(BasicBlock &BB) {
    if (!Quiet)
      Out << "Printing Analysis info for BasicBlock '" << BB.getName()
          << "': Pass " << PassToPrint->getPassName() << ":\n";

    // Get and print pass...
    getAnalysisID<Pass>(PassToPrint->getTypeInfo()).print(Out, 
            BB.getParent()->getParent());
    return false;
  }

  virtual const char *getPassName() const { return PassName.c_str(); }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequiredID(PassToPrint->getTypeInfo());
    AU.setPreservesAll();
  }
};

char BasicBlockPassPrinter::ID = 0;

struct BreakpointPrinter : public FunctionPass {
  raw_ostream &Out;
  static char ID;

  BreakpointPrinter(raw_ostream &out)
    : FunctionPass(ID), Out(out) {
    }

  virtual bool runOnFunction(Function &F) {
    BasicBlock &EntryBB = F.getEntryBlock();
    BasicBlock::const_iterator BI = EntryBB.end();
    --BI;
    do {
      const Instruction *In = BI;
      const DebugLoc DL = In->getDebugLoc();
      if (!DL.isUnknown()) {
        DIScope S(DL.getScope(getGlobalContext()));
        Out << S.getFilename() << " " << DL.getLine() << "\n";
        break;
      }
      --BI;
    } while (BI != EntryBB.begin());
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
};

char BreakpointPrinter::ID = 0;

inline void addPass(PassManagerBase &PM, Pass *P) {
  // Add the pass to the pass manager...
  PM.add(P);

  // If we are verifying all of the intermediate steps, add the verifier...
  if (VerifyEach) PM.add(createVerifierPass());
}

/// AddOptimizationPasses - This routine adds optimization passes
/// based on selected optimization level, OptLevel. This routine
/// duplicates llvm-gcc behaviour.
///
/// OptLevel - Optimization Level
void AddOptimizationPasses(PassManagerBase &MPM, PassManagerBase &FPM,
                           unsigned OptLevel) {
  createStandardFunctionPasses(&FPM, OptLevel);

  llvm::Pass *InliningPass = 0;
  if (DisableInline) {
    // No inlining pass
  } else if (OptLevel) {
    unsigned Threshold = 225;
    if (OptLevel > 2)
      Threshold = 275;
    InliningPass = createFunctionInliningPass(Threshold);
  } else {
    InliningPass = createAlwaysInlinerPass();
  }
  createStandardModulePasses(&MPM, OptLevel,
                             /*OptimizeSize=*/ false,
                             UnitAtATime,
                             /*UnrollLoops=*/ OptLevel > 1,
                             !DisableSimplifyLibCalls,
                             /*HaveExceptions=*/ true,
                             InliningPass);
}

void AddStandardCompilePasses(PassManagerBase &PM) {
  PM.add(createVerifierPass());                  // Verify that input is correct

  addPass(PM, createLowerSetJmpPass());          // Lower llvm.setjmp/.longjmp

  // If the -strip-debug command line option was specified, do it.
  if (StripDebug)
    addPass(PM, createStripSymbolsPass(true));

  if (DisableOptimizations) return;

  llvm::Pass *InliningPass = !DisableInline ? createFunctionInliningPass() : 0;

  // -std-compile-opts adds the same module passes as -O3.
  createStandardModulePasses(&PM, 3,
                             /*OptimizeSize=*/ false,
                             /*UnitAtATime=*/ true,
                             /*UnrollLoops=*/ true,
                             !DisableSimplifyLibCalls,
                             /*HaveExceptions=*/ true,
                             InliningPass);
}

void AddStandardLinkPasses(PassManagerBase &PM) {
  PM.add(createVerifierPass());                  // Verify that input is correct

  // If the -strip-debug command line option was specified, do it.
  if (StripDebug)
    addPass(PM, createStripSymbolsPass(true));

  if (DisableOptimizations) return;

  createStandardLTOPasses(&PM, /*Internalize=*/ !DisableInternalize,
                          /*RunInliner=*/ !DisableInline,
                          /*VerifyEach=*/ VerifyEach);
}

} // anonymous namespace


//===----------------------------------------------------------------------===//
// main for opt
//
int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal();
  llvm::PrettyStackTraceProgram X(argc, argv);

  // Enable debug stream buffering.
  EnableDebugBuffering = true;

  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.
  LLVMContext &Context = getGlobalContext();
  
  // Initialize passes
  PassRegistry &Registry = *PassRegistry::getPassRegistry();
  initializeCore(Registry);
  initializeScalarOpts(Registry);
  initializeIPO(Registry);
  initializeAnalysis(Registry);
  initializeIPA(Registry);
  initializeTransformUtils(Registry);
  initializeInstCombine(Registry);
  initializeInstrumentation(Registry);
  initializeTarget(Registry);
  
  cl::ParseCommandLineOptions(argc, argv,
    "llvm .bc -> .bc modular optimizer and analysis printer\n");

  if (AnalyzeOnly && NoOutput) {
    errs() << argv[0] << ": analyze mode conflicts with no-output mode.\n";
    return 1;
  }

  // Allocate a full target machine description only if necessary.
  // FIXME: The choice of target should be controllable on the command line.
  std::auto_ptr<TargetMachine> target;

  SMDiagnostic Err;

  // Load the input module...
  std::auto_ptr<Module> M;
  M.reset(ParseIRFile(InputFilename, Err, Context));

  if (M.get() == 0) {
    Err.Print(argv[0], errs());
    return 1;
  }

  // Figure out what stream we are supposed to write to...
  OwningPtr<tool_output_file> Out;
  if (NoOutput) {
    if (!OutputFilename.empty())
      errs() << "WARNING: The -o (output filename) option is ignored when\n"
                "the --disable-output option is used.\n";
  } else {
    // Default to standard output.
    if (OutputFilename.empty())
      OutputFilename = "-";

    std::string ErrorInfo;
    Out.reset(new tool_output_file(OutputFilename.c_str(), ErrorInfo,
                                   raw_fd_ostream::F_Binary));
    if (!ErrorInfo.empty()) {
      errs() << ErrorInfo << '\n';
      return 1;
    }
  }

  // If the output is set to be emitted to standard out, and standard out is a
  // console, print out a warning message and refuse to do it.  We don't
  // impress anyone by spewing tons of binary goo to a terminal.
  if (!Force && !NoOutput && !AnalyzeOnly && !OutputAssembly)
    if (CheckBitcodeOutputToConsole(Out->os(), !Quiet))
      NoOutput = true;

  // Create a PassManager to hold and optimize the collection of passes we are
  // about to build.
  //
  PassManager Passes;

  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  TargetLibraryInfo *TLI = new TargetLibraryInfo(Triple(M->getTargetTriple()));
  
  // The -disable-simplify-libcalls flag actually disables all builtin optzns.
  if (DisableSimplifyLibCalls)
    TLI->disableAllFunctions();
  Passes.add(TLI);
  
  // Add an appropriate TargetData instance for this module.
  TargetData *TD = 0;
  const std::string &ModuleDataLayout = M.get()->getDataLayout();
  if (!ModuleDataLayout.empty())
    TD = new TargetData(ModuleDataLayout);
  else if (!DefaultDataLayout.empty())
    TD = new TargetData(DefaultDataLayout);

  if (TD)
    Passes.add(TD);

  OwningPtr<PassManager> FPasses;
  if (OptLevelO1 || OptLevelO2 || OptLevelO3) {
    FPasses.reset(new PassManager());
    if (TD)
      FPasses->add(new TargetData(*TD));
  }

  if (PrintBreakpoints) {
    // Default to standard output.
    if (!Out) {
      if (OutputFilename.empty())
        OutputFilename = "-";
      
      std::string ErrorInfo;
      Out.reset(new tool_output_file(OutputFilename.c_str(), ErrorInfo,
                                     raw_fd_ostream::F_Binary));
      if (!ErrorInfo.empty()) {
        errs() << ErrorInfo << '\n';
        return 1;
      }
    }
    Passes.add(new BreakpointPrinter(Out->os()));
    NoOutput = true;
  }

  // If the -strip-debug command line option was specified, add it.  If
  // -std-compile-opts was also specified, it will handle StripDebug.
  if (StripDebug && !StandardCompileOpts)
    addPass(Passes, createStripSymbolsPass(true));

  // Create a new optimization pass for each one specified on the command line
  for (unsigned i = 0; i < PassList.size(); ++i) {
    // Check to see if -std-compile-opts was specified before this option.  If
    // so, handle it.
    if (StandardCompileOpts &&
        StandardCompileOpts.getPosition() < PassList.getPosition(i)) {
      AddStandardCompilePasses(Passes);
      StandardCompileOpts = false;
    }

    if (StandardLinkOpts &&
        StandardLinkOpts.getPosition() < PassList.getPosition(i)) {
      AddStandardLinkPasses(Passes);
      StandardLinkOpts = false;
    }

    if (OptLevelO1 && OptLevelO1.getPosition() < PassList.getPosition(i)) {
      AddOptimizationPasses(Passes, *FPasses, 1);
      OptLevelO1 = false;
    }

    if (OptLevelO2 && OptLevelO2.getPosition() < PassList.getPosition(i)) {
      AddOptimizationPasses(Passes, *FPasses, 2);
      OptLevelO2 = false;
    }

    if (OptLevelO3 && OptLevelO3.getPosition() < PassList.getPosition(i)) {
      AddOptimizationPasses(Passes, *FPasses, 3);
      OptLevelO3 = false;
    }

    const PassInfo *PassInf = PassList[i];
    Pass *P = 0;
    if (PassInf->getNormalCtor())
      P = PassInf->getNormalCtor()();
    else
      errs() << argv[0] << ": cannot create pass: "
             << PassInf->getPassName() << "\n";
    if (P) {
      PassKind Kind = P->getPassKind();
      addPass(Passes, P);

      if (AnalyzeOnly) {
        switch (Kind) {
        case PT_BasicBlock:
          Passes.add(new BasicBlockPassPrinter(PassInf, Out->os()));
          break;
        case PT_Region:
          Passes.add(new RegionPassPrinter(PassInf, Out->os()));
          break;
        case PT_Loop:
          Passes.add(new LoopPassPrinter(PassInf, Out->os()));
          break;
        case PT_Function:
          Passes.add(new FunctionPassPrinter(PassInf, Out->os()));
          break;
        case PT_CallGraphSCC:
          Passes.add(new CallGraphSCCPassPrinter(PassInf, Out->os()));
          break;
        default:
          Passes.add(new ModulePassPrinter(PassInf, Out->os()));
          break;
        }
      }
    }

    if (PrintEachXForm)
      Passes.add(createPrintModulePass(&errs()));
  }

  // If -std-compile-opts was specified at the end of the pass list, add them.
  if (StandardCompileOpts) {
    AddStandardCompilePasses(Passes);
    StandardCompileOpts = false;
  }

  if (StandardLinkOpts) {
    AddStandardLinkPasses(Passes);
    StandardLinkOpts = false;
  }

  if (OptLevelO1)
    AddOptimizationPasses(Passes, *FPasses, 1);

  if (OptLevelO2)
    AddOptimizationPasses(Passes, *FPasses, 2);

  if (OptLevelO3)
    AddOptimizationPasses(Passes, *FPasses, 3);

  if (OptLevelO1 || OptLevelO2 || OptLevelO3)
    FPasses->run(*M.get());

  // Check that the module is well formed on completion of optimization
  if (!NoVerify && !VerifyEach)
    Passes.add(createVerifierPass());

  // Write bitcode or assembly to the output as the last step...
  if (!NoOutput && !AnalyzeOnly) {
    if (OutputAssembly)
      Passes.add(createPrintModulePass(&Out->os()));
    else
      Passes.add(createBitcodeWriterPass(Out->os()));
  }

  // Now that we have all of the passes ready, run them.
  Passes.run(*M.get());

  // Declare success.
  if (!NoOutput || PrintBreakpoints)
    Out->keep();

  return 0;
}

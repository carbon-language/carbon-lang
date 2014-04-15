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

#include "BreakpointPrinter.h"
#include "NewPMDriver.h"
#include "PassPrinters.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/RegionPass.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassNameParser.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/InitializePasses.h"
#include "llvm/LinkAllIR.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/PassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include <algorithm>
#include <memory>
using namespace llvm;
using namespace opt_tool;

// The OptimizationList is automatically populated with registered Passes by the
// PassNameParser.
//
static cl::list<const PassInfo*, bool, PassNameParser>
PassList(cl::desc("Optimizations available:"));

// This flag specifies a textual description of the optimization pass pipeline
// to run over the module. This flag switches opt to use the new pass manager
// infrastructure, completely disabling all of the flags specific to the old
// pass management.
static cl::opt<std::string> PassPipeline(
    "passes",
    cl::desc("A textual description of the pass pipeline for optimizing"),
    cl::Hidden);

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
           cl::desc("Optimization level 1. Similar to clang -O1"));

static cl::opt<bool>
OptLevelO2("O2",
           cl::desc("Optimization level 2. Similar to clang -O2"));

static cl::opt<bool>
OptLevelOs("Os",
           cl::desc("Like -O2 with extra optimizations for size. Similar to clang -Os"));

static cl::opt<bool>
OptLevelOz("Oz",
           cl::desc("Like -Os but reduces code size further. Similar to clang -Oz"));

static cl::opt<bool>
OptLevelO3("O3",
           cl::desc("Optimization level 3. Similar to clang -O3"));

static cl::opt<std::string>
TargetTriple("mtriple", cl::desc("Override target triple for module"));

static cl::opt<bool>
UnitAtATime("funit-at-a-time",
            cl::desc("Enable IPO. This is same as llvm-gcc's -funit-at-a-time"),
            cl::init(true));

static cl::opt<bool>
DisableLoopUnrolling("disable-loop-unrolling",
                     cl::desc("Disable loop unrolling in all relevant passes"),
                     cl::init(false));
static cl::opt<bool>
DisableLoopVectorization("disable-loop-vectorization",
                     cl::desc("Disable the loop vectorization pass"),
                     cl::init(false));

static cl::opt<bool>
DisableSLPVectorization("disable-slp-vectorization",
                        cl::desc("Disable the slp vectorization pass"),
                        cl::init(false));


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



static inline void addPass(PassManagerBase &PM, Pass *P) {
  // Add the pass to the pass manager...
  PM.add(P);

  // If we are verifying all of the intermediate steps, add the verifier...
  if (VerifyEach) {
    PM.add(createVerifierPass());
    PM.add(createDebugInfoVerifierPass());
  }
}

/// AddOptimizationPasses - This routine adds optimization passes
/// based on selected optimization level, OptLevel. This routine
/// duplicates llvm-gcc behaviour.
///
/// OptLevel - Optimization Level
static void AddOptimizationPasses(PassManagerBase &MPM,FunctionPassManager &FPM,
                                  unsigned OptLevel, unsigned SizeLevel) {
  FPM.add(createVerifierPass());          // Verify that input is correct
  MPM.add(createDebugInfoVerifierPass()); // Verify that debug info is correct

  PassManagerBuilder Builder;
  Builder.OptLevel = OptLevel;
  Builder.SizeLevel = SizeLevel;

  if (DisableInline) {
    // No inlining pass
  } else if (OptLevel > 1) {
    Builder.Inliner = createFunctionInliningPass(OptLevel, SizeLevel);
  } else {
    Builder.Inliner = createAlwaysInlinerPass();
  }
  Builder.DisableUnitAtATime = !UnitAtATime;
  Builder.DisableUnrollLoops = (DisableLoopUnrolling.getNumOccurrences() > 0) ?
                               DisableLoopUnrolling : OptLevel == 0;

  // This is final, unless there is a #pragma vectorize enable
  if (DisableLoopVectorization)
    Builder.LoopVectorize = false;
  // If option wasn't forced via cmd line (-vectorize-loops, -loop-vectorize)
  else if (!Builder.LoopVectorize)
    Builder.LoopVectorize = OptLevel > 1 && SizeLevel < 2;

  // When #pragma vectorize is on for SLP, do the same as above
  Builder.SLPVectorize =
      DisableSLPVectorization ? false : OptLevel > 1 && SizeLevel < 2;

  Builder.populateFunctionPassManager(FPM);
  Builder.populateModulePassManager(MPM);
}

static void AddStandardCompilePasses(PassManagerBase &PM) {
  PM.add(createVerifierPass());                  // Verify that input is correct

  // If the -strip-debug command line option was specified, do it.
  if (StripDebug)
    addPass(PM, createStripSymbolsPass(true));

  // Verify debug info only after it's (possibly) stripped.
  PM.add(createDebugInfoVerifierPass());

  if (DisableOptimizations) return;

  // -std-compile-opts adds the same module passes as -O3.
  PassManagerBuilder Builder;
  if (!DisableInline)
    Builder.Inliner = createFunctionInliningPass();
  Builder.OptLevel = 3;
  Builder.populateModulePassManager(PM);
}

static void AddStandardLinkPasses(PassManagerBase &PM) {
  PM.add(createVerifierPass());                  // Verify that input is correct

  // If the -strip-debug command line option was specified, do it.
  if (StripDebug)
    addPass(PM, createStripSymbolsPass(true));

  // Verify debug info only after it's (possibly) stripped.
  PM.add(createDebugInfoVerifierPass());

  if (DisableOptimizations) return;

  PassManagerBuilder Builder;
  Builder.populateLTOPassManager(PM, /*Internalize=*/ !DisableInternalize,
                                 /*RunInliner=*/ !DisableInline);
}

//===----------------------------------------------------------------------===//
// CodeGen-related helper functions.
//

CodeGenOpt::Level GetCodeGenOptLevel() {
  if (OptLevelO1)
    return CodeGenOpt::Less;
  if (OptLevelO2)
    return CodeGenOpt::Default;
  if (OptLevelO3)
    return CodeGenOpt::Aggressive;
  return CodeGenOpt::None;
}

// Returns the TargetMachine instance or zero if no triple is provided.
static TargetMachine* GetTargetMachine(Triple TheTriple) {
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(MArch, TheTriple,
                                                         Error);
  // Some modules don't specify a triple, and this is okay.
  if (!TheTarget) {
    return 0;
  }

  // Package up features to be passed to target/subtarget
  std::string FeaturesStr;
  if (MAttrs.size()) {
    SubtargetFeatures Features;
    for (unsigned i = 0; i != MAttrs.size(); ++i)
      Features.AddFeature(MAttrs[i]);
    FeaturesStr = Features.getString();
  }

  return TheTarget->createTargetMachine(TheTriple.getTriple(),
                                        MCPU, FeaturesStr,
                                        InitTargetOptionsFromCodeGenFlags(),
                                        RelocModel, CMModel,
                                        GetCodeGenOptLevel());
}

#ifdef LINK_POLLY_INTO_TOOLS
namespace polly {
void initializePollyPasses(llvm::PassRegistry &Registry);
}
#endif

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

  InitializeAllTargets();
  InitializeAllTargetMCs();

  // Initialize passes
  PassRegistry &Registry = *PassRegistry::getPassRegistry();
  initializeCore(Registry);
  initializeDebugIRPass(Registry);
  initializeScalarOpts(Registry);
  initializeObjCARCOpts(Registry);
  initializeVectorization(Registry);
  initializeIPO(Registry);
  initializeAnalysis(Registry);
  initializeIPA(Registry);
  initializeTransformUtils(Registry);
  initializeInstCombine(Registry);
  initializeInstrumentation(Registry);
  initializeTarget(Registry);
  // For codegen passes, only passes that do IR to IR transformation are
  // supported. For now, just add CodeGenPrepare.
  initializeCodeGenPreparePass(Registry);

#ifdef LINK_POLLY_INTO_TOOLS
  polly::initializePollyPasses(Registry);
#endif

  cl::ParseCommandLineOptions(argc, argv,
    "llvm .bc -> .bc modular optimizer and analysis printer\n");

  if (AnalyzeOnly && NoOutput) {
    errs() << argv[0] << ": analyze mode conflicts with no-output mode.\n";
    return 1;
  }

  SMDiagnostic Err;

  // Load the input module...
  std::unique_ptr<Module> M;
  M.reset(ParseIRFile(InputFilename, Err, Context));

  if (M.get() == 0) {
    Err.print(argv[0], errs());
    return 1;
  }

  // If we are supposed to override the target triple, do so now.
  if (!TargetTriple.empty())
    M->setTargetTriple(Triple::normalize(TargetTriple));

  // Figure out what stream we are supposed to write to...
  std::unique_ptr<tool_output_file> Out;
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
                                   sys::fs::F_None));
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

  if (PassPipeline.getNumOccurrences() > 0) {
    OutputKind OK = OK_NoOutput;
    if (!NoOutput)
      OK = OutputAssembly ? OK_OutputAssembly : OK_OutputBitcode;

    VerifierKind VK = VK_VerifyInAndOut;
    if (NoVerify)
      VK = VK_NoVerifier;
    else if (VerifyEach)
      VK = VK_VerifyEachPass;

    // The user has asked to use the new pass manager and provided a pipeline
    // string. Hand off the rest of the functionality to the new code for that
    // layer.
    return runPassPipeline(argv[0], Context, *M.get(), Out.get(), PassPipeline,
                           OK, VK)
               ? 0
               : 1;
  }

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

  // Add an appropriate DataLayout instance for this module.
  const DataLayout *DL = M.get()->getDataLayout();
  if (!DL && !DefaultDataLayout.empty()) {
    M->setDataLayout(DefaultDataLayout);
    DL = M.get()->getDataLayout();
  }

  if (DL)
    Passes.add(new DataLayoutPass(M.get()));

  Triple ModuleTriple(M->getTargetTriple());
  TargetMachine *Machine = 0;
  if (ModuleTriple.getArch())
    Machine = GetTargetMachine(Triple(ModuleTriple));
  std::unique_ptr<TargetMachine> TM(Machine);

  // Add internal analysis passes from the target machine.
  if (TM.get())
    TM->addAnalysisPasses(Passes);

  std::unique_ptr<FunctionPassManager> FPasses;
  if (OptLevelO1 || OptLevelO2 || OptLevelOs || OptLevelOz || OptLevelO3) {
    FPasses.reset(new FunctionPassManager(M.get()));
    if (DL)
      FPasses->add(new DataLayoutPass(M.get()));
    if (TM.get())
      TM->addAnalysisPasses(*FPasses);

  }

  if (PrintBreakpoints) {
    // Default to standard output.
    if (!Out) {
      if (OutputFilename.empty())
        OutputFilename = "-";

      std::string ErrorInfo;
      Out.reset(new tool_output_file(OutputFilename.c_str(), ErrorInfo,
                                     sys::fs::F_None));
      if (!ErrorInfo.empty()) {
        errs() << ErrorInfo << '\n';
        return 1;
      }
    }
    Passes.add(createBreakpointPrinter(Out->os()));
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
      AddOptimizationPasses(Passes, *FPasses, 1, 0);
      OptLevelO1 = false;
    }

    if (OptLevelO2 && OptLevelO2.getPosition() < PassList.getPosition(i)) {
      AddOptimizationPasses(Passes, *FPasses, 2, 0);
      OptLevelO2 = false;
    }

    if (OptLevelOs && OptLevelOs.getPosition() < PassList.getPosition(i)) {
      AddOptimizationPasses(Passes, *FPasses, 2, 1);
      OptLevelOs = false;
    }

    if (OptLevelOz && OptLevelOz.getPosition() < PassList.getPosition(i)) {
      AddOptimizationPasses(Passes, *FPasses, 2, 2);
      OptLevelOz = false;
    }

    if (OptLevelO3 && OptLevelO3.getPosition() < PassList.getPosition(i)) {
      AddOptimizationPasses(Passes, *FPasses, 3, 0);
      OptLevelO3 = false;
    }

    const PassInfo *PassInf = PassList[i];
    Pass *P = 0;
    if (PassInf->getTargetMachineCtor())
      P = PassInf->getTargetMachineCtor()(TM.get());
    else if (PassInf->getNormalCtor())
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
          Passes.add(createBasicBlockPassPrinter(PassInf, Out->os(), Quiet));
          break;
        case PT_Region:
          Passes.add(createRegionPassPrinter(PassInf, Out->os(), Quiet));
          break;
        case PT_Loop:
          Passes.add(createLoopPassPrinter(PassInf, Out->os(), Quiet));
          break;
        case PT_Function:
          Passes.add(createFunctionPassPrinter(PassInf, Out->os(), Quiet));
          break;
        case PT_CallGraphSCC:
          Passes.add(createCallGraphPassPrinter(PassInf, Out->os(), Quiet));
          break;
        default:
          Passes.add(createModulePassPrinter(PassInf, Out->os(), Quiet));
          break;
        }
      }
    }

    if (PrintEachXForm)
      Passes.add(createPrintModulePass(errs()));
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
    AddOptimizationPasses(Passes, *FPasses, 1, 0);

  if (OptLevelO2)
    AddOptimizationPasses(Passes, *FPasses, 2, 0);

  if (OptLevelOs)
    AddOptimizationPasses(Passes, *FPasses, 2, 1);

  if (OptLevelOz)
    AddOptimizationPasses(Passes, *FPasses, 2, 2);

  if (OptLevelO3)
    AddOptimizationPasses(Passes, *FPasses, 3, 0);

  if (OptLevelO1 || OptLevelO2 || OptLevelOs || OptLevelOz || OptLevelO3) {
    FPasses->doInitialization();
    for (Module::iterator F = M->begin(), E = M->end(); F != E; ++F)
      FPasses->run(*F);
    FPasses->doFinalization();
  }

  // Check that the module is well formed on completion of optimization
  if (!NoVerify && !VerifyEach) {
    Passes.add(createVerifierPass());
    Passes.add(createDebugInfoVerifierPass());
  }

  // Write bitcode or assembly to the output as the last step...
  if (!NoOutput && !AnalyzeOnly) {
    if (OutputAssembly)
      Passes.add(createPrintModulePass(Out->os()));
    else
      Passes.add(createBitcodeWriterPass(Out->os()));
  }

  // Before executing passes, print the final values of the LLVM options.
  cl::PrintOptionValues();

  // Now that we have all of the passes ready, run them.
  Passes.run(*M.get());

  // Declare success.
  if (!NoOutput || PrintBreakpoints)
    Out->keep();

  return 0;
}

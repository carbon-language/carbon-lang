//===----------------------------------------------------------------------===//
// LLVM 'OPT' UTILITY 
//
// Optimizations may be specified an arbitrary number of times on the command
// line, they are run in the order specified.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Bytecode/WriteBytecodePass.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Transforms/UnifyMethodExitNodes.h"
#include "llvm/Transforms/ConstantMerge.h"
#include "llvm/Transforms/CleanupGCCOutput.h"
#include "llvm/Transforms/LevelChange.h"
#include "llvm/Transforms/MethodInlining.h"
#include "llvm/Transforms/SymbolStripping.h"
#include "llvm/Transforms/ChangeAllocations.h"
#include "llvm/Transforms/IPO/SimpleStructMutation.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/ConstantProp.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/InstructionCombining.h"
#include "llvm/Transforms/Scalar/PromoteMemoryToRegister.h"
#include "llvm/Transforms/Instrumentation/TraceValues.h"
#include "llvm/Transforms/Instrumentation/ProfilePaths.h"
#include "Support/CommandLine.h"
#include <fstream>
#include <memory>

// Opts enum - All of the transformations we can do...
enum Opts {
  // Basic optimizations
  dce, constprop, inlining, constmerge, strip, mstrip, mergereturn,

  // Miscellaneous Transformations
  raiseallocs, cleangcc,

  // Printing and verifying...
  print, verify,

  // More powerful optimizations
  indvars, instcombine, sccp, adce, raise, mem2reg,

  // Instrumentation
  trace, tracem, paths,

  // Interprocedural optimizations...
  globaldce, swapstructs, sortstructs,
};


// New template functions - Provide functions that return passes of specified
// types, with specified arguments...
//
template<class PassClass>
Pass *New() {
  return new PassClass();
}

template<class PassClass, typename ArgTy1, ArgTy1 Arg1>
Pass *New() {
  return new PassClass(Arg1);
}

template<class PassClass, typename ArgTy1, ArgTy1 Arg1, 
                          typename ArgTy2, ArgTy1 Arg2>
Pass *New() {
  return new PassClass(Arg1, Arg2);
}

static Pass *NewPrintMethodPass() {
  return new PrintMethodPass("Current Method: \n", &cerr);
}

// OptTable - Correlate enum Opts to Pass constructors...
//
struct {
  enum Opts OptID;
  Pass * (*PassCtor)();
} OptTable[] = {
  { dce        , New<DeadCodeElimination> },
  { constprop  , New<ConstantPropogation> }, 
  { inlining   , New<MethodInlining> },
  { constmerge , New<ConstantMerge> },
  { strip      , New<SymbolStripping> },
  { mstrip     , New<FullSymbolStripping> },
  { mergereturn, New<UnifyMethodExitNodes> },

  { indvars    , New<InductionVariableSimplify> },
  { instcombine, New<InstructionCombining> },
  { sccp       , New<SCCPPass> },
  { adce       , New<AgressiveDCE> },
  { raise      , New<RaisePointerReferences> },
  { mem2reg    , newPromoteMemoryToRegister },

  { trace      , New<InsertTraceCode, bool, true, bool, true> },
  { tracem     , New<InsertTraceCode, bool, false, bool, true> },
  { paths      , New<ProfilePaths> },
  { print      , NewPrintMethodPass },
  { verify     , createVerifierPass },
  { raiseallocs, New<RaiseAllocations> },
  { cleangcc   , New<CleanupGCCOutput> },
  { globaldce  , New<GlobalDCE> },
  { swapstructs, New<SimpleStructMutation, SimpleStructMutation::Transform,
                     SimpleStructMutation::SwapElements>},
  { sortstructs, New<SimpleStructMutation, SimpleStructMutation::Transform,
                     SimpleStructMutation::SortElements>},
};

// Command line option handling code...
//
cl::String InputFilename ("", "Load <arg> file to optimize", cl::NoFlags, "-");
cl::String OutputFilename("o", "Override output filename", cl::NoFlags, "");
cl::Flag   Force         ("f", "Overwrite output files", cl::NoFlags, false);
cl::Flag   PrintEachXForm("p", "Print module after each transformation");
cl::Flag   Quiet         ("q", "Don't print modifying pass names", 0, false);
cl::Alias  QuietA        ("quiet", "Alias for -q", cl::NoFlags, Quiet);
cl::EnumList<enum Opts> OptimizationList(cl::NoFlags,
  clEnumVal(dce        , "Dead Code Elimination"),
  clEnumVal(constprop  , "Simple constant propogation"),
 clEnumValN(inlining   , "inline", "Method integration"),
  clEnumVal(constmerge , "Merge identical global constants"),
  clEnumVal(strip      , "Strip symbols"),
  clEnumVal(mstrip     , "Strip module symbols"),
  clEnumVal(mergereturn, "Unify method exit nodes"),

  clEnumVal(indvars    , "Simplify Induction Variables"),
  clEnumVal(instcombine, "Combine redundant instructions"),
  clEnumVal(sccp       , "Sparse Conditional Constant Propogation"),
  clEnumVal(adce       , "Agressive DCE"),
  clEnumVal(mem2reg    , "Promote alloca locations to registers"),

  clEnumVal(globaldce  , "Remove unreachable globals"),
  clEnumVal(swapstructs, "Swap structure types around"),
  clEnumVal(sortstructs, "Sort structure elements"),

  clEnumVal(raiseallocs, "Raise allocations from calls to instructions"),
  clEnumVal(cleangcc   , "Cleanup GCC Output"),
  clEnumVal(raise      , "Raise to Higher Level"),
  clEnumVal(trace      , "Insert BB & Method trace code"),
  clEnumVal(tracem     , "Insert Method trace code only"),
  clEnumVal(paths      , "Insert path profiling instrumentation"),
  clEnumVal(print      , "Print working method to stderr"),
  clEnumVal(verify     , "Verify module is well formed"),
0);



int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv,
			      " llvm .bc -> .bc modular optimizer\n");

  // Load the input module...
  std::auto_ptr<Module> M(ParseBytecodeFile(InputFilename));
  if (M.get() == 0) {
    cerr << "bytecode didn't read correctly.\n";
    return 1;
  }

  // Figure out what stream we are supposed to write to...
  std::ostream *Out = &std::cout;  // Default to printing to stdout...
  if (OutputFilename != "") {
    if (!Force && std::ifstream(OutputFilename.c_str())) {
      // If force is not specified, make sure not to overwrite a file!
      cerr << "Error opening '" << OutputFilename << "': File exists!\n"
           << "Use -f command line argument to force output\n";
      return 1;
    }
    Out = new std::ofstream(OutputFilename.c_str());

    if (!Out->good()) {
      cerr << "Error opening " << OutputFilename << "!\n";
      return 1;
    }
  }

  // Create a PassManager to hold and optimize the collection of passes we are
  // about to build...
  //
  PassManager Passes;

  // Create a new optimization pass for each one specified on the command line
  for (unsigned i = 0; i < OptimizationList.size(); ++i) {
    enum Opts Opt = OptimizationList[i];
    for (unsigned j = 0; j < sizeof(OptTable)/sizeof(OptTable[0]); ++j)
      if (Opt == OptTable[j].OptID) {
        Passes.add(OptTable[j].PassCtor());
        break;
      }

    if (PrintEachXForm)
      Passes.add(new PrintModulePass(&std::cerr));
  }

  // Check that the module is well formed on completion of optimization
  Passes.add(createVerifierPass());

  // Write bytecode out to disk or cout as the last step...
  Passes.add(new WriteBytecodePass(Out, Out != &std::cout));

  // Now that we have all of the passes ready, run them.
  if (Passes.run(M.get()) && !Quiet)
    cerr << "Program modified.\n";

  return 0;
}

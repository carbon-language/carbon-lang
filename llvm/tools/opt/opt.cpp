//===----------------------------------------------------------------------===//
// LLVM 'OPT' UTILITY 
//
// Optimizations may be specified an arbitrary number of times on the command
// line, they are run in the order specified.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Bytecode/Writer.h"
#include "llvm/Optimizations/AllOpts.h"
#include "llvm/Transforms/Instrumentation/TraceValues.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Transforms/ConstantMerge.h"
#include "llvm/Transforms/CleanupGCCOutput.h"
#include "llvm/Transforms/LevelChange.h"
#include "llvm/Transforms/IPO/SimpleStructMutation.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/InstructionCombining.h"
#include "Support/CommandLine.h"
#include <fstream>
#include <memory>

enum Opts {
  // Basic optimizations
  dce, constprop, inlining, constmerge, strip, mstrip,

  // Miscellaneous Transformations
  trace, tracem, print, cleangcc,

  // More powerful optimizations
  indvars, instcombine, sccp, adce, raise,

  // Interprocedural optimizations...
  globaldce, swapstructs, sortstructs,
};

struct {
  enum Opts OptID;
  Pass *ThePass;
} OptTable[] = {
  { dce        , new opt::DeadCodeElimination() },
  { constprop  , new opt::ConstantPropogation() }, 
  { inlining   , new opt::MethodInlining() },
  { constmerge , new ConstantMerge() },
  { strip      , new opt::SymbolStripping() },
  { mstrip     , new opt::FullSymbolStripping() },
  { indvars    , new InductionVariableSimplify() },
  { instcombine, new InstructionCombining() },
  { sccp       , new opt::SCCPPass() },
  { adce       , new opt::AgressiveDCE() },
  { raise      , new RaisePointerReferences() },
  { trace      , new InsertTraceCode(true, true) },
  { tracem     , new InsertTraceCode(false, true) },
  { print      , new PrintMethodPass("Current Method: \n",&cerr) },
  { cleangcc   , new CleanupGCCOutput() },
  { globaldce  , new GlobalDCE() },
  { swapstructs, new SimpleStructMutation(SimpleStructMutation::SwapElements) },
  { sortstructs, new SimpleStructMutation(SimpleStructMutation::SortElements) },
};

cl::String InputFilename ("", "Load <arg> file to optimize", cl::NoFlags, "-");
cl::String OutputFilename("o", "Override output filename", cl::NoFlags, "");
cl::Flag   Force         ("f", "Overwrite output files", cl::NoFlags, false);
cl::Flag   Quiet         ("q", "Don't print modifying pass names", 0, false);
cl::Alias  QuietA        ("quiet", "Alias for -q", cl::NoFlags, Quiet);
cl::EnumList<enum Opts> OptimizationList(cl::NoFlags,
  clEnumVal(dce        , "Dead Code Elimination"),
  clEnumVal(constprop  , "Simple Constant Propogation"),
 clEnumValN(inlining   , "inline", "Method Integration"),
  clEnumVal(constmerge , "Merge identical global constants"),
  clEnumVal(strip      , "Strip Symbols"),
  clEnumVal(mstrip     , "Strip Module Symbols"),
  clEnumVal(indvars    , "Simplify Induction Variables"),
  clEnumVal(instcombine, "Simplify Induction Variables"),
  clEnumVal(sccp       , "Sparse Conditional Constant Propogation"),
  clEnumVal(adce       , "Agressive DCE"),

  clEnumVal(globaldce  , "Remove unreachable globals"),
  clEnumVal(swapstructs, "Swap structure types around"),
  clEnumVal(sortstructs, "Sort structure elements"),

  clEnumVal(cleangcc   , "Cleanup GCC Output"),
  clEnumVal(raise      , "Raise to Higher Level"),
  clEnumVal(trace      , "Insert BB & Method trace code"),
  clEnumVal(tracem     , "Insert Method trace code only"),
  clEnumVal(print      , "Print working method to stderr"),
0);

static void RunOptimization(Module *M, enum Opts Opt) {
  for (unsigned j = 0; j < sizeof(OptTable)/sizeof(OptTable[0]); ++j)
    if (Opt == OptTable[j].OptID) {
      if (OptTable[j].ThePass->run(M) && !Quiet)
	cerr << OptimizationList.getArgName(Opt)
	     << " pass made modifications!\n";
      return;
    }
  
  cerr << "Optimization tables inconsistent!!\n";
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv,
			      " llvm .bc -> .bc modular optimizer\n");
  std::auto_ptr<Module> M(ParseBytecodeFile(InputFilename));
  if (M.get() == 0) {
    cerr << "bytecode didn't read correctly.\n";
    return 1;
  }

  PassManager Passes;

  // Run all of the optimizations specified on the command line
  for (unsigned i = 0; i < OptimizationList.size(); ++i)
    RunOptimization(M.get(), OptimizationList[i]);

  std::ostream *Out = &std::cout;  // Default to printing to stdout...
  if (OutputFilename != "") {
    if (!Force && !std::ifstream(OutputFilename.c_str())) {
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

  // Okay, we're done now... write out result...
  WriteBytecodeToFile(M.get(), *Out);

  if (Out != &cout) delete Out;
  return 0;
}

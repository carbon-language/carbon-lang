//===----------------------------------------------------------------------===//
// LLVM 'OPT' UTILITY 
//
// Optimizations may be specified an arbitrary number of times on the command
// line, they are run in the order specified.
//
//===----------------------------------------------------------------------===//

#include <iostream.h>
#include <fstream.h>
#include "llvm/Module.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Bytecode/Writer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Optimizations/AllOpts.h"
#include "llvm/Transforms/Instrumentation/TraceValues.h"
#include "llvm/Transforms/PrintModulePass.h"

using namespace opt;

enum Opts {
  // Basic optimizations
  dce, constprop, inlining, strip, mstrip,

  // Miscellaneous Transformations
  trace, tracem, print,

  // More powerful optimizations
  indvars, sccp, adce, raise,
};

struct {
  enum Opts OptID;
  Pass *ThePass;
} OptTable[] = {
  { dce      , new opt::DeadCodeElimination() },
  { constprop, new opt::ConstantPropogation() }, 
  { inlining , new opt::MethodInlining() },
  { strip    , new opt::SymbolStripping() },
  { mstrip   , new opt::FullSymbolStripping() },
  { indvars  , new opt::InductionVariableCannonicalize() },
  { sccp     , new opt::SCCPPass() },
  { adce     , new opt::AgressiveDCE() },
  { raise    , new opt::RaiseRepresentation() },
  { trace    , new InsertTraceCode(true, true) },
  { tracem   , new InsertTraceCode(false, true) },
  { print    , new PrintModulePass("Current Method: \n",&cerr) },
};

cl::String InputFilename ("", "Load <arg> file to optimize", cl::NoFlags, "-");
cl::String OutputFilename("o", "Override output filename", cl::NoFlags, "");
cl::Flag   Force         ("f", "Overwrite output files", cl::NoFlags, false);
cl::Flag   Quiet         ("q", "Don't print modifying pass names", 0, false);
cl::Alias  QuietA        ("quiet", "Alias for -q", cl::NoFlags, Quiet);
cl::EnumList<enum Opts> OptimizationList(cl::NoFlags,
  clEnumVal(dce      , "Dead Code Elimination"),
  clEnumVal(constprop, "Simple Constant Propogation"),
 clEnumValN(inlining , "inline", "Method Integration"),
  clEnumVal(strip    , "Strip Symbols"),
  clEnumVal(mstrip   , "Strip Module Symbols"),
  clEnumVal(indvars  , "Simplify Induction Variables"),
  clEnumVal(sccp     , "Sparse Conditional Constant Propogation"),
  clEnumVal(adce     , "Agressive DCE"),
  clEnumVal(raise    , "Raise to Higher Level"),
  clEnumVal(trace    , "Insert BB & Method trace code"),
  clEnumVal(tracem   , "Insert Method trace code only"),
  clEnumVal(print    , "Print working method to stderr"),
0);


int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv,
			      " llvm .bc -> .bc modular optimizer\n");
 
  Module *C = ParseBytecodeFile(InputFilename);
  if (C == 0) {
    cerr << "bytecode didn't read correctly.\n";
    return 1;
  }

  for (unsigned i = 0; i < OptimizationList.size(); ++i) {
    enum Opts Opt = OptimizationList[i];

    unsigned j;
    for (j = 0; j < sizeof(OptTable)/sizeof(OptTable[0]); ++j) {
      if (Opt == OptTable[j].OptID) {
        if (OptTable[j].ThePass->run(C) && !Quiet)
          cerr << OptimizationList.getArgName(Opt)
	       << " pass made modifications!\n";
        break;
      }
    }

    if (j == sizeof(OptTable)/sizeof(OptTable[0])) 
      cerr << "Optimization tables inconsistent!!\n";
  }

  ostream *Out = &cout;  // Default to printing to stdout...
  if (OutputFilename != "") {
    Out = new ofstream(OutputFilename.c_str(), 
                       (Force ? 0 : ios::noreplace)|ios::out);
    if (!Out->good()) {
      cerr << "Error opening " << OutputFilename << "!\n";
      delete C;
      return 1;
    }
  }

  // Okay, we're done now... write out result...
  WriteBytecodeToFile(C, *Out);
  delete C;

  if (Out != &cout) delete Out;
  return 0;
}

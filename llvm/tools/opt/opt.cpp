//===------------------------------------------------------------------------===
// LLVM 'OPT' UTILITY 
//
// This utility may be invoked in the following manner:
//  opt --help               - Output information about command line switches
//  opt [options] -dce       - Run a dead code elimination pass on input 
//                             bytecodes
//  opt [options] -constprop - Run a constant propogation pass on input 
//                             bytecodes
//  opt [options] -inline    - Run a method inlining pass on input bytecodes
//  opt [options] -strip     - Strip symbol tables out of methods
//  opt [options] -mstrip    - Strip module & method symbol tables
//
// Optimizations may be specified an arbitrary number of times on the command
// line, they are run in the order specified.
//
// TODO: Add a -all option to keep applying all optimizations until the program
//       stops permuting.
//
//===------------------------------------------------------------------------===

#include <iostream.h>
#include <fstream.h>
#include "llvm/Module.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Bytecode/Writer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Optimizations/AllOpts.h"

using namespace opt;

enum Opts {
  // Basic optimizations
  dce, constprop, inlining, strip, mstrip,

  // More powerful optimizations
  indvars, sccp, cpm, adce, raise,
};

struct {
  enum Opts OptID;
  bool (*OptPtr)(Module *C);
} OptTable[] = {
  { dce      , DoDeadCodeElimination },
  { constprop, DoConstantPropogation }, 
  { inlining , DoMethodInlining      },
  { strip    , DoSymbolStripping     },
  { mstrip   , DoFullSymbolStripping },
  { indvars  , DoInductionVariableCannonicalize },
  { sccp     , DoSCCP                },
  { cpm      , DoConstantPoolMerging },
  { adce     , DoADCE                },
  { raise    , DoRaiseRepresentation },
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
  clEnumVal(cpm      , "Constant Pool Merging"),
  clEnumVal(adce     , "Agressive DCE"),
  clEnumVal(raise    , "Raise to Higher Level"),
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
        if (OptTable[j].OptPtr(C) && !Quiet)
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

//===------------------------------------------------------------------------===
// LLVM 'Analyze' UTILITY 
//
// This utility is designed to print out the results of running various analysis
// passes on a program.  This is useful for understanding a program, or for 
// debugging an analysis pass.
//
//  analyze --help           - Output information about command line switches
//  analyze --quiet          - Do not print analysis name before output
//
//===------------------------------------------------------------------------===

#include <iostream>
#include "llvm/Module.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Assembly/Parser.h"
#include "llvm/Tools/CommandLine.h"
#include "llvm/Analysis/Writer.h"

#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/IntervalPartition.h"

static void PrintIntervalPartition(const Method *M) {
  cout << cfg::IntervalPartition((Method*)M);
}

static void PrintDominatorSets(const Method *M) {
  cout << cfg::DominatorSet(M);
}
static void PrintImmediateDominators(const Method *M) {
  cout << cfg::ImmediateDominators(M);
}
static void PrintDominatorTree(const Method *M) {
  cout << cfg::DominatorTree(M);
}
static void PrintDominanceFrontier(const Method *M) {
  cout << cfg::DominanceFrontier(M);
}

struct {
  const string ArgName, Name;
  void (*AnPtr)(const Method *M);
} AnTable[] = {
  { "-intervals"    , "Interval Partition"  , PrintIntervalPartition },
  { "-domset"       , "Dominator Sets"      , PrintDominatorSets },
  { "-immdom"       , "Immediate Dominators", PrintImmediateDominators },
  { "-domtree"      , "Dominator Tree"      , PrintDominatorTree },
  { "-domfrontier"  , "Dominance Frontier"  , PrintDominanceFrontier },
};

int main(int argc, char **argv) {
  ToolCommandLine Options(argc, argv, false);
  bool Quiet = false;

  for (int i = 1; i < argc; i++) {
    if (string(argv[i]) == string("--help")) {
      cerr << argv[0] << " usage:\n"
           << "  " << argv[0] << " --help\t - Print this usage information\n"
	   << "\t  --quiet\t - Do not print optimization name before output\n";
      for (unsigned j = 0; j < sizeof(AnTable)/sizeof(AnTable[0]); ++j) {
	cerr << "\t   " << AnTable[j].ArgName << "\t - Print " 
	     << AnTable[j].Name << endl;
      }
      return 1;
    } else if (string(argv[i]) == string("-q") ||
	       string(argv[i]) == string("--quiet")) {
      Quiet = true; argv[i] = 0;
    }
  }
  
  const Module *C = ParseBytecodeFile(Options.getInputFilename());
  if (C == 0) {
    C = ParseAssemblyFile(Options);
    if (C == 0) {
      cerr << "Input file didn't read correctly.\n";
      return 1;
    }
  }

  // Loop over all of the methods in the module...
  for (Module::const_iterator I = C->begin(), E = C->end(); I != E; ++I) {
    const Method *M = *I;

    // Loop over all of the optimizations to be run...
    for (int i = 1; i < argc; i++) {
      if (argv[i] == 0) continue;
      unsigned j;
      for (j = 0; j < sizeof(AnTable)/sizeof(AnTable[0]); j++) {
	if (string(argv[i]) == AnTable[j].ArgName) {
	  if (!Quiet)
	    cerr << "Running: " << AnTable[j].Name << " analysis!\n";
	  AnTable[j].AnPtr(M);
	  break;
	}
      }
      
      if (j == sizeof(AnTable)/sizeof(AnTable[0])) 
	cerr << "'" << argv[i] << "' argument unrecognized: ignored\n";
    }
  }

  delete C;
  return 0;
}
